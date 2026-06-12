from __future__ import annotations

import csv
import json
import random
import time
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .baseline import BaselineLoss, make_catalog_model, make_gaussian_heatmap, train_step
from .decode import decode_predictions
from .io import write_prediction_catalog
from .schema import BANDS, PredictionRecord


@dataclass(frozen=True)
class DatasetSampleIndex:
    cutout_id: str
    shard: str
    shard_sample_index: int
    field_id: str
    center_source_id: str
    ra: float
    dec: float
    x: float
    y: float
    origin_x: int
    origin_y: int


@dataclass(frozen=True)
class LossConfig:
    variant: str
    center_weight: float = 1.0
    photometry_weight: float = 1.0
    multiband_weight: float = 0.05
    psf_reconstruction_weight: float = 0.2
    class_weight: float = 0.5


class NpzCutoutDataset(Dataset):
    """Read pilot NPZ cutout shards and build point-supervision targets."""

    def __init__(
        self,
        dataset_dir: str | Path,
        heatmap_sigma: float = 1.5,
        magnitude_zeropoint: float = 22.5,
        split_path: str | Path | None = None,
        split_name: str | None = None,
        limit_samples: int | None = None,
        shard_cache_size: int = 0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.heatmap_sigma = heatmap_sigma
        self.magnitude_zeropoint = magnitude_zeropoint
        self.shard_cache_size = max(0, int(shard_cache_size))
        self._shard_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self.metadata = _load_optional_json(self.dataset_dir / "metadata.json")
        self.samples = _load_manifest_index(self.dataset_dir / "manifest.csv")
        if split_path is not None:
            if split_name is None:
                raise ValueError("split_name is required when split_path is provided")
            split_ids = _load_split_ids(split_path, split_name)
            self.samples = [sample for sample in self.samples if sample.center_source_id in split_ids]
        if limit_samples is not None:
            if limit_samples < 0:
                raise ValueError("limit_samples must be non-negative")
            self.samples = self.samples[:limit_samples]
        if not self.samples:
            raise ValueError(f"dataset manifest has no samples: {self.dataset_dir / 'manifest.csv'}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sample = self.samples[index]
        shard = self._read_shard(sample.shard)
        local_index = sample.shard_sample_index
        image_np = np.asarray(shard["images"][local_index], dtype=np.float32)
        offsets = shard["source_offsets"]
        start = int(offsets[local_index])
        end = int(offsets[local_index + 1])
        source_x = np.asarray(shard["source_x"][start:end], dtype=np.float32)
        source_y = np.asarray(shard["source_y"][start:end], dtype=np.float32)
        source_label = np.asarray(shard["source_label"][start:end], dtype=np.int16)
        if "source_weight" in shard:
            source_weight = np.asarray(shard["source_weight"][start:end], dtype=np.float32)
        else:
            source_weight = np.ones(end - start, dtype=np.float32)
        source_mags = {
            band: np.asarray(shard[f"source_mag_{band}"][start:end], dtype=np.float32)
            for band in BANDS
            if f"source_mag_{band}" in shard
        }

        image = torch.from_numpy(image_np)
        _, height, width = image.shape
        points = [
            (float(x), float(y))
            for x, y, weight in zip(source_x, source_y, source_weight, strict=True)
            if weight > 0.0 and np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height
        ]
        target_heatmap = make_gaussian_heatmap(points, height, width, self.heatmap_sigma).unsqueeze(0)
        target_flux = torch.zeros_like(image)
        inverse_variance = torch.zeros_like(image)
        target_class_map = torch.full((height, width), -100, dtype=torch.long)

        for source_index, (x_float, y_float, label_value) in enumerate(
            zip(source_x, source_y, source_label, strict=True)
        ):
            weight = float(source_weight[source_index])
            if weight <= 0.0:
                continue
            if not np.isfinite(x_float) or not np.isfinite(y_float):
                continue
            x = int(round(float(x_float)))
            y = int(round(float(y_float)))
            if not (0 <= x < width and 0 <= y < height):
                continue
            class_index = int(label_value) - 1
            if class_index in {0, 1}:
                target_class_map[y, x] = class_index
            for band_index, band in enumerate(BANDS[: image.shape[0]]):
                mag_values = source_mags.get(band)
                if mag_values is None:
                    continue
                flux = _mag_to_flux(float(mag_values[source_index]), self.magnitude_zeropoint)
                if flux > 0.0:
                    target_flux[band_index, y, x] = flux
                    inverse_variance[band_index, y, x] = weight

        return {
            "image": image,
            "target_heatmap": target_heatmap,
            "target_flux": target_flux,
            "inverse_variance": inverse_variance,
            "observed_image": image,
            "valid_mask": torch.ones(1, height, width, dtype=torch.float32),
            "target_class_map": target_class_map,
            "cutout_id": sample.cutout_id,
            "field_id": sample.field_id,
            "origin_ra": torch.tensor(sample.ra, dtype=torch.float64),
            "origin_dec": torch.tensor(sample.dec, dtype=torch.float64),
            "center_x": torch.tensor(sample.x, dtype=torch.float64),
            "center_y": torch.tensor(sample.y, dtype=torch.float64),
            "origin_x": torch.tensor(sample.origin_x, dtype=torch.int64),
            "origin_y": torch.tensor(sample.origin_y, dtype=torch.int64),
        }

    def shard_cache_stats(self) -> dict[str, int]:
        return {
            "enabled": int(self.shard_cache_size > 0),
            "size": len(self._shard_cache),
            "max_size": self.shard_cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def _read_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        if self.shard_cache_size <= 0:
            self._cache_misses += 1
            return _load_shard_arrays(self.dataset_dir / "shards" / shard_name)
        cached = self._shard_cache.get(shard_name)
        if cached is not None:
            self._cache_hits += 1
            self._shard_cache.move_to_end(shard_name)
            return cached
        self._cache_misses += 1
        arrays = _load_shard_arrays(self.dataset_dir / "shards" / shard_name)
        self._shard_cache[shard_name] = arrays
        self._shard_cache.move_to_end(shard_name)
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return arrays


class ShardBatchSampler(Sampler[list[int]]):
    """Yield batches grouped by NPZ shard while preserving epoch-level shuffle."""

    def __init__(self, samples: list[DatasetSampleIndex], batch_size: int, *, seed: int = 42, shuffle: bool = True):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        by_shard: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            by_shard.setdefault(sample.shard, []).append(index)
        self._shards = sorted(by_shard)
        self._indices_by_shard = {shard: by_shard[shard] for shard in self._shards}
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        shards = list(self._shards)
        if self.shuffle:
            rng.shuffle(shards)
        for shard in shards:
            indices = list(self._indices_by_shard[shard])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        batches = 0
        for indices in self._indices_by_shard.values():
            batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


def train_model(
    config_path: str | Path | None,
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    base_channels: int = 32,
    device: str = "cpu",
    seed: int = 42,
    heatmap_sigma: float = 1.5,
    psf_sigma: float = 1.3,
    psf_size: int = 9,
    split_path: str | Path | None = None,
    split_name: str | None = None,
    limit_samples: int | None = None,
    model_arch: str = "baseline",
    loader_mode: str = "sample",
    shard_cache_size: int = 0,
    num_workers: int = 0,
    pin_memory: bool | str = "auto",
    loss_variant: str = "full_psf_point_supervised",
    center_loss_weight: float | None = None,
    photometry_loss_weight: float | None = None,
    multiband_loss_weight: float | None = None,
    psf_reconstruction_loss_weight: float | None = None,
    class_loss_weight: float | None = None,
) -> dict[str, Any]:
    """Train the compact PSF-constrained baseline on prepared NPZ cutouts."""

    _seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = NpzCutoutDataset(
        dataset_dir,
        heatmap_sigma=heatmap_sigma,
        split_path=split_path,
        split_name=split_name,
        limit_samples=limit_samples,
        shard_cache_size=shard_cache_size,
    )
    device_obj = torch.device(device)
    pin_memory_enabled = _resolve_pin_memory(pin_memory, device_obj)
    loader = _make_train_loader(
        dataset,
        batch_size=batch_size,
        seed=seed,
        loader_mode=loader_mode,
        num_workers=num_workers,
        pin_memory=pin_memory_enabled,
    )
    config_payload = _load_config(config_path)
    loss_config = resolve_loss_config(
        config_payload,
        loss_variant=loss_variant,
        center_loss_weight=center_loss_weight,
        photometry_loss_weight=photometry_loss_weight,
        multiband_loss_weight=multiband_loss_weight,
        psf_reconstruction_loss_weight=psf_reconstruction_loss_weight,
        class_loss_weight=class_loss_weight,
    )
    model = make_catalog_model(model_arch, base_channels=base_channels).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = BaselineLoss(
        center_weight=loss_config.center_weight,
        photometry_weight=loss_config.photometry_weight,
        multiband_weight=loss_config.multiband_weight,
        psf_reconstruction_weight=loss_config.psf_reconstruction_weight,
        class_weight=loss_config.class_weight,
    )
    psf_kernel = gaussian_psf_kernel(psf_size, psf_sigma).to(device_obj)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    cuda_available = torch.cuda.is_available()
    cuda_training = device_obj.type == "cuda" and cuda_available
    device_name = _device_name(device_obj) if cuda_training else None
    if cuda_training:
        torch.cuda.reset_peak_memory_stats(device_obj)
    total_started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        epoch_losses: list[float] = []
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch - 1)
        for batch in loader:
            tensor_batch = _move_tensor_batch(batch, device_obj)
            tensor_batch["psf_kernel"] = psf_kernel
            loss_value = train_step(model, tensor_batch, optimizer, loss_fn)
            epoch_losses.append(loss_value)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_seconds = time.perf_counter() - epoch_started_at
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": mean_loss,
                "seconds": epoch_seconds,
                "samples_per_second": float(len(dataset) / epoch_seconds) if epoch_seconds > 0 else 0.0,
            }
        )
        if mean_loss <= best_loss:
            best_loss = mean_loss
            _save_checkpoint(
                output / "best.pt",
                model=model,
                config=config_payload,
                dataset_dir=str(Path(dataset_dir)),
                base_channels=base_channels,
                model_arch=model_arch,
                epoch=epoch,
                train_loss=mean_loss,
                heatmap_sigma=heatmap_sigma,
                loss_config=loss_config,
            )
    if cuda_training:
        torch.cuda.synchronize(device_obj)
    total_seconds = time.perf_counter() - total_started_at
    memory_allocated_peak_mb = _cuda_peak_memory_mb(device_obj, "allocated") if cuda_training else None
    memory_reserved_peak_mb = _cuda_peak_memory_mb(device_obj, "reserved") if cuda_training else None

    report = {
        "status": "trained",
        "generated_at": _utc_now(),
        "config_path": str(config_path) if config_path is not None else None,
        "dataset_dir": str(Path(dataset_dir)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "base_channels": base_channels,
        "heatmap_sigma": heatmap_sigma,
        "model_arch": model_arch,
        "model_parameters": count_parameters(model),
        "loss": {
            "variant": loss_config.variant,
            "center_weight": loss_config.center_weight,
            "photometry_weight": loss_config.photometry_weight,
            "multiband_weight": loss_config.multiband_weight,
            "psf_reconstruction_weight": loss_config.psf_reconstruction_weight,
            "class_weight": loss_config.class_weight,
        },
        "device": str(device_obj),
        "cuda_available": cuda_available,
        "device_name": device_name,
        "memory_allocated_peak_mb": memory_allocated_peak_mb,
        "memory_reserved_peak_mb": memory_reserved_peak_mb,
        "best_train_loss": best_loss,
        "history": history,
        "total_seconds": total_seconds,
        "samples_per_second": float(len(dataset) * max(epochs, 0) / total_seconds) if total_seconds > 0 else 0.0,
        "checkpoint": str(output / "best.pt"),
        "split_path": str(split_path) if split_path is not None else None,
        "split_name": split_name,
        "limit_samples": limit_samples,
        "loader": {
            "mode": loader_mode,
            "num_workers": num_workers,
            "pin_memory": pin_memory_enabled,
            "persistent_workers": bool(num_workers > 0),
            "prefetch_factor": 2 if num_workers > 0 else None,
            "batch_count": len(loader),
            "shard_cache_size": shard_cache_size,
            "shard_cache_stats": dataset.shard_cache_stats(),
        },
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def predict_dataset(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 16,
    threshold: float = 0.5,
    nms_radius: int = 2,
    device: str = "cpu",
    pixel_scale_arcsec: float = 0.396,
    split_path: str | Path | None = None,
    split_name: str | None = None,
    max_detections_per_cutout: int | None = None,
    limit_samples: int | None = None,
    shard_cache_size: int = 0,
    num_workers: int = 0,
    pin_memory: bool | str = "auto",
) -> list[PredictionRecord]:
    """Run checkpoint inference on a prepared NPZ dataset and write prediction CSV."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model", {})
    model = make_catalog_model(
        str(model_config.get("model_arch") or model_config.get("name") or "baseline"),
        in_channels=int(model_config.get("in_channels", 5)),
        num_classes=int(model_config.get("num_classes", 2)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    dataset = NpzCutoutDataset(
        dataset_dir,
        split_path=split_path,
        split_name=split_name,
        limit_samples=limit_samples,
        shard_cache_size=shard_cache_size,
    )
    field_wcs = _load_r_band_wcs_by_field(dataset.metadata, {sample.field_id for sample in dataset.samples})
    loader = _make_eval_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_resolve_pin_memory(pin_memory, device_obj),
    )
    records: list[PredictionRecord] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device_obj)
            outputs = model(images)
            field_ids = list(batch["field_id"])
            origin_x = [float(value) for value in batch["origin_x"].tolist()]
            origin_y = [float(value) for value in batch["origin_y"].tolist()]

            def pixel_to_radec(
                batch_index: int,
                x: float,
                y: float,
                *,
                field_ids=field_ids,
                origin_x=origin_x,
                origin_y=origin_y,
            ) -> tuple[float, float]:
                wcs = field_wcs[field_ids[batch_index]]
                world = wcs.all_pix2world([[origin_x[batch_index] + x, origin_y[batch_index] + y]], 1)
                ra, dec = world[0]
                return float(ra), float(dec)

            records.extend(
                decode_predictions(
                    outputs,
                    cutout_ids=list(batch["cutout_id"]),
                    origin_radec=[
                        (float(ra), float(dec))
                        for ra, dec in zip(batch["origin_ra"].tolist(), batch["origin_dec"].tolist(), strict=True)
                    ],
                    pixel_to_radec=pixel_to_radec,
                    pixel_scale_arcsec=pixel_scale_arcsec,
                    threshold=threshold,
                    nms_radius=nms_radius,
                    max_detections_per_cutout=max_detections_per_cutout,
                )
            )
    write_prediction_catalog(records, output_path)
    return records


def gaussian_psf_kernel(size: int = 9, sigma: float = 1.3) -> Tensor:
    if size % 2 == 0:
        raise ValueError("PSF kernel size must be odd")
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, size, size)


def resolve_loss_config(
    config: Mapping[str, Any] | None = None,
    *,
    loss_variant: str = "full_psf_point_supervised",
    center_loss_weight: float | None = None,
    photometry_loss_weight: float | None = None,
    multiband_loss_weight: float | None = None,
    psf_reconstruction_loss_weight: float | None = None,
    class_loss_weight: float | None = None,
) -> LossConfig:
    """Resolve method ablation loss weights from config plus explicit overrides."""

    config_losses = {}
    if isinstance(config, Mapping):
        method = config.get("method", {})
        if isinstance(method, Mapping):
            maybe_losses = method.get("losses", {})
            if isinstance(maybe_losses, Mapping):
                config_losses = maybe_losses
    variant = str(loss_variant or config_losses.get("variant") or "full_psf_point_supervised")
    defaults = {
        "center_weight": float(config_losses.get("center_weight", 1.0)),
        "photometry_weight": float(config_losses.get("photometry_weight", 1.0)),
        "multiband_weight": float(config_losses.get("multiband_consistency_weight", 0.05)),
        "psf_reconstruction_weight": float(config_losses.get("psf_reconstruction_weight", 0.2)),
        "class_weight": float(config_losses.get("class_weight", 0.5)),
    }
    if variant == "full_psf_point_supervised":
        weights = defaults
    elif variant == "no_psf_reconstruction":
        weights = {**defaults, "psf_reconstruction_weight": 0.0}
    elif variant == "center_only":
        weights = {
            "center_weight": defaults["center_weight"],
            "photometry_weight": 0.0,
            "multiband_weight": 0.0,
            "psf_reconstruction_weight": 0.0,
            "class_weight": 0.0,
        }
    else:
        raise ValueError(f"unsupported loss_variant: {variant}")

    overrides = {
        "center_weight": center_loss_weight,
        "photometry_weight": photometry_loss_weight,
        "multiband_weight": multiband_loss_weight,
        "psf_reconstruction_weight": psf_reconstruction_loss_weight,
        "class_weight": class_loss_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            weights[key] = float(value)
    return LossConfig(variant=variant, **weights)


def _load_shard_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as shard:
        return {name: np.asarray(shard[name]) for name in shard.files}


def _make_train_loader(
    dataset: NpzCutoutDataset,
    *,
    batch_size: int,
    seed: int,
    loader_mode: str,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if loader_mode == "sample":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    if loader_mode == "shard_grouped":
        return DataLoader(
            dataset,
            batch_sampler=ShardBatchSampler(dataset.samples, batch_size=batch_size, seed=seed, shuffle=True),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    raise ValueError(f"unsupported loader_mode: {loader_mode}")


def _make_eval_loader(
    dataset: NpzCutoutDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _resolve_pin_memory(pin_memory: bool | str, device: torch.device) -> bool:
    if isinstance(pin_memory, bool):
        return pin_memory
    normalized = str(pin_memory).lower()
    if normalized == "auto":
        return device.type == "cuda"
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("pin_memory must be 'auto', 'true', 'false', True, or False")


def _device_name(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    return str(torch.cuda.get_device_name(device))


def _cuda_peak_memory_mb(device: torch.device, kind: str) -> float:
    if kind == "allocated":
        bytes_value = torch.cuda.max_memory_allocated(device)
    elif kind == "reserved":
        bytes_value = torch.cuda.max_memory_reserved(device)
    else:
        raise ValueError(f"unsupported CUDA memory peak kind: {kind}")
    return float(bytes_value) / float(1024 * 1024)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _load_manifest_index(path: Path) -> list[DatasetSampleIndex]:
    if not path.exists():
        raise FileNotFoundError(f"missing dataset manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        DatasetSampleIndex(
            cutout_id=row["cutout_id"],
            shard=row["shard"],
            shard_sample_index=int(row["shard_sample_index"]),
            center_source_id=row["center_source_id"],
            field_id=row.get("field_id") or row["cutout_id"].split("__", 1)[0],
            ra=float(row["ra"]),
            dec=float(row["dec"]),
            x=_manifest_float(row, "x"),
            y=_manifest_float(row, "y"),
            origin_x=_manifest_int(row, "origin_x"),
            origin_y=_manifest_int(row, "origin_y"),
        )
        for row in rows
    ]


def _manifest_float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _manifest_int(row: Mapping[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def _load_r_band_wcs_by_field(metadata: Mapping[str, Any], required_field_ids: set[str]):
    try:
        from astropy.io.fits import Header
        from astropy.wcs import WCS
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise ImportError("prediction from prepared datasets requires astropy; install the 'baseline' extra") from exc

    fields = metadata.get("fields")
    if not isinstance(fields, list):
        raise ValueError("dataset metadata is missing r-band WCS metadata required for prediction: no fields list")

    by_field_id = {
        str(field.get("field_id")): field for field in fields if isinstance(field, Mapping) and field.get("field_id")
    }
    errors: list[str] = []
    valid_headers: dict[str, Mapping[str, Any]] = {}
    for field_id in sorted(required_field_ids):
        field = by_field_id.get(field_id)
        if not isinstance(field, Mapping):
            errors.append(f"{field_id} (missing field metadata)")
            continue
        headers = field.get("headers")
        if not isinstance(headers, Mapping):
            errors.append(f"{field_id} (missing headers)")
            continue
        r_header = headers.get("r")
        if not isinstance(r_header, Mapping):
            errors.append(f"{field_id} (missing r-band header)")
            continue
        missing_keys = _missing_wcs_header_keys(r_header)
        if missing_keys:
            errors.append(f"{field_id} (missing WCS keys: {', '.join(missing_keys)})")
            continue
        valid_headers[field_id] = r_header

    if errors:
        suffix = "; ".join(errors[:5])
        if len(errors) > 5:
            suffix += f"; ... {len(errors) - 5} more"
        raise ValueError(f"dataset metadata is missing r-band WCS metadata required for prediction: {suffix}")

    wcs_by_field = {}
    for field_id, header_values in valid_headers.items():
        header = Header()
        for key, value in header_values.items():
            header[str(key)] = value
        wcs_by_field[field_id] = WCS(header)
    return wcs_by_field


def _missing_wcs_header_keys(header: Mapping[str, Any]) -> list[str]:
    required = {"CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"}
    missing = required - set(header)
    has_cd = {"CD1_1", "CD1_2", "CD2_1", "CD2_2"}.issubset(header)
    has_cdelt = {"CDELT1", "CDELT2"}.issubset(header)
    if not has_cd and not has_cdelt:
        missing.add("CD matrix or CDELT")
    return sorted(missing)


def _load_split_ids(path: str | Path, split_name: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload.get("splits", {})
    if split_name not in splits:
        raise ValueError(f"split {split_name!r} not found in {path}")
    return {str(source_id) for source_id in splits[split_name]}


def _move_tensor_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    dataset_dir: str,
    base_channels: int,
    model_arch: str,
    epoch: int,
    train_loss: float,
    heatmap_sigma: float,
    loss_config: LossConfig,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "model": {
                "name": "AstronomyAwareBaseline" if model_arch == "baseline" else "AstronomyAwareUNetLite",
                "model_arch": model_arch,
                "in_channels": 5,
                "num_classes": 2,
                "base_channels": base_channels,
                "parameters": count_parameters(model),
            },
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "loss": {
                "variant": loss_config.variant,
                "center_weight": loss_config.center_weight,
                "photometry_weight": loss_config.photometry_weight,
                "multiband_weight": loss_config.multiband_weight,
                "psf_reconstruction_weight": loss_config.psf_reconstruction_weight,
                "class_weight": loss_config.class_weight,
            },
            "dataset_dir": dataset_dir,
            "epoch": epoch,
            "train_loss": train_loss,
            "heatmap_sigma": heatmap_sigma,
        },
        path,
    )


def _mag_to_flux(magnitude: float, zeropoint: float) -> float:
    if not np.isfinite(magnitude):
        return 0.0
    return float(10.0 ** (-0.4 * (magnitude - zeropoint)))


def _load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
