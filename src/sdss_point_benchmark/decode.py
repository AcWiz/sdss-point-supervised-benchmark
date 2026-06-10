from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from .schema import BANDS, PredictionRecord


class CatalogHeadOutput(Protocol):
    center_heatmap: Tensor
    class_logits: Tensor
    flux: Tensor
    shape_params: Tensor


def decode_predictions(
    outputs: CatalogHeadOutput,
    cutout_ids: list[str],
    origin_radec: list[tuple[float, float]] | None = None,
    pixel_to_radec: Callable[[int, float, float], tuple[float, float]] | None = None,
    pixel_scale_arcsec: float = 0.396,
    threshold: float = 0.5,
    nms_radius: int = 2,
    max_detections_per_cutout: int | None = None,
    class_names: tuple[str, ...] = ("star", "galaxy"),
) -> list[PredictionRecord]:
    """Decode center heatmaps into catalog records using local-max NMS."""

    scores = torch.sigmoid(outputs.center_heatmap.detach())
    pooled = F.max_pool2d(scores, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius)
    keep = (scores == pooled) & (scores >= threshold)
    records: list[PredictionRecord] = []
    batch_size = scores.shape[0]
    if len(cutout_ids) != batch_size:
        raise ValueError("cutout_ids length must match output batch size")
    if origin_radec is None:
        origin_radec = [(0.0, 0.0)] * batch_size
    if len(origin_radec) != batch_size:
        raise ValueError("origin_radec length must match output batch size")

    for batch_index in range(batch_size):
        if max_detections_per_cutout is not None:
            if max_detections_per_cutout < 0:
                raise ValueError("max_detections_per_cutout must be non-negative")
            flat_scores = scores[batch_index, 0].masked_fill(~keep[batch_index, 0], float("-inf")).flatten()
            k = min(max_detections_per_cutout, int(keep[batch_index, 0].sum().detach().cpu()))
            if k == 0:
                candidates = []
            else:
                values, flat_indices = torch.topk(flat_scores, k)
                width = int(scores.shape[-1])
                candidates = [
                    (float(score), int(flat_index // width), int(flat_index % width))
                    for score, flat_index in zip(values.detach().cpu(), flat_indices.detach().cpu(), strict=True)
                ]
        else:
            ys, xs = torch.where(keep[batch_index, 0])
            candidates = sorted(
                [(float(scores[batch_index, 0, y, x]), int(y), int(x)) for y, x in zip(ys, xs, strict=True)],
                reverse=True,
            )
        origin_ra, origin_dec = origin_radec[batch_index]
        for rank, (score, y, x) in enumerate(candidates):
            class_index = int(outputs.class_logits[batch_index, :, y, x].argmax().detach().cpu())
            label = class_names[class_index] if class_index < len(class_names) else str(class_index)
            flux_values = {
                f"flux_{band}": float(outputs.flux[batch_index, band_index, y, x].detach().cpu())
                for band_index, band in enumerate(BANDS[: outputs.flux.shape[1]])
            }
            if pixel_to_radec is None:
                ra = origin_ra + x * pixel_scale_arcsec / 3600.0
                dec = origin_dec + y * pixel_scale_arcsec / 3600.0
            else:
                ra, dec = pixel_to_radec(batch_index, float(x), float(y))
            records.append(
                PredictionRecord(
                    prediction_id=f"{cutout_ids[batch_index]}__pred{rank:04d}",
                    cutout_id=cutout_ids[batch_index],
                    ra=float(ra),
                    dec=float(dec),
                    label=label,
                    score=score,
                    x=float(x),
                    y=float(y),
                    size=float(outputs.shape_params[batch_index, 0, y, x].detach().cpu()),
                    ellipticity=float(outputs.shape_params[batch_index, 1, y, x].detach().cpu()),
                    **flux_values,
                )
            )
    return records
