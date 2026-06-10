from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decode import decode_predictions as decode_predictions


@dataclass(frozen=True)
class BaselineOutput:
    center_heatmap: Tensor
    class_logits: Tensor
    flux: Tensor
    shape_params: Tensor


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    parts: dict[str, Tensor]


class AstronomyAwareBaseline(nn.Module):
    """Compact multiband baseline with detection, class, flux, and shape heads."""

    def __init__(self, in_channels: int = 5, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(in_channels, base_channels),
            _conv_block(base_channels, base_channels),
            _conv_block(base_channels, base_channels),
        )
        self.center_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.class_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.flux_head = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.shape_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, image: Tensor) -> BaselineOutput:
        features = self.encoder(image)
        return BaselineOutput(
            center_heatmap=self.center_head(features),
            class_logits=self.class_head(features),
            flux=F.softplus(self.flux_head(features)),
            shape_params=self.shape_head(features),
        )


class AstronomyAwareUNetLite(nn.Module):
    """Lightweight multiscale model with the same catalog-generation heads."""

    def __init__(self, in_channels: int = 5, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        self.stem = _res_block(in_channels, base_channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _res_block(base_channels, base_channels * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _res_block(base_channels * 2, base_channels * 4))
        self.bottleneck = _res_block(base_channels * 4, base_channels * 4)
        self.up1 = _res_block(base_channels * 6, base_channels * 2)
        self.up2 = _res_block(base_channels * 3, base_channels)
        self.center_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.class_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.flux_head = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.shape_head = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, image: Tensor) -> BaselineOutput:
        skip0 = self.stem(image)
        skip1 = self.down1(skip0)
        encoded = self.bottleneck(self.down2(skip1))
        up1 = F.interpolate(encoded, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat([up1, skip1], dim=1))
        up2 = F.interpolate(up1, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        features = self.up2(torch.cat([up2, skip0], dim=1))
        return BaselineOutput(
            center_heatmap=self.center_head(features),
            class_logits=self.class_head(features),
            flux=F.softplus(self.flux_head(features)),
            shape_params=self.shape_head(features),
        )


def make_catalog_model(
    model_arch: str = "baseline",
    *,
    in_channels: int = 5,
    num_classes: int = 2,
    base_channels: int = 32,
) -> nn.Module:
    """Build a catalog-generation model while keeping checkpoint names stable."""

    normalized = model_arch
    if normalized == "AstronomyAwareBaseline":
        normalized = "baseline"
    elif normalized == "AstronomyAwareUNetLite":
        normalized = "unet_lite"
    if normalized == "baseline":
        return AstronomyAwareBaseline(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    if normalized == "unet_lite":
        return AstronomyAwareUNetLite(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    raise ValueError(f"unsupported model_arch: {model_arch}")


class BaselineLoss(nn.Module):
    """Loss terms for point-supervised catalog generation."""

    def __init__(
        self,
        center_weight: float = 1.0,
        photometry_weight: float = 1.0,
        multiband_weight: float = 0.05,
        psf_reconstruction_weight: float = 0.0,
        class_weight: float = 0.0,
    ):
        super().__init__()
        self.center_weight = center_weight
        self.photometry_weight = photometry_weight
        self.multiband_weight = multiband_weight
        self.psf_reconstruction_weight = psf_reconstruction_weight
        self.class_weight = class_weight

    def forward(
        self,
        outputs: BaselineOutput,
        target_heatmap: Tensor,
        target_flux: Tensor,
        inverse_variance: Tensor,
        observed_image: Tensor | None = None,
        psf_kernel: Tensor | None = None,
        valid_mask: Tensor | None = None,
        target_class_map: Tensor | None = None,
    ) -> LossOutput:
        center_per_pixel = F.binary_cross_entropy_with_logits(
            outputs.center_heatmap,
            target_heatmap,
            reduction="none",
        )
        center = _masked_mean(center_per_pixel, valid_mask)
        photometry = _masked_mean((outputs.flux - target_flux) ** 2 * inverse_variance, valid_mask)
        multiband_consistency = _multiband_consistency(outputs.flux)
        total = (
            self.center_weight * center
            + self.photometry_weight * photometry
            + self.multiband_weight * multiband_consistency
        )
        parts = {
            "center": center.detach(),
            "photometry": photometry.detach(),
            "multiband_consistency": multiband_consistency.detach(),
        }

        if observed_image is not None and psf_kernel is not None and self.psf_reconstruction_weight:
            reconstructed = psf_convolve(outputs.flux, psf_kernel)
            psf_reconstruction = _masked_mean((reconstructed - observed_image) ** 2, valid_mask)
            total = total + self.psf_reconstruction_weight * psf_reconstruction
            parts["psf_reconstruction"] = psf_reconstruction.detach()

        if target_class_map is not None and self.class_weight:
            if (target_class_map != -100).any():
                classification = F.cross_entropy(outputs.class_logits, target_class_map, ignore_index=-100)
            else:
                classification = outputs.class_logits.new_tensor(0.0)
            total = total + self.class_weight * classification
            parts["classification"] = classification.detach()

        return LossOutput(
            total=total,
            parts=parts,
        )


def make_gaussian_heatmap(points: list[tuple[float, float]], height: int, width: int, sigma: float) -> Tensor:
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    heatmap = torch.zeros(height, width, dtype=torch.float32)
    for x, y in points:
        heatmap = torch.maximum(
            heatmap,
            torch.exp(-0.5 * (((xx.float() - x) / sigma) ** 2 + ((yy.float() - y) / sigma) ** 2)),
        )
    return heatmap


def psf_convolve(image: Tensor, psf_kernel: Tensor) -> Tensor:
    """Convolve each band independently with a shared or per-band PSF kernel."""

    channels = image.shape[1]
    if psf_kernel.ndim != 4:
        raise ValueError("psf_kernel must have shape (1, 1, k, k) or (channels, 1, k, k)")
    if psf_kernel.shape[0] == 1:
        kernel = psf_kernel.to(device=image.device, dtype=image.dtype).expand(channels, -1, -1, -1)
    elif psf_kernel.shape[0] == channels:
        kernel = psf_kernel.to(device=image.device, dtype=image.dtype)
    else:
        raise ValueError("psf_kernel first dimension must be 1 or match image channels")
    padding = psf_kernel.shape[-1] // 2
    return F.conv2d(image, kernel, padding=padding, groups=channels)


def train_step(
    model: nn.Module,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: BaselineLoss,
) -> float:
    """Run one supervised optimization step for benchmark smoke tests and examples."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["image"])
    loss = loss_fn(
        outputs,
        batch["target_heatmap"],
        batch["target_flux"],
        batch["inverse_variance"],
        observed_image=batch.get("observed_image"),
        psf_kernel=batch.get("psf_kernel"),
        valid_mask=batch.get("valid_mask"),
        target_class_map=batch.get("target_class_map"),
    )
    loss.total.backward()
    optimizer.step()
    return float(loss.total.detach().cpu())


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_channels),
        nn.SiLU(inplace=True),
    )


def _res_block(in_channels: int, out_channels: int) -> nn.Module:
    return _ResidualBlock(in_channels, out_channels)


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        return self.activation(self.main(image) + self.skip(image))


def _multiband_consistency(flux: Tensor) -> Tensor:
    if flux.shape[1] < 2:
        return flux.new_tensor(0.0)
    normalized = torch.log1p(flux)
    return (normalized[:, 1:] - normalized[:, :-1]).abs().mean()


def _masked_mean(values: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return values.mean()
    mask = valid_mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = torch.broadcast_to(mask, values.shape)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator
