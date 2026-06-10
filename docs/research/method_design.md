# PSF-Constrained Point-Supervised Cataloger

## Goal

Build a compact, reproducible method for SDSS-style source catalog generation
from weak point supervision. The method consumes native-frame ugriz cutouts and
predicts a source catalog with centroids, class labels, confidence scores,
multiband flux proxies, and shape proxies.

## Core Assumption

SDSS PhotoObj labels are useful weak supervision, not final truth. Training can
use catalog centers, coarse STAR/GALAXY labels, and noisy photometry. Evaluation
must report where the weak catalog is reliable and where synthetic injections or
deeper cross-matches are required.

## Model

The v1 model is `AstronomyAwareBaseline`, a compact PyTorch CNN with four heads:

- center heatmap logits for source localization;
- per-pixel class logits for STAR/GALAXY classification;
- non-negative ugriz flux maps;
- shape proxy maps for size and ellipticity-style quantities.

The design keeps the catalog contract stable while allowing later replacement
with UNet, ConvNeXt-small, or query-based set prediction.

## Training Objective

The headline method combines five terms:

- point-supervised center BCE on Gaussian heatmaps;
- class cross entropy only at labeled source points;
- inverse-variance weighted photometry loss where catalog flux labels are valid;
- adjacent-band consistency on log flux maps;
- PSF-convolved reconstruction loss against the observed image, masked to valid
  pixels.

The PSF reconstruction term is the main method contribution for v1. It connects
weak point supervision to the imaging physics: predicted flux maps should explain
the observed multiband image after convolution with the local PSF.

## Decoding

Inference decodes source candidates with sigmoid heatmap thresholding and local
max NMS. Each retained peak is converted to a `PredictionRecord` with pixel
coordinates, approximate sky coordinates using the configured pixel scale, class
argmax, confidence, flux proxies, and shape proxies.

## Native Frame Policy

Headline experiments use native SDSS corrected-frame pixel grids. The local data
README states that this avoids adding interpolation choices, correlated noise,
and PSF changes from WCS reprojection. Reprojected RGB/WCS products may be used
for visualization or secondary analysis, not for headline model training unless
reported as a separate ablation.
