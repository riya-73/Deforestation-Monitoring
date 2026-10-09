# Forest Watch: Sentinel-2 forest segmentation starter

A runnable scaffold for steps 4–6 of a forest monitoring portfolio project: aligned raster patching, spatially grouped train/validation/test split, U-Net training with BCE + Dice, and test metrics/overlay.

## Important input limitation

The provided `stacked_8ch.npy` is channels-first with shape `(8, 2216, 2199)`, but NumPy arrays do not retain CRS, affine transform, band names, or dates. The preparation pipeline deliberately requires georeferenced GeoTIFFs for image and label and refuses to guess this metadata. Recover source metadata first, then export/rebuild a GeoTIFF stack on a known grid. Do not invent an affine transform. Configure band order and label definition in your own metadata and README.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For CUDA, install the PyTorch build matching your machine from https://pytorch.org/get-started/locally/ before `pip install -e .`.

## Inputs

- `data/raw/imagery/image_stack.tif`: multi-band image, band-first, reflectance scaled to 0–1, with CRS/transform.
- `data/raw/labels/forest_mask.tif`: one-band categorical raster aligned to the same grid, values 1=forest, 0=non-forest, 255=unknown/nodata.

See `configs/rondonia.yaml` for project settings. Use nearest-neighbor to align categorical labels. Never convert unknown/nodata into non-forest.

## Prepare patches and geographic split

```bash
python -m forest_watch.prepare \
  --image data/raw/imagery/image_stack.tif \
  --label data/raw/labels/forest_mask.tif \
  --out data/processed/patches --patch-size 256 --seed 17
```

Patches are grouped into contiguous geographic blocks (4×4 patch neighborhoods) and whole blocks are assigned to train/validation/test, reducing spatial leakage. Inspect `splits.json` and verify no geographic block appears in multiple splits. Small datasets may not have enough blocks for robust geographic separation; collect a larger AOI rather than falling back to random patch splits.

## Train U-Net

```bash
python -m forest_watch.train \
  --patch-dir data/processed/patches --in-channels 8 \
  --epochs 30 --batch-size 8 --lr 0.0003 \
  --out outputs/best_model.pt
```

The encoder is ImageNet-pretrained ResNet-34. Binary logits are optimized with equal-weight BCE and soft Dice loss. Geometric flips and 90-degree rotations are applied to training patches only. Ensure all bands are reflectance-like in [0,1]; update normalization consistently if your source differs. For pretrained encoders, consider experimenting with a 3-band pretrained encoder plus extra spectral input channels, but evaluate on held-out geographic blocks.

## Evaluate

```bash
python -m forest_watch.evaluate \
  --patch-dir data/processed/patches \
  --checkpoint outputs/best_model.pt --threshold 0.5 \
  --out outputs/evaluation
```

Writes test-set precision, recall, F1, IoU and confusion counts to `metrics.json`, plus a sample input/label/prediction image to `test_overlay.png`. Tune the probability threshold on validation blocks only; report final scores once on the untouched test blocks. The first-three-channel preview is not true RGB unless its band order is known.

## Repository hygiene

Raw imagery and generated patches are excluded from Git by `.gitignore`. Add small result figures and documented methods if publishing the project. Record dataset citations, dates, band order, reflectance scaling, CRS, spatial split strategy, model settings, and known limitations.
