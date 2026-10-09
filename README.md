# Deforestation Monitoring with Sentinel-2

A portfolio project scaffold for mapping forest cover from Sentinel-2 imagery with a U-Net segmentation model. The current implementation covers geospatial patch preparation, geographically grouped data splits, training, and held-out evaluation. It is designed to grow into a multi-date deforestation change-detection and interactive mapping pipeline.

## What the project does

Given a multi-band, georeferenced image stack and an aligned binary forest mask, the pipeline:

1. Checks that imagery and labels share the same CRS, affine transform, and dimensions.
2. Divides the rasters into 256 × 256-pixel patches and omits patches with too many unknown labels.
3. Assigns whole geographic blocks—not individual patches—to train, validation, and test sets to reduce spatial leakage.
4. Trains a binary U-Net with an ImageNet-pretrained ResNet-34 encoder, BCE + soft Dice loss, and geometric training augmentation.
5. Evaluates predictions on held-out test blocks and reports precision, recall, F1, and intersection-over-union (IoU).
6. Saves a sample visualization comparing input channels, ground truth, and predicted forest mask.

### Results produced by a run

After running the commands below, the main artifacts are:

- `data/processed/patches/splits.json` — geographic train/validation/test patch assignment.
- `outputs/best_model.pt` — model checkpoint with the best validation loss.
- `outputs/evaluation/metrics.json` — test-set precision, recall, F1, IoU, threshold, and confusion counts.
- `outputs/evaluation/test_overlay.png` — example input, ground-truth mask, and prediction.

**No model metrics or prediction screenshots are included yet.** Training and evaluation have not been run on a verified, georeferenced imagery/label pair, so there are no measured performance results to report. The output paths above describe what a successful run will create; they are not claims about achieved accuracy.

## Data status and limitations

The supplied `stacked_8ch.npy` is a channels-first `float32` array of shape `(8, 2216, 2199)`. A plain NumPy array does not store CRS, affine transform, band names, or acquisition dates. Its channels and map footprint must be verified from source metadata before it can be used in this geospatial pipeline. Do not invent georeferencing or assume the band/date order from array values alone.

The current scripts expect:

- `data/raw/imagery/image_stack.tif` — a multi-band GeoTIFF, band-first, with CRS and transform. Input values are expected to be reflectance in approximately `[0, 1]`.
- `data/raw/labels/forest_mask.tif` — a one-band raster on the exact same grid, with `1 = forest`, `0 = non-forest`, and `255 = unknown/nodata`.

Use nearest-neighbor resampling for categorical labels and preserve unknown pixels as nodata. Raw imagery and generated outputs are excluded from Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For CUDA training, install the PyTorch build matching your system from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) before installing the project dependencies.

## Prepare patches and spatial splits

```bash
python -m forest_watch.prepare \
  --image data/raw/imagery/image_stack.tif \
  --label data/raw/labels/forest_mask.tif \
  --out data/processed/patches \
  --patch-size 256 \
  --seed 17
```

Patches are grouped in contiguous 4 × 4 patch neighborhoods, and whole blocks are assigned to splits. This reduces leakage compared with random patch splits, although adjacent blocks can still be spatially correlated. Inspect the split manifest and use a sufficiently large study area for a meaningful held-out test.

## Train

```bash
python -m forest_watch.train \
  --patch-dir data/processed/patches \
  --in-channels 8 \
  --epochs 30 \
  --batch-size 8 \
  --lr 0.0003 \
  --out outputs/best_model.pt
```

The encoder is pretrained ResNet-34. The model returns one logit per pixel; equal-weight binary cross-entropy and soft Dice loss are combined for optimization. Horizontal/vertical flips and 90-degree rotations apply only to training patches. Confirm the number and order of input channels and their reflectance scaling before training.

## Evaluate

```bash
python -m forest_watch.evaluate \
  --patch-dir data/processed/patches \
  --checkpoint outputs/best_model.pt \
  --threshold 0.5 \
  --out outputs/evaluation
```

Choose the probability threshold using validation data, then evaluate once on the held-out test split. The overlay displays the first three input channels; it is a true-color image only if those channels are confirmed to be RGB in that order.

## Configuration and code

- `configs/rondonia.yaml` — example paths and hyperparameters.
- `configs/data_metadata_template.json` — metadata checklist for the supplied NumPy array.
- `src/forest_watch/prepare.py` — patch extraction and block-level split.
- `src/forest_watch/train.py` — dataset loader, augmentation, U-Net, loss, and training loop.
- `src/forest_watch/evaluate.py` — test metrics and prediction overlay.

## Planned extensions

The next portfolio milestones are cloud-masked Sentinel-2 composites for two dates, aligned forest labels, multi-date change detection with false-positive filtering, polygon/area summaries, and an interactive map. Those features are not yet implemented in this repository.

## Training curves

A successful training run also writes `outputs/training_history.csv` and `outputs/training_curves.png`. The figure plots train/validation BCE+Dice loss and validation IoU, F1, precision, and recall per epoch. Curves are generated from actual training/validation batches, not example or synthetic values. No curve image is committed until the project has run with real, aligned labels.

## Interactive multi-date change map

Once you have two **aligned, georeferenced binary forest masks** (or probability rasters in `[0,1]`) on the same grid, run:

```bash
python -m forest_watch.change_map \
  --date1 data/processed/forest_2022.tif \
  --date2 data/processed/forest_2024.tif \
  --date1-name 2022 --date2-name 2024 \
  --out outputs/interactive_change_map.html
```

The output is a standalone Folium HTML map with toggleable forest layers for both dates, a red newly-cleared overlay, and polygon popups/tooltips with area in hectares. It also writes a GeoJSON of filtered polygons. The script requires matching CRS/transform/shape and defines candidate clearing as forest at date 1 and non-forest at date 2; cloud masking and seasonal/date comparability must be addressed upstream. This map generator has not been run on the supplied array because the array has no georeferencing and there are no verified date masks. The synthetic-looking map preview is intentionally not included.

## Resume bullets (accurate before benchmark results)

- Designed a geospatial forest-segmentation workflow that validates raster alignment, tiles imagery into labeled patches, and uses geographic block splits to reduce spatial leakage in model evaluation.
- Implemented a multi-spectral U-Net with an ImageNet-pretrained ResNet-34 encoder, combined BCE–Dice objective, and training-only geometric augmentation for binary forest-cover mapping.
- Built reproducible model-evaluation and mapping outputs: epoch-level loss/IoU/F1 curves plus a Folium change-map workflow that polygonizes newly cleared areas and reports hectare estimates from georeferenced masks.

After the first successful run, replace generic wording with verified study area, dataset, test metrics, and detected area. Do not claim measured impact before those results exist.
