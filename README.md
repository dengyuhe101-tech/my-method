# ORSDet

ORSDet is the cleaned reproducibility code for the SKAO SDC1 oriented radio
source detector. The public entry points are intentionally small:

```bash
python train.py
python test.py
```

The ORSDet implementation is under `orsdet/`. The root `src/` directory keeps
the CIANNA C/CUDA backend that ORSDet builds against. Historical experiment
folders and release artifacts are not included in this repository layout.

Large files are external by design except for the released best checkpoint:

- official SKAO SDC1 560 MHz FITS/catalog files
- generated `fwd_res`, prediction CSVs, catalogs, plots, and score outputs

## Layout

```text
ORSDet/
  train.py                  # training entry point
  test.py                   # inference/evaluation entry point
  requirements.txt
  weights/net0_s2700.dat    # released checkpoint used by default test.py
  external_data/README.md
  orsdet/                   # ORSDet data, target, detector, flux-head, eval code
  src/                      # CIANNA C/CUDA backend source
```

The included checkpoint is:

```text
weights/net0_s2700.dat
sha256 = ffbdb7988ebf46bf2c6a8b495f28e73483dfeb75dd690965753bdb556749fcbb
epoch = 2700
```

## Data

Place the official SDC1 files in:

```text
external_data/560Mhz-1kh/
  sdc1_560MHz_1000h.fits
  PrimaryBeam_560MHz.fits
  TrainingSet_560MHz.txt
  True_560MHz.txt
```

or set:

```bash
export SDC1_RAW_DATA_DIR=/path/to/560Mhz-1kh
```

## Install

Use a Python 3.10 environment with a working CUDA toolchain:

```bash
python -m pip install -r requirements.txt
cd src
USE_CUDA=ON python -m pip install -e .
```

## Test The Released Checkpoint

After installing dependencies and preparing the SDC1 raw data, run:

```bash
python test.py --gpu 0
```

`test.py` uses `weights/net0_s2700.dat` by default and writes outputs to:

```text
outputs/test_eval/
```

To test another checkpoint:

```bash
python test.py --gpu 0 --checkpoint /path/to/net0_s2700.dat --epoch 2700
```

To score an existing catalog without GPU inference:

```bash
python test.py --catalog /path/to/catalog_sdc1_2700.txt
```

## Train

```bash
python train.py --gpu 0 --epochs 2700 --save-every 100
```

The default output directory is `outputs/train/`. The final packaged checkpoint
is written under `outputs/train/net_save/`.

Generated outputs are written under `outputs/` and are ignored by Git.
