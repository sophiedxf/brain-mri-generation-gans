# 2D Brain MRI Generation with GANs

This project trains **DCGAN** and **WGAN-GP** models to generate **2D brain MRI slices** from the **BraTS 2023** dataset.

It includes:
- preprocessing from 3D NIfTI volumes to packed 2D slices
- patient-level train/val/test splitting
- DCGAN and WGAN-GP training
- FID/KID evaluation
- image generation from trained checkpoints

Supported image sizes:
- `64 x 64`
- `128 x 128`
- `256 x 256`

Supported MRI modalities:
- `t1n`
- `t1c`
- `t2w`
- `t2f`

<p align="center">
  <img src="grid_G_ema_64_1600px.png" width="600" alt="Example generated MRI slices">
</p>

## Overview

The typical workflow is:

1. Put BraTS data under `data/raw/`
2. Run preprocessing to create packed slices and metadata
3. Train `DCGAN` or `WGAN-GP`
4. Evaluate checkpoints with `FID` and `KID`
5. Generate sample images from a trained model

## Setup

Recommended environment:

```bat
conda create -n dcgan_wgan python=3.10.11 -y
conda activate dcgan_wgan
pip install -r requirements_all.txt
```

The scripts use CUDA automatically if available. CPU also works, but training and evaluation will be much slower.

## Repository Layout

```text
data/
  raw/                     BraTS data goes here
  preprocessed_slices_64/  Created by preprocessing

preprocess/
  preprocess.py

train_dcgan/
  train_dcgan.py
  dataset.py
  models_dcgan.py

train_wgangp/
  train_wgangp.py
  dataset.py
  models_dcgan.py
  models_wgangp.py

evaluate/
  eval_fid.py

generate/
  generate.py

runs/
  checkpoints, samples, plots, generated outputs
```

## Data

This repository does **not** include BraTS data.

Place the extracted BraTS folders under:

```text
data/raw/
```

The preprocessing script searches recursively. Expected BraTS modality suffixes are:
- `-t1n.nii.gz`
- `-t1c.nii.gz`
- `-t2w.nii.gz`
- `-t2f.nii.gz`

## Preprocessing

Preprocessing converts BraTS volumes into:
- a packed slice array: `*_packed.npy`
- a metadata file: `*_packed_metadata.npz`

The metadata stores the patient ID for each slice, which allows **patient-level splitting** during training and evaluation.

Example:

```bat
python preprocess\preprocess.py ^
  --raw_dir data\raw ^
  --out_dir data\preprocessed_slices_64 ^
  --modality t2f ^
  --target_size 64 ^
  --min_foreground 500 ^
  --target_total_slices 10000 ^
  --selection topk_foreground ^
  --seed 42 ^
  --save_png_samples
```

Good starting defaults:
- `--target_size 64`
- `--modality t2f`
- `--min_foreground 500`
- `--target_total_slices 10000`

## Train DCGAN

Example:

```bat
python train_dcgan\train_dcgan.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\dcgan_64 ^
  --image_size 64 ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --batch_size 128 ^
  --epochs 100 ^
  --lrG 4e-4 ^
  --lrD 2e-4 ^
  --beta1 0.5 ^
  --beta2 0.999 ^
  --use_amp ^
  --ema ^
  --ema_beta 0.999 ^
  --save_progress_every 1
```

Notes:
- `test_ratio` is the remainder: `1 - train_ratio - val_ratio`
- DCGAN saves progression frames during training
- `generator_progression.gif` is assembled after training finishes

## Train WGAN-GP

Example:

```bat
python train_wgangp\train_wgangp.py ^
  --data_dir data\preprocessed_slices_64 ^
  --out_dir runs\wgangp_64 ^
  --image_size 64 ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --batch_size 64 ^
  --epochs 100 ^
  --lr 1e-4 ^
  --beta1 0.0 ^
  --beta2 0.9 ^
  --n_critic 5 ^
  --lambda_gp 10 ^
  --gp_every 2 ^
  --ema ^
  --ema_beta 0.999
```

## Evaluate

`evaluate/eval_fid.py` computes **FID** and **KID** using TorchMetrics Inception-v3 features.

Use the **same** `seed`, `train_ratio`, and `val_ratio` as training so the held-out test set matches.

Example:

```bat
python evaluate\eval_fid.py ^
  --data_dir data\preprocessed_slices_64 ^
  --ckpt runs\dcgan_64\checkpoint_latest.pt ^
  --seed 42 ^
  --train_ratio 0.8 ^
  --val_ratio 0.1 ^
  --num_real 2000 ^
  --num_fake 2000 ^
  --batch_size 32 ^
  --use_ema ^
  --kid_subset_size 1000
```

## Generate Images

Example:

```bat
python generate\generate.py ^
  --ckpt runs\wgangp_64\checkpoint_latest.pt ^
  --out_dir runs\generated\wgangp_64 ^
  --num 64 ^
  --batch_size 64 ^
  --save_grid ^
  --grid_nrow 8 ^
  --grid_px 1600 ^
  --use_ema ^
  --tag wgangp64
```

## Patient-Level Split

The project uses **patient-level** rather than slice-level splitting.

That means all slices from one patient go entirely into:
- train
- val
- or test

This avoids leakage between splits and makes evaluation more realistic.

## Tips

- Start with `64 x 64` before trying larger sizes
- Use `t2f` first if you want a simple baseline
- Do not judge GAN quality from loss curves alone; always inspect sample images
- Early DCGAN samples often look noisy even when training is behaving normally

## Outputs

Typical outputs in `runs/...` include:
- checkpoints
- loss curves
- sample grids
- DCGAN progression frames
- `generator_progression.gif`

## License

This repository includes a `LICENSE` file. BraTS data is not distributed with this project and must be obtained separately under its own terms.
