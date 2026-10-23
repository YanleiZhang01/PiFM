<div align="center">

# PiFM — Path-independent Flow Matching

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>

</div>

## Overview

**PiFM (Pi Flow Matching)** learns a joint vector field over a two-dimensional parameter grid `(t, s) ∈ [0,1]²`.
Given a source population `x₀` and two independent transport targets `x_axis1` (along `t`) and `x_axis2` (along `s`), PiFM trains two vector fields `(vt, vs) = f(x, t, s)` that simultaneously satisfies both marginals while enforcing approximate path-independence (commutativity) via an optional Lie-bracket regulariser.

At inference the model can integrate along either axis order — `axis1 → axis2` or `axis2 → axis1` — and evaluate at any corner `(t_end, s_end)` of the grid, enabling **out-of-distribution extrapolation** to unseen combinations of conditions.

### Training phases

1. **Bridge warmup** (`bridge_warmup_steps > 0`): two lightweight `BridgeMLP` networks are pre-trained on each axis independently before the model is touched.
2. **PiFM training**: the `VectorField` is trained with a boundary + interior matching loss. An optional kNN-smoothed reference velocity field (`curly_fm` bridges) can supervise interior states.

---

## Pipeline Architecture

```
configs/
  data/          ← LightningDataModule configs
  model/         ← PiFMLitModule config (pifm.yaml)
  experiment/    ← per-experiment overrides
  hparams_search/← Hydra multirun sweep configs

src/
  data/          ← datamodules (axis_pifm, curly_flow, circle_trio, …)
  models/        ← PiFMLitModule (training loop, test metrics)
  pifm/
    bridges.py   ← BridgeMLP, bridge_matching_loss, local_velocity_lookup
    product.py   ← ProductVectorField
    losses.py    ← lie_bracket_loss, product_boundary_loss
    pipeline.py  ← compute_pifm_loss, compute_bridge_pretrain_loss
    sampling.py  ← integrate_axis1_axis2, integrate_axis2_axis1

scripts/         ← one shell script per experiment
tests/           ← pytest unit tests for datamodules
```

Key design points:

- Every experiment is a Hydra **experiment config** (`configs/experiment/<name>.yaml`) that overrides the default `data`, `model`, and `trainer` configs.
- `PiFMLitModule` automatically picks up `reference_latent` / `reference_velocity_sources` from the datamodule when they are present, activating the kNN reference-velocity teacher without any code changes.
- The `axis.method` field in the model config controls the bridge type per axis: `conditional_fm` (standard OT-aligned flow) or `curly_fm` (kNN-supervised curly bridge).

---

## Installation

```bash
# create and activate a conda environment
conda create -n pifm python=3.10
conda activate pifm

# install PyTorch (adjust cuda version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# install remaining dependencies
pip install -r requirements.txt
```

---

## Experiments

### 1. Curly-FM comparison (synthetic schematic)

**What it tests:** A 2D asymmetric-circle dataset where the two transport axes follow analytic curly (non-straight) reference velocities — a rotation field and an expansion field. The model must learn to compose them into a joint product flow. The experiment can be run with or without the Lie-bracket regulariser.

**Data geometry:**
- Source `x₀`: biased samples from an asymmetric circle
- Axis 1 target `x₁ = −x₀` (half-rotation), reference velocity `ψ_rot(x) = [πy, −πx]`
- Axis 2 target `x₂ = 3x₀` (radial expansion), reference velocity `ψ_exp(x) = x`
- Product target (held out): `−3x₀`

**Train and evaluate:**

```bash
bash scripts/run_curly_flow_pifm.sh
```

The script trains to convergence (up to 200 epochs, early-stop patience 30), saves the best checkpoint, then runs `src/eval.py` on the test set and prints commutativity MSE and per-axis integration errors.

**Common overrides:**

```bash
# disable Lie-bracket regularisation
bash scripts/run_curly_flow_pifm.sh model.use_lie_bracket=false

# change training budget or batch size
bash scripts/run_curly_flow_pifm.sh trainer.max_epochs=100 data.batch_size=256

# train on GPU
bash scripts/run_curly_flow_pifm.sh trainer=gpu
```

**Logged metrics** (visible in CSV / W&B / TensorBoard):

| Metric | Description |
|--------|-------------|
| `val_loss` | Product matching loss (monitored for checkpointing) |
| `test/mse_axis1_axis2_to_real` | MSE of axis1→axis2 integration vs. ground truth |
| `test/mse_axis2_axis1_to_real` | MSE of axis2→axis1 integration vs. ground truth |
| `test/commutativity_mse` | Squared difference between the two integration orders |

---

### 2. Meta-FM comparison (circle trio)

**What it tests:** Three circles in 2D — a source circle `xa` at the origin, and two OT-aligned target circles `xb` (shifted right, smaller) and `xc` (shifted up). PiFM learns to extrapolate to the out-of-distribution product corner `(t=1, s=0.5)`, reaching a fourth circle `xd` at `(5, 2.5)` that is never seen during training. Two sweeps reproduce the batch-size sensitivity and seed-robustness studies.

**Data geometry:**
- `xa`: disk, radius 1, centre (0, 0) — source / anchor
- `xb`: disk, radius 0.5, centre (5, 0) — axis 1 target (OT-aligned to `xa`)
- `xc`: disk, radius 1, centre (0, 5) — axis 2 target (OT-aligned to `xa`)
- `xd`: disk, radius 0.5, centre (5, 2.5) — test target at `(t=1, s=0.5)`

This experiment uses `conditional_fm` bridges (no curly teacher, no bridge warmup).

#### Single run

```bash
python src/train.py experiment=circle_trio_pifm
```

#### Batch-size sensitivity sweep (8 batch sizes × 5 seeds = 40 runs)

```bash
bash scripts/run_circle_trio_batchsize.sh
```

Each run trains independently. Results land in `logs/circle_trio_pifm/sweeps/batchsize/<timestamp>/`.

```bash
# override trainer budget for a quick smoke test
bash scripts/run_circle_trio_batchsize.sh trainer.max_epochs=20
```

#### Seed robustness sweep (10 seeds, batch size 512)

```bash
bash scripts/run_circle_trio_seeds.sh
```

Results land in `logs/circle_trio_pifm/sweeps/seeds/<timestamp>/`.

**Batch sizes swept:** 64, 128, 192, 256, 320, 384, 448, 512  
**Seeds swept (batch-size study):** 0–4  
**Seeds swept (seed study):** 0–9

**Key test metric:** `test/mse_axis2_axis1_to_real` — integration along `s` first (0→0.5) then `t` (0→1) versus the held-out `xd` distribution, matching the inference order used in the original comparison study.

---

### 3. Single-cell (reprmgramming)

**What it tests:** Real scRNA-seq data from the pancreas dataset. Two biological axes (e.g. pseudo-time stages) are aligned with OT-based pseudo-velocities and a curly-FM bridge with kNN reference velocity supervision. The model must extrapolate to held-out combinations of axis1/axis2 values.

**Prerequisites:** Download the preprocessed `.h5ad` file and place it at the path specified by `data.data_path` in `configs/experiment/singlecell_pifm.yaml` (or override via `paths.data_dir`).

**Train and evaluate:**

```bash
bash scripts/run_singlecell_pifm.sh
```

The script trains to the best `val_loss`, saves the checkpoint, then evaluates with `src/eval.py`.

**Common overrides:**

```bash
# use a GPU
bash scripts/run_singlecell_pifm.sh trainer=gpu

# tune bridge warmup
bash scripts/run_singlecell_pifm.sh \
  model.bridge_warmup_steps=1000 \
```

**Architecture differences vs. synthetic experiments:**

| Setting | Curly-FM / Circle-trio | Single-cell |
|---------|------------------------|-------------|
| `axis.method` | `curly_fm` / `conditional_fm` | `conditional_fm` |
| `bridge_warmup_steps` | 500 / 0 | 500 |
| `use_lie_bracket` | `true` / `false` | `true` |
| `product_teacher_use_local_velocity` | `true` / `false` | `false` |

---

## Project Structure

```
configs/
├── data/
│   ├── curly_flow_pifm.yaml       # synthetic curly-flow data
│   ├── circle_trio_pifm.yaml      # three-circle data
│   └── singlecell_pifm.yaml       # single-cell data
├── experiment/
│   ├── curly_flow_pifm.yaml       # Curly-FM comparison experiment
│   ├── circle_trio_pifm.yaml      # Meta-FM comparison experiment
│   └── singlecell_pifm.yaml       # single-cell experiment
├── hparams_search/
│   ├── circle_trio_batchsize.yaml # 8 batch sizes × 5 seeds sweep
│   └── circle_trio_seeds.yaml     # 10-seed robustness sweep
└── model/
    └── pifm.yaml                  # shared PiFMLitModule defaults

src/
├── data/
│   ├── curly_flow_datamodule.py   # CurlyFlowDataModule + ψ_rot / ψ_exp
│   ├── circle_trio_datamodule.py  # CircleTrioDataModule + OT alignment
│   └── axis_pifm_datamodule.py    # AxisPiFMDataModule (AnnData / single-cell)
├── models/
│   └── pifm_module.py             # PiFMLitModule
└── pifm/
    ├── bridges.py                 # BridgeMLP, local_velocity_lookup
    ├── product.py                 # ProductVectorField
    ├── losses.py                  # lie_bracket_loss, product_boundary_loss
    ├── pipeline.py                # compute_pifm_loss
    └── sampling.py                # Euler integrators

scripts/
├── run_curly_flow_pifm.sh         # train + eval (Curly-FM experiment)
├── run_circle_trio_batchsize.sh   # Hydra multirun batch-size sweep
├── run_circle_trio_seeds.sh       # Hydra multirun seed sweep
└── run_singlecell_pifm.sh         # train + eval (single-cell)

tests/
├── test_curly_flow_datamodule.py  # unit tests for CurlyFlowDataModule
└── test_circle_trio_datamodule.py # unit tests for CircleTrioDataModule
```

---

## Common Options

**Hardware:**

```bash
# CPU (default)
python src/train.py experiment=curly_flow_pifm trainer=cpu

# Single GPU
python src/train.py experiment=curly_flow_pifm trainer=gpu
```

**Logging:**

```bash
# CSV logger (always on by default)
python src/train.py experiment=curly_flow_pifm logger=csv

# Weights & Biases
python src/train.py experiment=curly_flow_pifm logger=wandb

# TensorBoard
python src/train.py experiment=curly_flow_pifm logger=tensorboard
```

**Debugging:**

```bash
# one batch, fast check
python src/train.py experiment=curly_flow_pifm debug=fdr

# overfit a single batch to verify the loss decreases
python src/train.py experiment=curly_flow_pifm debug=overfit
```

**Resume from checkpoint:**

```bash
python src/train.py experiment=curly_flow_pifm ckpt_path=/path/to/best.ckpt
```

**Running tests:**

```bash
pytest tests/test_curly_flow_datamodule.py
pytest tests/test_circle_trio_datamodule.py
```

All logs, checkpoints, and Hydra configs for a run are saved under `logs/<task_name>/runs/<YYYY-MM-DD_HH-MM-SS>/`. Sweep results go to `logs/<task_name>/sweeps/<study>/<timestamp>/`.
