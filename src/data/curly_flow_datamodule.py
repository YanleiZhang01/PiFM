from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.datasets import make_circles
from torch.utils.data import DataLoader, Dataset


def _asymmetric_circle(
    n: int,
    noise: float,
    bias: str,
    factor: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    X, _ = make_circles(n_samples=n * 4, noise=noise, factor=factor)
    w = np.ones(len(X))
    if bias == "right":
        w += X[:, 0]
    elif bias == "left":
        w -= X[:, 0]
    elif bias == "top":
        w += X[:, 1]
    elif bias == "bottom":
        w -= X[:, 1]
    w = np.clip(w, 0, None)
    idx = rng.choice(len(X), size=n, p=w / w.sum(), replace=True)
    return torch.tensor(X[idx], dtype=torch.float32)


def psi_rotation(x: torch.Tensor) -> torch.Tensor:
    """Rotation reference velocity: [π·y, -π·x]."""
    return torch.stack([math.pi * x[..., 1], -math.pi * x[..., 0]], dim=-1)


def psi_expansion(x: torch.Tensor) -> torch.Tensor:
    """Expansion reference velocity: identity map."""
    return x.clone()


class _TrainDataset(Dataset):
    def __init__(self, x0: torch.Tensor, x_axis1: torch.Tensor, x_axis2: torch.Tensor):
        n = x0.shape[0]
        self.x0 = x0
        self.x_axis1 = x_axis1
        self.x_axis2 = x_axis2
        self.axis1_start = torch.zeros(n)
        self.axis2_start = torch.zeros(n)
        self.axis1_delta = torch.ones(n)
        self.axis2_delta = torch.ones(n)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "x0": self.x0[idx],
            "x_axis1": self.x_axis1[idx],
            "x_axis2": self.x_axis2[idx],
            "axis1_start": self.axis1_start[idx],
            "axis2_start": self.axis2_start[idx],
            "axis1_delta": self.axis1_delta[idx],
            "axis2_delta": self.axis2_delta[idx],
        }


class _HoldoutDataset(Dataset):
    def __init__(self, x_anchor: torch.Tensor, x_real: torch.Tensor):
        n = x_anchor.shape[0]
        self.x_anchor = x_anchor
        self.x_real = x_real
        self.axis1_start = torch.zeros(n)
        self.axis2_start = torch.zeros(n)
        self.axis1_end = torch.ones(n)
        self.axis2_end = torch.ones(n)
        self.axis1_steps = torch.ones(n)
        self.axis2_steps = torch.ones(n)

    def __len__(self) -> int:
        return self.x_anchor.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "x_anchor": self.x_anchor[idx],
            "x_real": self.x_real[idx],
            "axis1_start": self.axis1_start[idx],
            "axis2_start": self.axis2_start[idx],
            "axis1_end": self.axis1_end[idx],
            "axis2_end": self.axis2_end[idx],
            "axis1_steps": self.axis1_steps[idx],
            "axis2_steps": self.axis2_steps[idx],
        }


class CurlyFlowDataModule(LightningDataModule):
    """Synthetic 2-D asymmetric-circle datamodule for the curly-flow schematic.

    Axis 1 (rotation):  x0 → -x0,   reference velocity = [π·y, -π·x]
    Axis 2 (expansion): x0 → 3·x0,  reference velocity = x
    Ideal product target: -3·x0
    """

    def __init__(
        self,
        dim: int = 2,
        n_train: int = 5000,
        n_val: int = 500,
        n_test: int = 2000,
        n_ref: int = 10000,
        noise: float = 0.1,
        bias: str = "right",
        factor: float = 0.9,
        batch_size: int = 512,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dim = dim
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.reference_latent: Optional[np.ndarray] = None
        self.reference_velocity_sources: Optional[dict] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return
        hp = self.hparams

        def _gen(n: int, offset: int) -> torch.Tensor:
            rng = np.random.default_rng(hp.seed + offset)
            return _asymmetric_circle(n, hp.noise, hp.bias, hp.factor, rng)

        x0_train = _gen(hp.n_train, 0)
        x0_val = _gen(hp.n_val, 1)
        x0_test = _gen(hp.n_test, 2)
        x0_ref = _gen(hp.n_ref, 3)

        self.data_train = _TrainDataset(x0_train, -x0_train, x0_train * 3.0)
        self.data_val = _TrainDataset(x0_val, -x0_val, x0_val * 3.0)
        self.data_test = _HoldoutDataset(x0_test, -3.0 * x0_test)

        self.reference_latent = x0_ref.numpy()
        self.reference_velocity_sources = {
            "axis1": psi_rotation(x0_ref).numpy(),
            "axis2": psi_expansion(x0_ref).numpy(),
        }

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
