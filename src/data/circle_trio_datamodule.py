from __future__ import annotations

from typing import Optional

import numpy as np
import ot
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def _sample_disk(
    n: int,
    radius: float,
    cx: float,
    cy: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    angles = 2 * np.pi * rng.random(n)
    r = radius * np.sqrt(rng.random(n))
    x = r * np.cos(angles) + cx
    y = r * np.sin(angles) + cy
    return torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)


def _ot_align(
    source: torch.Tensor,
    target: torch.Tensor,
    reg: float,
) -> torch.Tensor:
    """OT-align target to source using Sinkhorn (reg>0) or exact EMD (reg=0)."""
    n_s, n_t = len(source), len(target)
    a = np.ones(n_s) / n_s
    b = np.ones(n_t) / n_t
    M = torch.cdist(source, target).pow(2).detach().cpu().numpy()
    if reg == 0.0:
        pi = ot.emd(a, b, M)
    else:
        pi = ot.sinkhorn(a, b, M, reg=reg)
    return target[np.argmax(pi, axis=1)]


class _TrainDataset(Dataset):
    def __init__(
        self,
        x0: torch.Tensor,
        x_axis1: torch.Tensor,
        x_axis2: torch.Tensor,
    ):
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
    def __init__(
        self,
        x_anchor: torch.Tensor,
        x_real: torch.Tensor,
        axis2_end: float,
    ):
        n = x_anchor.shape[0]
        self.x_anchor = x_anchor
        self.x_real = x_real
        self.axis1_start = torch.zeros(n)
        self.axis2_start = torch.zeros(n)
        self.axis1_end = torch.ones(n)
        self.axis2_end = torch.full((n,), float(axis2_end))
        self.axis1_steps = torch.ones(n)
        self.axis2_steps = torch.full((n,), float(axis2_end))

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


class CircleTrioDataModule(LightningDataModule):
    """Three-circle 2-D datamodule for the product flow matching batch/seed study.

    Axis 1 (horizontal): xa → xb  (OT-aligned)
    Axis 2 (vertical):   xa → xc  (OT-aligned)
    Test target:         xd at (axis1_end=1, axis2_end=0.5) corner of the product grid
    """

    def __init__(
        self,
        dim: int = 2,
        n_train: int = 3000,
        n_val: int = 500,
        n_test: int = 2000,
        xa_radius: float = 1.0,
        xa_cx: float = 0.0,
        xa_cy: float = 0.0,
        xb_radius: float = 0.5,
        xb_cx: float = 5.0,
        xb_cy: float = 0.0,
        xc_radius: float = 1.0,
        xc_cx: float = 0.0,
        xc_cy: float = 5.0,
        xd_radius: float = 0.5,
        xd_cx: float = 5.0,
        xd_cy: float = 2.5,
        axis2_end: float = 0.5,
        ot_reg: float = 0.05,
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

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return
        hp = self.hparams

        def _xa(n: int, offset: int) -> torch.Tensor:
            return _sample_disk(n, hp.xa_radius, hp.xa_cx, hp.xa_cy,
                                np.random.default_rng(hp.seed + offset))

        def _xb(n: int, offset: int) -> torch.Tensor:
            return _sample_disk(n, hp.xb_radius, hp.xb_cx, hp.xb_cy,
                                np.random.default_rng(hp.seed + offset))

        def _xc(n: int, offset: int) -> torch.Tensor:
            return _sample_disk(n, hp.xc_radius, hp.xc_cx, hp.xc_cy,
                                np.random.default_rng(hp.seed + offset))

        def _xd(n: int, offset: int) -> torch.Tensor:
            return _sample_disk(n, hp.xd_radius, hp.xd_cx, hp.xd_cy,
                                np.random.default_rng(hp.seed + offset))

        def _build_train(xa: torch.Tensor, xb: torch.Tensor, xc: torch.Tensor) -> _TrainDataset:
            xb_al = _ot_align(xa, xb, hp.ot_reg)
            xc_al = _ot_align(xa, xc, hp.ot_reg)
            return _TrainDataset(xa, xb_al, xc_al)

        self.data_train = _build_train(_xa(hp.n_train, 0), _xb(hp.n_train, 1), _xc(hp.n_train, 2))
        self.data_val = _build_train(_xa(hp.n_val, 3), _xb(hp.n_val, 4), _xc(hp.n_val, 5))
        self.data_test = _HoldoutDataset(_xa(hp.n_test, 6), _xd(hp.n_test, 7), hp.axis2_end)

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
