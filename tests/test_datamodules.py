from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.axis_pifm_datamodule import AxisPiFMDataModule
from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_axis_pifm_datamodule_builds_axis_batches(tmp_path):
    ad = pytest.importorskip("anndata")

    axis1_values = ["0", "1", "2"]
    axis2_values = ["A", "B", "C"]
    rows = []
    obs_axis1 = []
    obs_axis2 = []
    for i, axis1 in enumerate(axis1_values):
        for j, axis2 in enumerate(axis2_values):
            for k in range(4):
                rows.append([i, j, k / 10.0])
                obs_axis1.append(axis1)
                obs_axis2.append(axis2)

    adata = ad.AnnData(np.asarray(rows, dtype=np.float32))
    adata.obs["axis1"] = obs_axis1
    adata.obs["axis2"] = obs_axis2
    path = tmp_path / "axis_pfm.h5ad"
    adata.write_h5ad(path)

    dm = AxisPiFMDataModule(
        data_path=str(path),
        axis1_key="axis1",
        axis2_key="axis2",
        axis1_values=axis1_values,
        axis2_values=axis2_values,
        anchor={"axis1": "0", "axis2": "A"},
        holdout_points=[{"axis1": "2", "axis2": "C"}],
        axis1_method="curly_fm",
        axis2_method="conditional_fm",
        latent_dim=3,
        n_samples=2,
        batch_size=2,
    )
    dm.setup()

    train_batch = next(iter(dm.train_dataloader()))
    assert set(train_batch) == {"x0", "x_axis1", "x_axis2", "v_axis1", "axis1_delta", "axis2_delta"}
    assert train_batch["x0"].shape == (2, 3)
    assert train_batch["v_axis1"].shape == (2, 3)
    assert train_batch["axis1_delta"].shape == (2,)
    assert train_batch["axis2_delta"].shape == (2,)

    test_batch = next(iter(dm.test_dataloader()))
    assert set(test_batch) == {"x_anchor", "x_real", "axis1_steps", "axis2_steps"}
    assert test_batch["x_anchor"].shape == (2, 3)
    assert torch.all(test_batch["axis1_steps"] == 2)
    assert torch.all(test_batch["axis2_steps"] == 2)


def test_axis_pifm_datamodule_auto_uses_raw_when_dim_is_small(tmp_path):
    ad = pytest.importorskip("anndata")

    axis1_values = ["0", "1"]
    axis2_values = ["A", "B"]
    rows = []
    obs_axis1 = []
    obs_axis2 = []
    for i, axis1 in enumerate(axis1_values):
        for j, axis2 in enumerate(axis2_values):
            for k in range(3):
                rows.append([i, j, k / 10.0])
                obs_axis1.append(axis1)
                obs_axis2.append(axis2)

    adata = ad.AnnData(np.asarray(rows, dtype=np.float32))
    adata.obs["axis1"] = obs_axis1
    adata.obs["axis2"] = obs_axis2
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    path = tmp_path / "small_raw_with_pca.h5ad"
    adata.write_h5ad(path)

    dm = AxisPiFMDataModule(
        data_path=str(path),
        axis1_key="axis1",
        axis2_key="axis2",
        axis1_values=axis1_values,
        axis2_values=axis2_values,
        obsm_key="auto",
        latent_dim=50,
        raw_dim_threshold=50,
        n_samples=2,
        batch_size=2,
    )
    dm.setup()

    train_batch = next(iter(dm.train_dataloader()))
    assert dm.dim == 3
    assert train_batch["x0"].shape[1] == 3
