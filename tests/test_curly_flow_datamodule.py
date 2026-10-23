import math

import numpy as np
import torch

from src.data.curly_flow_datamodule import CurlyFlowDataModule, psi_expansion, psi_rotation


def test_psi_rotation_orthogonal_to_position():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    v = psi_rotation(x)
    dot = (v * x).sum(dim=-1)
    assert torch.allclose(dot, torch.zeros(3), atol=1e-5)


def test_psi_rotation_magnitude():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = psi_rotation(x)
    expected_norm = math.pi * torch.ones(2)
    assert torch.allclose(v.norm(dim=-1), expected_norm, atol=1e-5)


def test_psi_expansion_is_identity():
    x = torch.randn(10, 2)
    assert torch.allclose(psi_expansion(x), x)


def test_setup_populates_splits():
    dm = CurlyFlowDataModule(n_train=100, n_val=20, n_test=30, n_ref=200, batch_size=16)
    dm.setup()

    assert len(dm.data_train) == 100
    assert len(dm.data_val) == 20
    assert len(dm.data_test) == 30
    assert dm.dim == 2


def test_reference_fields_shape_and_values():
    dm = CurlyFlowDataModule(n_train=50, n_val=10, n_test=10, n_ref=100, batch_size=8)
    dm.setup()

    assert dm.reference_latent.shape == (100, 2)
    assert dm.reference_velocity_sources["axis1"].shape == (100, 2)
    assert dm.reference_velocity_sources["axis2"].shape == (100, 2)

    x_ref = torch.tensor(dm.reference_latent)
    assert torch.allclose(torch.tensor(dm.reference_velocity_sources["axis1"]), psi_rotation(x_ref), atol=1e-5)
    assert torch.allclose(torch.tensor(dm.reference_velocity_sources["axis2"]), psi_expansion(x_ref), atol=1e-5)


def test_train_batch_keys_and_shapes():
    dm = CurlyFlowDataModule(n_train=100, n_val=20, n_test=30, n_ref=50, batch_size=16)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    required = {"x0", "x_axis1", "x_axis2", "axis1_start", "axis2_start", "axis1_delta", "axis2_delta"}
    assert required.issubset(batch.keys())
    assert batch["x0"].shape == (16, 2)
    assert batch["x_axis1"].shape == (16, 2)
    assert batch["x_axis2"].shape == (16, 2)
    assert batch["axis1_delta"].shape == (16,)


def test_train_batch_axis_targets():
    dm = CurlyFlowDataModule(n_train=100, n_val=20, n_test=30, n_ref=50, batch_size=32)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    x0 = batch["x0"]
    assert torch.allclose(batch["x_axis1"], -x0, atol=1e-5)
    assert torch.allclose(batch["x_axis2"], x0 * 3.0, atol=1e-5)
    assert torch.all(batch["axis1_delta"] == 1.0)
    assert torch.all(batch["axis2_delta"] == 1.0)


def test_test_batch_keys_and_ideal_target():
    dm = CurlyFlowDataModule(n_train=100, n_val=20, n_test=30, n_ref=50, batch_size=16)
    dm.setup()

    batch = next(iter(dm.test_dataloader()))
    required = {"x_anchor", "x_real", "axis1_start", "axis2_start", "axis1_end", "axis2_end", "axis1_steps", "axis2_steps"}
    assert required.issubset(batch.keys())
    assert batch["x_anchor"].shape == (16, 2)
    assert torch.allclose(batch["x_real"], -3.0 * batch["x_anchor"], atol=1e-5)
    assert torch.all(batch["axis1_steps"] == 1.0)
    assert torch.all(batch["axis2_steps"] == 1.0)


def test_setup_is_idempotent():
    dm = CurlyFlowDataModule(n_train=50, n_val=10, n_test=10, n_ref=50, batch_size=8)
    dm.setup()
    ref_latent_id = id(dm.reference_latent)
    dm.setup()
    assert id(dm.reference_latent) == ref_latent_id


def test_different_seeds_produce_different_data():
    dm1 = CurlyFlowDataModule(n_train=100, n_val=20, n_test=20, n_ref=50, batch_size=32, seed=0)
    dm2 = CurlyFlowDataModule(n_train=100, n_val=20, n_test=20, n_ref=50, batch_size=32, seed=99)
    dm1.setup()
    dm2.setup()
    x1 = dm1.data_train.x0
    x2 = dm2.data_train.x0
    assert not torch.allclose(x1, x2)
