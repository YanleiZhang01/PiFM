import torch

from src.data.circle_trio_datamodule import CircleTrioDataModule


def test_setup_populates_splits():
    dm = CircleTrioDataModule(n_train=100, n_val=20, n_test=30, batch_size=16)
    dm.setup()

    assert len(dm.data_train) == 100
    assert len(dm.data_val) == 20
    assert len(dm.data_test) == 30
    assert dm.dim == 2


def test_no_reference_fields():
    dm = CircleTrioDataModule(n_train=50, n_val=10, n_test=10, batch_size=8)
    dm.setup()

    assert not hasattr(dm, "reference_latent") or dm.reference_latent is None
    assert not hasattr(dm, "reference_velocity_sources") or dm.reference_velocity_sources is None


def test_train_batch_keys_and_shapes():
    dm = CircleTrioDataModule(n_train=100, n_val=20, n_test=30, batch_size=16)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    required = {"x0", "x_axis1", "x_axis2", "axis1_start", "axis2_start", "axis1_delta", "axis2_delta"}
    assert required.issubset(batch.keys())
    assert batch["x0"].shape == (16, 2)
    assert batch["x_axis1"].shape == (16, 2)
    assert batch["x_axis2"].shape == (16, 2)
    assert batch["axis1_delta"].shape == (16,)


def test_train_batch_deltas_are_unit():
    dm = CircleTrioDataModule(n_train=100, n_val=20, n_test=30, batch_size=32)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    assert torch.all(batch["axis1_delta"] == 1.0)
    assert torch.all(batch["axis2_delta"] == 1.0)
    assert torch.all(batch["axis1_start"] == 0.0)
    assert torch.all(batch["axis2_start"] == 0.0)


def test_test_batch_keys_and_shapes():
    dm = CircleTrioDataModule(n_train=50, n_val=10, n_test=30, batch_size=16)
    dm.setup()

    batch = next(iter(dm.test_dataloader()))
    required = {"x_anchor", "x_real", "axis1_start", "axis2_start", "axis1_end", "axis2_end", "axis1_steps", "axis2_steps"}
    assert required.issubset(batch.keys())
    assert batch["x_anchor"].shape == (16, 2)
    assert batch["x_real"].shape == (16, 2)


def test_test_batch_axis_extents():
    dm = CircleTrioDataModule(n_train=50, n_val=10, n_test=30, batch_size=16, axis2_end=0.5)
    dm.setup()

    batch = next(iter(dm.test_dataloader()))
    assert torch.all(batch["axis1_end"] == 1.0)
    assert torch.allclose(batch["axis2_end"], torch.full((16,), 0.5))
    assert torch.all(batch["axis1_steps"] == 1.0)
    assert torch.allclose(batch["axis2_steps"], torch.full((16,), 0.5))


def test_setup_is_idempotent():
    dm = CircleTrioDataModule(n_train=50, n_val=10, n_test=10, batch_size=8)
    dm.setup()
    ref_id = id(dm.data_train)
    dm.setup()
    assert id(dm.data_train) == ref_id


def test_different_seeds_produce_different_data():
    dm1 = CircleTrioDataModule(n_train=100, n_val=20, n_test=20, batch_size=32, seed=0)
    dm2 = CircleTrioDataModule(n_train=100, n_val=20, n_test=20, batch_size=32, seed=99)
    dm1.setup()
    dm2.setup()
    assert not torch.allclose(dm1.data_train.x0, dm2.data_train.x0)


def test_source_near_origin():
    dm = CircleTrioDataModule(n_train=200, n_val=20, n_test=20, batch_size=32)
    dm.setup()
    x0 = dm.data_train.x0
    # xa is disk of radius 1 centred at (0, 0)
    assert (x0[:, 0].abs() < 1.5).all()
    assert (x0[:, 1].abs() < 1.5).all()


def test_axis1_target_near_xb():
    dm = CircleTrioDataModule(n_train=200, n_val=20, n_test=20, batch_size=32)
    dm.setup()
    x_axis1 = dm.data_train.x_axis1
    # xb is disk of radius 0.5 centred at (5, 0)
    assert ((x_axis1[:, 0] - 5.0).abs() < 1.0).all()
    assert (x_axis1[:, 1].abs() < 1.0).all()


def test_test_real_near_xd():
    dm = CircleTrioDataModule(n_train=50, n_val=10, n_test=200, batch_size=200)
    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    x_real = batch["x_real"]
    # xd is disk of radius 0.5 centred at (5, 2.5)
    assert ((x_real[:, 0] - 5.0).abs() < 1.0).all()
    assert ((x_real[:, 1] - 2.5).abs() < 1.0).all()
