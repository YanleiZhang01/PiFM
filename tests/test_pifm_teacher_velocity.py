import torch

from src.pifm.losses import _resolve_teacher_velocity


def test_resolve_teacher_velocity_prefers_reference_field():
    query = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    reference_x = query.clone()
    reference_v = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    direct_velocity = torch.ones_like(query)
    bridge_velocity = torch.zeros_like(query)
    axis_delta = torch.full((2, 1), 2.0)

    teacher = _resolve_teacher_velocity(
        bridge_velocity=bridge_velocity,
        direct_velocity=direct_velocity,
        reference_x=reference_x,
        reference_v=reference_v,
        query=query,
        axis_delta=axis_delta,
        ref_knn_k=1,
        ref_kernel="inverse_distance",
        ref_bandwidth=None,
    )

    assert torch.allclose(teacher, reference_v)


def test_resolve_teacher_velocity_uses_direct_velocity_without_delta_scaling():
    query = torch.zeros(2, 2)
    direct_velocity = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    bridge_velocity = torch.zeros_like(query)
    axis_delta = torch.full((2, 1), 2.0)

    teacher = _resolve_teacher_velocity(
        bridge_velocity=bridge_velocity,
        direct_velocity=direct_velocity,
        reference_x=None,
        reference_v=None,
        query=query,
        axis_delta=axis_delta,
        ref_knn_k=1,
        ref_kernel="inverse_distance",
        ref_bandwidth=None,
    )

    assert torch.allclose(teacher, direct_velocity)


def test_resolve_teacher_velocity_scales_bridge_derivative_by_axis_delta():
    query = torch.zeros(2, 2)
    bridge_velocity = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    axis_delta = torch.full((2, 1), 2.0)

    teacher = _resolve_teacher_velocity(
        bridge_velocity=bridge_velocity,
        direct_velocity=None,
        reference_x=None,
        reference_v=None,
        query=query,
        axis_delta=axis_delta,
        ref_knn_k=1,
        ref_kernel="inverse_distance",
        ref_bandwidth=None,
    )

    assert torch.allclose(teacher, bridge_velocity / axis_delta)
