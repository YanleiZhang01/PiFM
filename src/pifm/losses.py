import torch
from torch import nn

from src.pifm.bridges import get_xt_xt_dot, local_velocity_lookup


def lie_bracket_loss(
    model: nn.Module,
    xt: torch.Tensor,
    xs: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Compute the zero-curvature residual of the product field.

    Enforces path independence through the transport form of the condition,
    keeping the convective terms that come from the spatial dependence of each
    velocity on the other axis:

        ||partial_s vt - partial_t vs
          + (partial vt / partial xs) vs - (partial vs / partial xt) vt||^2
    """
    with torch.enable_grad():
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        xt_g = xt.detach().requires_grad_(True)
        xs_g = xs.detach().requires_grad_(True)
        t_g = t.detach().requires_grad_(True)
        s_g = s.detach().requires_grad_(True)
        vt, vs = model(xt_g, xs_g, t_g, s_g)

        dvt_ds = torch.cat(
            [
                torch.autograd.grad(
                    vt[:, i].sum(), s_g, create_graph=True, retain_graph=True
                )[0]
                for i in range(vt.shape[-1])
            ],
            dim=-1,
        )
        dvs_dt = torch.cat(
            [
                torch.autograd.grad(
                    vs[:, i].sum(), t_g, create_graph=True, retain_graph=True
                )[0]
                for i in range(vs.shape[-1])
            ],
            dim=-1,
        )

        # (partial vt / partial xs) vs, one output component at a time.
        conv_t = torch.stack(
            [
                (
                    torch.autograd.grad(
                        vt[:, i].sum(), xs_g, create_graph=True, retain_graph=True
                    )[0]
                    * vs
                ).sum(dim=-1)
                for i in range(vt.shape[-1])
            ],
            dim=-1,
        )
        # (partial vs / partial xt) vt, one output component at a time.
        conv_s = torch.stack(
            [
                (
                    torch.autograd.grad(
                        vs[:, i].sum(), xt_g, create_graph=True, retain_graph=True
                    )[0]
                    * vt
                ).sum(dim=-1)
                for i in range(vs.shape[-1])
            ],
            dim=-1,
        )

        residual = dvt_ds - dvs_dt + conv_t - conv_s
        return residual.pow(2).mean()


def product_boundary_loss(
    model: nn.Module,
    bridge_axis1: nn.Module,
    bridge_axis2: nn.Module,
    x0: torch.Tensor,
    x_axis1: torch.Tensor,
    x_axis2: torch.Tensor,
    axis1_start: torch.Tensor | None = None,
    axis2_start: torch.Tensor | None = None,
    v_axis1: torch.Tensor | None = None,
    v_axis2: torch.Tensor | None = None,
    ref_x_axis1: torch.Tensor | None = None,
    ref_v_axis1: torch.Tensor | None = None,
    ref_x_axis2: torch.Tensor | None = None,
    ref_v_axis2: torch.Tensor | None = None,
    ref_knn_k: int = 16,
    ref_kernel: str = "inverse_distance",
    ref_bandwidth: float | None = None,
    axis1_delta: torch.Tensor | None = None,
    axis2_delta: torch.Tensor | None = None,
    sigma: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Boundary supervision from one-axis bridge samples.

    The product field is trained like two one-axis flows: vt on intervals
    with s fixed, and vs on intervals with t fixed. The bridge supplies both
    the sampled point and the teacher velocity for each axis.
    """
    batch_size = x0.shape[0]
    tau_t = torch.rand(batch_size, device=x0.device, dtype=x0.dtype)
    tau_s = torch.rand(batch_size, device=x0.device, dtype=x0.dtype)
    axis1_start = torch.zeros(batch_size, device=x0.device, dtype=x0.dtype) if axis1_start is None else axis1_start.to(device=x0.device, dtype=x0.dtype)
    axis2_start = torch.zeros(batch_size, device=x0.device, dtype=x0.dtype) if axis2_start is None else axis2_start.to(device=x0.device, dtype=x0.dtype)
    axis1_delta = torch.ones(batch_size, device=x0.device, dtype=x0.dtype) if axis1_delta is None else axis1_delta.to(device=x0.device, dtype=x0.dtype)
    axis2_delta = torch.ones(batch_size, device=x0.device, dtype=x0.dtype) if axis2_delta is None else axis2_delta.to(device=x0.device, dtype=x0.dtype)

    axis1_delta_c = _as_column(axis1_delta, x0).clamp_min(1e-6)
    axis2_delta_c = _as_column(axis2_delta, x0).clamp_min(1e-6)
    tau_t_c = _as_column(tau_t, x0)
    tau_s_c = _as_column(tau_s, x0)

    x_t, teacher_t, _ = get_xt_xt_dot(
        tau_t_c,
        x0,
        x_axis1,
        bridge_axis1,
        sigma=sigma,
        create_graph=False,
    )
    x_s, teacher_s, _ = get_xt_xt_dot(
        tau_s_c,
        x0,
        x_axis2,
        bridge_axis2,
        sigma=sigma,
        create_graph=False,
    )
    teacher_t = _resolve_teacher_velocity(
        bridge_velocity=teacher_t,
        direct_velocity=v_axis1,
        reference_x=ref_x_axis1,
        reference_v=ref_v_axis1,
        query=x_t,
        axis_delta=axis1_delta_c,
        ref_knn_k=ref_knn_k,
        ref_kernel=ref_kernel,
        ref_bandwidth=ref_bandwidth,
    )
    teacher_s = _resolve_teacher_velocity(
        bridge_velocity=teacher_s,
        direct_velocity=v_axis2,
        reference_x=ref_x_axis2,
        reference_v=ref_v_axis2,
        query=x_s,
        axis_delta=axis2_delta_c,
        ref_knn_k=ref_knn_k,
        ref_kernel=ref_kernel,
        ref_bandwidth=ref_bandwidth,
    )

    t_global = _as_column(axis1_start, x0) + tau_t_c * axis1_delta_c
    s_global = _as_column(axis2_start, x0) + tau_s_c * axis2_delta_c
    s_fixed = _as_column(axis2_start, x0)
    t_fixed = _as_column(axis1_start, x0)

    vt_pred, _ = model(x_t, x0, t_global, s_fixed)
    _, vs_pred = model(x0, x_s, t_fixed, s_global)
    loss_t = (vt_pred - teacher_t.detach()).pow(2).mean()
    loss_s = (vs_pred - teacher_s.detach()).pow(2).mean()
    return loss_t + loss_s, {
        "loss_t": loss_t.detach(),
        "loss_s": loss_s.detach(),
        "t_local": tau_t.detach(),
        "s_local": tau_s.detach(),
        "t": t_global.detach().squeeze(-1),
        "s": s_global.detach().squeeze(-1),
        "x_t": x_t.detach(),
        "x_s": x_s.detach(),
    }


def _as_column(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.dim() == 1:
        value = value.unsqueeze(-1)
    return value


def _resolve_teacher_velocity(
    bridge_velocity: torch.Tensor,
    direct_velocity: torch.Tensor | None,
    reference_x: torch.Tensor | None,
    reference_v: torch.Tensor | None,
    query: torch.Tensor,
    axis_delta: torch.Tensor,
    ref_knn_k: int,
    ref_kernel: str,
    ref_bandwidth: float | None,
) -> torch.Tensor:
    """Choose the strongest available one-axis teacher for product supervision."""
    if reference_x is not None and reference_v is not None:
        return local_velocity_lookup(
            query.detach(),
            reference_x,
            reference_v,
            k=ref_knn_k,
            kernel=ref_kernel,
            bandwidth=ref_bandwidth,
        )
    if direct_velocity is not None:
        return direct_velocity.to(device=query.device, dtype=query.dtype)
    return bridge_velocity / axis_delta
