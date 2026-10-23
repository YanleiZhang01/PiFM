import torch
from torch import nn


@torch.no_grad()
def integrate_axis1_axis2(
    model: nn.Module,
    x0: torch.Tensor,
    axis1_start: torch.Tensor | float | None = None,
    axis2_start: torch.Tensor | float | None = None,
    axis1_end: torch.Tensor | float | None = None,
    axis2_end: torch.Tensor | float | None = None,
    axis1_scale: torch.Tensor | float = 1.0,
    axis2_scale: torch.Tensor | float = 1.0,
    n_steps: int = 100,
    steps_per_unit: float | None = None,
) -> list[torch.Tensor]:
    """Euler integration along axis1 first, then axis2."""
    model.eval()
    x = x0.clone()
    x_ref = x0.clone()
    traj = [x.clone()]
    axis1_scale = _as_column(axis1_scale, x)
    axis2_scale = _as_column(axis2_scale, x)
    axis1_start = _as_column(0.0 if axis1_start is None else axis1_start, x)
    axis2_start = _as_column(0.0 if axis2_start is None else axis2_start, x)
    axis1_end = _as_column(axis1_start + axis1_scale if axis1_end is None else axis1_end, x)
    axis2_end = _as_column(axis2_start + axis2_scale if axis2_end is None else axis2_end, x)
    n_steps_axis1 = _resolve_n_steps(axis1_scale, n_steps, steps_per_unit)
    n_steps_axis2 = _resolve_n_steps(axis2_scale, n_steps, steps_per_unit)
    dt = 1.0 / n_steps_axis1 if n_steps_axis1 > 0 else 0.0
    for i in range(n_steps_axis1):
        frac = i * dt
        t = axis1_start + frac * (axis1_end - axis1_start)
        s = axis2_start.clone()
        vt, _ = model(x, x_ref, t, s)
        x = x + vt * dt * axis1_scale
        traj.append(x.clone())
    x_axis1 = x.clone()
    dt = 1.0 / n_steps_axis2 if n_steps_axis2 > 0 else 0.0
    for i in range(n_steps_axis2):
        frac = i * dt
        t = axis1_end.clone()
        s = axis2_start + frac * (axis2_end - axis2_start)
        _, vs = model(x_axis1, x, t, s)
        x = x + vs * dt * axis2_scale
        traj.append(x.clone())
    return traj


@torch.no_grad()
def integrate_axis2_axis1(
    model: nn.Module,
    x0: torch.Tensor,
    axis1_start: torch.Tensor | float | None = None,
    axis2_start: torch.Tensor | float | None = None,
    axis1_end: torch.Tensor | float | None = None,
    axis2_end: torch.Tensor | float | None = None,
    axis1_scale: torch.Tensor | float = 1.0,
    axis2_scale: torch.Tensor | float = 1.0,
    n_steps: int = 100,
    steps_per_unit: float | None = None,
) -> list[torch.Tensor]:
    """Euler integration along axis2 first, then axis1."""
    model.eval()
    x = x0.clone()
    x_ref = x0.clone()
    traj = [x.clone()]
    axis1_scale = _as_column(axis1_scale, x)
    axis2_scale = _as_column(axis2_scale, x)
    axis1_start = _as_column(0.0 if axis1_start is None else axis1_start, x)
    axis2_start = _as_column(0.0 if axis2_start is None else axis2_start, x)
    axis1_end = _as_column(axis1_start + axis1_scale if axis1_end is None else axis1_end, x)
    axis2_end = _as_column(axis2_start + axis2_scale if axis2_end is None else axis2_end, x)
    n_steps_axis1 = _resolve_n_steps(axis1_scale, n_steps, steps_per_unit)
    n_steps_axis2 = _resolve_n_steps(axis2_scale, n_steps, steps_per_unit)
    dt = 1.0 / n_steps_axis2 if n_steps_axis2 > 0 else 0.0
    for i in range(n_steps_axis2):
        frac = i * dt
        t = axis1_start.clone()
        s = axis2_start + frac * (axis2_end - axis2_start)
        _, vs = model(x_ref, x, t, s)
        x = x + vs * dt * axis2_scale
        traj.append(x.clone())
    x_axis2 = x.clone()
    dt = 1.0 / n_steps_axis1 if n_steps_axis1 > 0 else 0.0
    for i in range(n_steps_axis1):
        frac = i * dt
        t = axis1_start + frac * (axis1_end - axis1_start)
        s = axis2_end.clone()
        vt, _ = model(x, x_axis2, t, s)
        x = x + vt * dt * axis1_scale
        traj.append(x.clone())
    return traj


def _as_column(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.dim() == 1:
            value = value.unsqueeze(-1)
        return value
    return torch.full((reference.shape[0], 1), float(value), device=reference.device, dtype=reference.dtype)


def _resolve_n_steps(
    axis_scale: torch.Tensor,
    n_steps: int,
    steps_per_unit: float | None,
) -> int:
    if steps_per_unit is None:
        return int(n_steps)
    scale = float(axis_scale.detach().abs().max().cpu())
    if scale == 0.0:
        return 0
    return max(1, int(round(scale * float(steps_per_unit))))