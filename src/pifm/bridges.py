from typing import Optional, Tuple

import torch
from torch import nn


class BridgeMLP(nn.Module):
    """Network that bends linear interpolation between paired samples."""

    def __init__(self, dim: int, hidden_dim: int = 256, depth: int = 3):
        super().__init__()
        input_dim = 2 * dim + 1
        layers = [nn.Linear(input_dim, hidden_dim), nn.SELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SELU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x0, x1, t], dim=-1))


def get_xt(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    bridge_model: nn.Module,
    sigma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bridge mean, noisy sample, and noise for time t."""
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    mu_t = (1 - t) * x0 + t * x1 + t * (1 - t) * bridge_model(x0, x1, t)
    eps = torch.randn_like(x0)
    x_t = mu_t + torch.sqrt(t * (1 - t)).clamp_min(0.0) * sigma * eps
    return mu_t, x_t, eps


def get_xt_xt_dot(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    bridge_model: nn.Module,
    sigma: float = 0.0,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return interpolated sample and d(mu_t)/dt computed with autograd."""
    with torch.enable_grad():
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.detach().requires_grad_(True)
        mu_t, xt, eps = get_xt(t, x0, x1, bridge_model, sigma=sigma)
        mu_t_dot = torch.cat(
            [
                torch.autograd.grad(
                    mu_t[..., i].sum(),
                    t,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                for i in range(mu_t.shape[-1])
            ],
            dim=-1,
        )
    return xt, mu_t_dot, eps


def bridge_matching_loss(
    bridge_model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    target_velocity: Optional[torch.Tensor] = None,
    reference_x: Optional[torch.Tensor] = None,
    reference_v: Optional[torch.Tensor] = None,
    reference_knn_k: int = 16,
    reference_kernel: str = "inverse_distance",
    reference_bandwidth: Optional[float] = None,
    axis_delta: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
    sigma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train bridge velocity toward either supplied velocity or straight displacement."""
    if t is None:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
    xt, mu_t_dot, _ = get_xt_xt_dot(t, x0, x1, bridge_model, sigma=sigma)
    if axis_delta is not None:
        if axis_delta.dim() == 1:
            axis_delta = axis_delta.unsqueeze(-1)
        axis_delta = axis_delta.clamp_min(1e-6)
        mu_t_dot = mu_t_dot / axis_delta
    if reference_x is not None and reference_v is not None:
        target_velocity = local_velocity_lookup(
            xt.detach(),
            reference_x,
            reference_v,
            k=reference_knn_k,
            kernel=reference_kernel,
            bandwidth=reference_bandwidth,
        )
    if target_velocity is None:
        target_velocity = x1 - x0
        if axis_delta is not None:
            target_velocity = target_velocity / axis_delta
    loss = (mu_t_dot - target_velocity.detach()).pow(2).mean()
    return loss, xt, mu_t_dot


def local_velocity_lookup(
    x_query: torch.Tensor,
    reference_x: torch.Tensor,
    reference_v: torch.Tensor,
    k: int = 16,
    kernel: str = "inverse_distance",
    bandwidth: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate local drift by kNN averaging in latent space."""
    if reference_x.numel() == 0 or reference_v.numel() == 0:
        raise ValueError("reference_x and reference_v must be non-empty for local velocity lookup.")
    k = max(1, min(int(k), reference_x.shape[0]))
    distances = torch.cdist(x_query, reference_x)
    nn_dist, nn_idx = torch.topk(distances, k=k, dim=1, largest=False)
    if kernel == "inverse_distance":
        weights = 1.0 / nn_dist.clamp_min(1e-6)
    elif kernel == "gaussian":
        if bandwidth is None or float(bandwidth) <= 0.0:
            h = nn_dist[:, -1:].clamp_min(eps)
        else:
            h = torch.full_like(nn_dist[:, -1:], float(bandwidth)).clamp_min(eps)
        weights = torch.exp(-(nn_dist.square()) / (2.0 * h.square()))
    else:
        raise ValueError(f"Unsupported local velocity kernel: {kernel}")
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    neighbors = reference_v[nn_idx]
    return (neighbors * weights.unsqueeze(-1)).sum(dim=1)