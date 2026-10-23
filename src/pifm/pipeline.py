from dataclasses import dataclass

import torch
from torch import nn

from src.pifm.bridges import bridge_matching_loss
from src.pifm.losses import lie_bracket_loss, product_boundary_loss


@dataclass
class PiFMLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def compute_pifm_loss(
    product_model: nn.Module,
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
    lambda_lb: float = 0.1,
    use_lie_bracket: bool = True,
) -> PiFMLossOutput:
    """Compute product boundary loss with optional Lie bracket regularization."""
    match_loss, info = product_boundary_loss(
        product_model,
        bridge_axis1,
        bridge_axis2,
        x0,
        x_axis1,
        x_axis2,
        axis1_start=axis1_start,
        axis2_start=axis2_start,
        v_axis1=v_axis1,
        v_axis2=v_axis2,
        ref_x_axis1=ref_x_axis1,
        ref_v_axis1=ref_v_axis1,
        ref_x_axis2=ref_x_axis2,
        ref_v_axis2=ref_v_axis2,
        ref_knn_k=ref_knn_k,
        ref_kernel=ref_kernel,
        ref_bandwidth=ref_bandwidth,
        axis1_delta=axis1_delta,
        axis2_delta=axis2_delta,
        sigma=sigma,
    )
    use_lb = bool(use_lie_bracket) and float(lambda_lb) != 0.0
    if use_lb:
        t = info["t"]
        s = info["s"]
        lb_loss = lie_bracket_loss(product_model, info["x_t"], info["x_s"], t, s)
    else:
        lb_loss = torch.zeros((), device=x0.device, dtype=x0.dtype)
    loss = match_loss + float(lambda_lb) * lb_loss if use_lb else match_loss
    return PiFMLossOutput(
        loss=loss,
        metrics={
            "match_loss": match_loss.detach(),
            "lie_bracket_loss": lb_loss.detach(),
            "boundary_t_loss": info["loss_t"],
            "boundary_s_loss": info["loss_s"],
        },
    )
def compute_bridge_pretrain_loss(
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
    axis1_delta: torch.Tensor | None = None,
    axis2_delta: torch.Tensor | None = None,
    sigma: float = 0.0,
) -> PiFMLossOutput:
    """Pretrain both bridges toward straight paired displacements."""
    loss_axis1, _, _ = bridge_matching_loss(
        bridge_axis1,
        x0,
        x_axis1,
        target_velocity=v_axis1,
        reference_x=ref_x_axis1,
        reference_v=ref_v_axis1,
        reference_knn_k=ref_knn_k,
        axis_delta=axis1_delta,
        sigma=sigma,
    )
    loss_axis2, _, _ = bridge_matching_loss(
        bridge_axis2,
        x0,
        x_axis2,
        target_velocity=v_axis2,
        reference_x=ref_x_axis2,
        reference_v=ref_v_axis2,
        reference_knn_k=ref_knn_k,
        axis_delta=axis2_delta,
        sigma=sigma,
    )
    loss = loss_axis1 + loss_axis2
    return PiFMLossOutput(
        loss=loss,
        metrics={
            "bridge_axis1_loss": loss_axis1.detach(),
            "bridge_axis2_loss": loss_axis2.detach(),
        },
    )