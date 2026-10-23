"""Reusable product flow matching utilities."""

from src.pifm.bridges import BridgeMLP, get_xt, get_xt_xt_dot
from src.pifm.losses import lie_bracket_loss, product_boundary_loss
from src.pifm.product import ProductVectorField
from src.pifm.sampling import integrate_axis1_axis2, integrate_axis2_axis1

__all__ = [
    "BridgeMLP",
    "ProductVectorField",
    "get_xt",
    "get_xt_xt_dot",
    "integrate_axis1_axis2",
    "integrate_axis2_axis1",
    "lie_bracket_loss",
    "product_boundary_loss",
]
