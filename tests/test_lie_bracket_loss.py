import torch
from torch import nn

from src.pifm.losses import lie_bracket_loss


class _TwoCond(nn.Module):
    """Minimal product field: (xt, xs, t, s) -> (vt, vs)."""

    def __init__(self, dim: int = 3, hidden: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim * 2 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_t = nn.Linear(hidden, dim)
        self.head_s = nn.Linear(hidden, dim)

    def forward(self, xt, xs, t, s):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        h = self.backbone(torch.cat([xt, xs, t, s], dim=-1))
        return self.head_t(h), self.head_s(h)


class _ConstantField(nn.Module):
    """Constant velocities: every curvature term vanishes by construction."""

    def __init__(self, dim: int = 3):
        super().__init__()
        self.a = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.randn(dim))

    def forward(self, xt, xs, t, s):
        n = xt.shape[0]
        # keep a graph path to every input while the derivatives stay zero
        zero = 0.0 * xt + 0.0 * xs + 0.0 * t + 0.0 * s
        return self.a.expand(n, -1) + zero, self.b.expand(n, -1) + zero


def _reference_residual(model, xt, xs, t, s):
    """Per-sample residual built from full jacobians, independent of the impl."""
    dim = xt.shape[-1]
    total = 0.0
    for k in range(xt.shape[0]):

        def single(a, b, c, d):
            return model(a.unsqueeze(0), b.unsqueeze(0), c.view(1, 1), d.view(1, 1))

        args = (xt[k], xs[k], t[k].reshape(()), s[k].reshape(()))
        jac_t = torch.autograd.functional.jacobian(
            lambda *a: single(*a)[0].squeeze(0), args
        )
        jac_s = torch.autograd.functional.jacobian(
            lambda *a: single(*a)[1].squeeze(0), args
        )
        vt, vs = (out.squeeze(0) for out in single(*args))
        # jac_* is ordered (d/d xt, d/d xs, d/d t, d/d s)
        residual = (
            jac_t[3].reshape(-1)
            - jac_s[2].reshape(-1)
            + jac_t[1] @ vs
            - jac_s[0] @ vt
        )
        total = total + residual.pow(2).sum()
    return total / (xt.shape[0] * dim)


def test_lie_bracket_loss_matches_jacobian_reference():
    torch.manual_seed(0)
    model = _TwoCond(dim=3)
    xt, xs = torch.randn(6, 3), torch.randn(6, 3)
    t, s = torch.rand(6), torch.rand(6)

    got = lie_bracket_loss(model, xt, xs, t, s)
    expected = _reference_residual(model, xt, xs, t, s)

    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-7)


def test_lie_bracket_loss_runs_under_no_grad_and_keeps_param_gradients():
    torch.manual_seed(0)
    model = _TwoCond(dim=3)
    xt, xs = torch.randn(6, 3), torch.randn(6, 3)
    t, s = torch.rand(6), torch.rand(6)

    # validation_step wraps the loss in no_grad, so it must re-enable grad itself
    with torch.no_grad():
        under_no_grad = lie_bracket_loss(model, xt, xs, t, s)
    assert torch.isfinite(under_no_grad)

    lie_bracket_loss(model, xt, xs, t, s).backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_lie_bracket_loss_accepts_1d_and_column_time_inputs():
    torch.manual_seed(0)
    model = _TwoCond(dim=3)
    xt, xs = torch.randn(6, 3), torch.randn(6, 3)
    t, s = torch.rand(6), torch.rand(6)

    flat = lie_bracket_loss(model, xt, xs, t, s)
    column = lie_bracket_loss(model, xt, xs, t.unsqueeze(-1), s.unsqueeze(-1))

    assert torch.allclose(flat, column)


def test_lie_bracket_loss_vanishes_for_constant_field():
    torch.manual_seed(0)
    model = _ConstantField(dim=3)
    xt, xs = torch.randn(6, 3), torch.randn(6, 3)
    t, s = torch.rand(6), torch.rand(6)

    assert lie_bracket_loss(model, xt, xs, t, s).item() == 0.0
