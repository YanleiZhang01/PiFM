import torch
from torch import nn


class ProductVectorField(nn.Module):
    """Two-condition product vector field: (xt, xs, t, s) -> (vt, vs)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        max_velocity_norm: float | None = None,
    ):
        super().__init__()
        self.max_velocity_norm = max_velocity_norm
        input_dim = 2 * dim + 2
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.head_t = nn.Linear(hidden_dim, dim)
        self.head_s = nn.Linear(hidden_dim, dim)
        for head in (self.head_t, self.head_s):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        xt: torch.Tensor,
        xs: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        h = self.backbone(torch.cat([xt, xs, t, s], dim=-1))
        return self._bound_velocity(self.head_t(h)), self._bound_velocity(self.head_s(h))

    def _bound_velocity(self, velocity: torch.Tensor) -> torch.Tensor:
        if self.max_velocity_norm is None:
            return velocity
        max_norm = float(self.max_velocity_norm)
        if max_norm <= 0.0:
            return velocity
        norm = torch.sqrt(velocity.square().sum(dim=-1, keepdim=True) + 1e-12)
        ratio = norm / max_norm
        scale = torch.tanh(ratio) / ratio.clamp_min(1e-12)
        return velocity * scale