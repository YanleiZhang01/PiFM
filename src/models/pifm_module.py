from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from src.pifm.bridges import BridgeMLP, bridge_matching_loss
from src.pifm.pipeline import PiFMLossOutput, compute_bridge_pretrain_loss, compute_pifm_loss
from src.pifm.product import ProductVectorField
from src.pifm.sampling import integrate_axis1_axis2, integrate_axis2_axis1


class PiFMLitModule(LightningModule):
    """Lightning wrapper for product flow matching on paired tensors."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        bridge_depth: int = 3,
        product_depth: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        sigma: float = 0.01,
        axis1: dict[str, Any] | None = None,
        axis2: dict[str, Any] | None = None,
        lambda_lb: float = 0.1,
        use_lie_bracket: bool = True,
        bridge_warmup_steps: int = 1000,
        grad_clip_norm: float = 1.0,
        sample_steps: int = 100,
        bridge_teacher_knn_k: int = 16,
        max_velocity_norm: float | None = None,
        bridge_warmup_use_velocity_targets: bool = True,
        product_teacher_use_local_velocity: bool = False,
        product_teacher_knn_k: int = 100,
        product_teacher_kernel: str = "gaussian",
        product_teacher_bandwidth: float | None = None,
        optimizer: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.bridge_axis1 = BridgeMLP(dim=dim, hidden_dim=hidden_dim, depth=bridge_depth)
        self.bridge_axis2 = BridgeMLP(dim=dim, hidden_dim=hidden_dim, depth=bridge_depth)
        self.product_model = ProductVectorField(
            dim=dim,
            hidden_dim=hidden_dim,
            depth=product_depth,
            max_velocity_norm=max_velocity_norm,
        )
        self.reference_x_axis1: torch.Tensor | None = None
        self.reference_v_axis1: torch.Tensor | None = None
        self.reference_x_axis2: torch.Tensor | None = None
        self.reference_v_axis2: torch.Tensor | None = None
        self._warmup_finished_logged = False
        self._bridges_frozen = False

    def forward(
        self,
        xt: torch.Tensor,
        xs: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.product_model(xt, xs, t, s)

    def training_step(self, batch: Any, batch_idx: int):
        self._maybe_load_reference_fields()
        x0 = batch["x0"]
        x_axis1 = batch["x_axis1"]
        x_axis2 = batch["x_axis2"]
        axis1_start = batch.get("axis1_start")
        axis2_start = batch.get("axis2_start")
        v_axis1 = batch.get("v_axis1")
        v_axis2 = batch.get("v_axis2")
        axis1_delta = batch.get("axis1_delta")
        axis2_delta = batch.get("axis2_delta")
        if self._in_bridge_warmup():
            output = self._manual_bridge_warmup_step(
                x0=x0,
                x_axis1=x_axis1,
                x_axis2=x_axis2,
                v_axis1=v_axis1,
                v_axis2=v_axis2,
                axis1_delta=axis1_delta,
                axis2_delta=axis2_delta,
            )
            prefix = "train/bridge"
            self.log("train/phase", torch.zeros((), device=self.device), on_step=True, on_epoch=False)
        else:
            self._freeze_bridges()
            output = compute_pifm_loss(
                self.product_model,
                self.bridge_axis1,
                self.bridge_axis2,
                x0,
                x_axis1,
                x_axis2,
                axis1_start=axis1_start,
                axis2_start=axis2_start,
                v_axis1=v_axis1,
                v_axis2=v_axis2,
                ref_x_axis1=(
                    self.reference_x_axis1
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis1")
                    else None
                ),
                ref_v_axis1=(
                    self.reference_v_axis1
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis1")
                    else None
                ),
                ref_x_axis2=(
                    self.reference_x_axis2
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis2")
                    else None
                ),
                ref_v_axis2=(
                    self.reference_v_axis2
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis2")
                    else None
                ),
                ref_knn_k=self.hparams.product_teacher_knn_k,
                ref_kernel=self.hparams.product_teacher_kernel,
                ref_bandwidth=self.hparams.product_teacher_bandwidth,
                axis1_delta=axis1_delta,
                axis2_delta=axis2_delta,
                sigma=self.hparams.sigma,
                lambda_lb=self.hparams.lambda_lb,
                use_lie_bracket=self.hparams.use_lie_bracket,
            )
            self._manual_product_step(output.loss)
            prefix = "train/product"
            self.log("train/phase", torch.ones((), device=self.device), on_step=True, on_epoch=False)
        self._log_optimizer_lrs()
        train_loss = output.loss.detach()
        self.log("train/loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/loss", train_loss, on_step=False, on_epoch=True)
        for name, value in output.metrics.items():
            self.log(f"{prefix}/{name}", value, on_step=False, on_epoch=True)
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int):
        self._maybe_load_reference_fields()
        x0 = batch["x0"]
        x_axis1 = batch["x_axis1"]
        x_axis2 = batch["x_axis2"]
        axis1_start = batch.get("axis1_start")
        axis2_start = batch.get("axis2_start")
        v_axis1 = batch.get("v_axis1")
        v_axis2 = batch.get("v_axis2")
        axis1_delta = batch.get("axis1_delta")
        axis2_delta = batch.get("axis2_delta")
        if self._in_bridge_warmup():
            output = compute_bridge_pretrain_loss(
                self.bridge_axis1,
                self.bridge_axis2,
                x0,
                x_axis1,
                x_axis2,
                axis1_start=axis1_start,
                axis2_start=axis2_start,
                v_axis1=v_axis1 if self.hparams.bridge_warmup_use_velocity_targets else None,
                v_axis2=v_axis2 if self.hparams.bridge_warmup_use_velocity_targets else None,
                ref_x_axis1=(
                    self.reference_x_axis1
                    if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis1")
                    else None
                ),
                ref_v_axis1=(
                    self.reference_v_axis1
                    if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis1")
                    else None
                ),
                ref_x_axis2=(
                    self.reference_x_axis2
                    if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis2")
                    else None
                ),
                ref_v_axis2=(
                    self.reference_v_axis2
                    if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis2")
                    else None
                ),
                ref_knn_k=self.hparams.bridge_teacher_knn_k,
                axis1_delta=axis1_delta,
                axis2_delta=axis2_delta,
                sigma=self.hparams.sigma,
            )
            warmup_proxy = self._warmup_val_proxy(x0)
            self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss", warmup_proxy, on_step=False, on_epoch=True)
            for name, value in output.metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)
            self.log("val/warmup_proxy_loss", warmup_proxy, on_step=False, on_epoch=True)
        else:
            output = compute_pifm_loss(
                self.product_model,
                self.bridge_axis1,
                self.bridge_axis2,
                x0,
                x_axis1,
                x_axis2,
                axis1_start=axis1_start,
                axis2_start=axis2_start,
                v_axis1=v_axis1,
                v_axis2=v_axis2,
                ref_x_axis1=(
                    self.reference_x_axis1
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis1")
                    else None
                ),
                ref_v_axis1=(
                    self.reference_v_axis1
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis1")
                    else None
                ),
                ref_x_axis2=(
                    self.reference_x_axis2
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis2")
                    else None
                ),
                ref_v_axis2=(
                    self.reference_v_axis2
                    if self.hparams.product_teacher_use_local_velocity and self._use_local_teacher("axis2")
                    else None
                ),
                ref_knn_k=self.hparams.product_teacher_knn_k,
                ref_kernel=self.hparams.product_teacher_kernel,
                ref_bandwidth=self.hparams.product_teacher_bandwidth,
                axis1_delta=axis1_delta,
                axis2_delta=axis2_delta,
                sigma=self.hparams.sigma,
                lambda_lb=self.hparams.lambda_lb,
                use_lie_bracket=self.hparams.use_lie_bracket,
            )
            self.log("val/loss", output.loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_loss", output.loss, on_step=False, on_epoch=True)
            for name, value in output.metrics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True)
        return output.loss

    def test_step(self, batch: Any, batch_idx: int):
        x0 = batch["x_anchor"]
        target = batch["x_real"]
        axis1_start = batch.get("axis1_start")
        axis2_start = batch.get("axis2_start")
        axis1_end = batch.get("axis1_end")
        axis2_end = batch.get("axis2_end")
        axis1_steps = batch["axis1_steps"]
        axis2_steps = batch["axis2_steps"]
        traj_12 = integrate_axis1_axis2(
            self.product_model,
            x0,
            axis1_start=axis1_start,
            axis2_start=axis2_start,
            axis1_end=axis1_end,
            axis2_end=axis2_end,
            axis1_scale=axis1_steps,
            axis2_scale=axis2_steps,
            n_steps=self.hparams.sample_steps,
        )
        traj_21 = integrate_axis2_axis1(
            self.product_model,
            x0,
            axis1_start=axis1_start,
            axis2_start=axis2_start,
            axis1_end=axis1_end,
            axis2_end=axis2_end,
            axis1_scale=axis1_steps,
            axis2_scale=axis2_steps,
            n_steps=self.hparams.sample_steps,
        )
        comm_mse = (traj_12[-1] - traj_21[-1]).pow(2).mean()
        mse_12 = (traj_12[-1] - target).pow(2).mean()
        mse_21 = (traj_21[-1] - target).pow(2).mean()
        centroid_mse_12 = (traj_12[-1].mean(dim=0) - target.mean(dim=0)).pow(2).mean()
        centroid_mse_21 = (traj_21[-1].mean(dim=0) - target.mean(dim=0)).pow(2).mean()
        self.log("test/commutativity_mse", comm_mse, on_step=False, on_epoch=True)
        self.log("test/mse_axis1_axis2_to_real", mse_12, on_step=False, on_epoch=True)
        self.log("test/mse_axis2_axis1_to_real", mse_21, on_step=False, on_epoch=True)
        self.log("test/centroid_mse_axis1_axis2_to_real", centroid_mse_12, on_step=False, on_epoch=True)
        self.log("test/centroid_mse_axis2_axis1_to_real", centroid_mse_21, on_step=False, on_epoch=True)
        return {
            "commutativity_mse": comm_mse,
            "mse_axis1_axis2_to_real": mse_12,
            "mse_axis2_axis1_to_real": mse_21,
            "centroid_mse_axis1_axis2_to_real": centroid_mse_12,
            "centroid_mse_axis2_axis1_to_real": centroid_mse_21,
        }

    def configure_optimizers(self):
        bridge_axis1_lr, bridge_axis1_weight_decay = self._optimizer_hparams("bridge_axis1")
        bridge_axis2_lr, bridge_axis2_weight_decay = self._optimizer_hparams("bridge_axis2")
        product_lr, product_weight_decay = self._optimizer_hparams("product")
        bridge_axis1_optimizer = torch.optim.AdamW(
            self.bridge_axis1.parameters(),
            lr=bridge_axis1_lr,
            weight_decay=bridge_axis1_weight_decay,
        )
        bridge_axis2_optimizer = torch.optim.AdamW(
            self.bridge_axis2.parameters(),
            lr=bridge_axis2_lr,
            weight_decay=bridge_axis2_weight_decay,
        )
        product_optimizer = torch.optim.AdamW(
            self.product_model.parameters(),
            lr=product_lr,
            weight_decay=product_weight_decay,
        )
        return [bridge_axis1_optimizer, bridge_axis2_optimizer, product_optimizer]

    def on_train_epoch_end(self) -> None:
        self._maybe_log_phase_transition()
        keys = [
            "train/loss_epoch",
            "train/bridge/loss",
            "train/bridge/bridge_axis1_loss",
            "train/bridge/bridge_axis2_loss",
            "train/product/loss",
            "train/product/match_loss",
            "train/product/lie_bracket_loss",
            "train/product/boundary_t_loss",
            "train/product/boundary_s_loss",
        ]
        self._log_metric_snapshot("train_epoch_end", keys)

    def on_validation_epoch_end(self) -> None:
        keys = [
            "val/loss",
            "val_loss",
            "val/bridge_axis1_loss",
            "val/bridge_axis2_loss",
            "val/warmup_proxy_loss",
            "val/match_loss",
            "val/lie_bracket_loss",
            "val/boundary_t_loss",
            "val/boundary_s_loss",
        ]
        self._log_metric_snapshot("val_epoch_end", keys)

    def _use_local_teacher(self, axis_name: str) -> bool:
        axis_cfg = getattr(self.hparams, axis_name, None) or {}
        return axis_cfg.get("method") == "curly_fm"

    def _in_bridge_warmup(self) -> bool:
        return self._bridge_warmup_progress() < int(self.hparams.bridge_warmup_steps)

    def _warmup_val_proxy(self, reference: torch.Tensor) -> torch.Tensor:
        remaining = max(int(self.hparams.bridge_warmup_steps) - self._bridge_warmup_progress(), 0)
        return torch.full((), 1_000_000.0 + float(remaining), device=reference.device, dtype=reference.dtype)

    def _bridge_warmup_progress(self) -> int:
        return int(self.global_step // 2)

    def _manual_bridge_warmup_step(
        self,
        x0: torch.Tensor,
        x_axis1: torch.Tensor,
        x_axis2: torch.Tensor,
        v_axis1: torch.Tensor | None = None,
        v_axis2: torch.Tensor | None = None,
        axis1_delta: torch.Tensor | None = None,
        axis2_delta: torch.Tensor | None = None,
    ):
        opt_axis1, opt_axis2, _ = self.optimizers()
        opt_axis1.zero_grad(set_to_none=True)
        opt_axis2.zero_grad(set_to_none=True)

        loss_axis1, _, _ = bridge_matching_loss(
            self.bridge_axis1,
            x0,
            x_axis1,
            target_velocity=v_axis1 if self.hparams.bridge_warmup_use_velocity_targets else None,
            reference_x=(
                self.reference_x_axis1
                if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis1")
                else None
            ),
            reference_v=(
                self.reference_v_axis1
                if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis1")
                else None
            ),
            reference_knn_k=self.hparams.bridge_teacher_knn_k,
            axis_delta=axis1_delta,
            sigma=self.hparams.sigma,
        )
        loss_axis2, _, _ = bridge_matching_loss(
            self.bridge_axis2,
            x0,
            x_axis2,
            target_velocity=v_axis2 if self.hparams.bridge_warmup_use_velocity_targets else None,
            reference_x=(
                self.reference_x_axis2
                if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis2")
                else None
            ),
            reference_v=(
                self.reference_v_axis2
                if self.hparams.bridge_warmup_use_velocity_targets and self._use_local_teacher("axis2")
                else None
            ),
            reference_knn_k=self.hparams.bridge_teacher_knn_k,
            axis_delta=axis2_delta,
            sigma=self.hparams.sigma,
        )

        self.manual_backward(loss_axis1)
        self._clip_optimizer_gradients(opt_axis1)
        opt_axis1.step()

        self.manual_backward(loss_axis2)
        self._clip_optimizer_gradients(opt_axis2)
        opt_axis2.step()

        loss = loss_axis1.detach() + loss_axis2.detach()
        return PiFMLossOutput(
            loss=loss,
            metrics={
                "bridge_axis1_loss": loss_axis1.detach(),
                "bridge_axis2_loss": loss_axis2.detach(),
            },
        )

    def _manual_product_step(self, loss: torch.Tensor) -> None:
        opt_axis1, opt_axis2, opt_product = self.optimizers()
        opt_axis1.zero_grad(set_to_none=True)
        opt_axis2.zero_grad(set_to_none=True)
        opt_product.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        self._clip_optimizer_gradients(opt_product)
        opt_product.step()

    def _clip_optimizer_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.hparams.grad_clip_norm or self.hparams.grad_clip_norm <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(self.hparams.grad_clip_norm),
            gradient_clip_algorithm="norm",
        )

    def _freeze_bridges(self) -> None:
        if self._bridges_frozen:
            return
        for bridge in (self.bridge_axis1, self.bridge_axis2):
            bridge.eval()
            for parameter in bridge.parameters():
                parameter.requires_grad_(False)
        self._bridges_frozen = True

    def _log_optimizer_lrs(self) -> None:
        names = ("bridge_axis1", "bridge_axis2", "product")
        for name, optimizer in zip(names, self.optimizers()):
            lr = optimizer.param_groups[0]["lr"]
            self.log(
                f"train/lr/{name}",
                torch.tensor(float(lr), device=self.device),
                on_step=True,
                on_epoch=False,
            )

    def _optimizer_hparams(self, name: str) -> tuple[float, float]:
        optimizer_cfg = getattr(self.hparams, "optimizer", None) or {}
        group_cfg = optimizer_cfg.get(name, {}) if hasattr(optimizer_cfg, "get") else {}
        lr = group_cfg.get("lr", self.hparams.lr) if hasattr(group_cfg, "get") else self.hparams.lr
        weight_decay = (
            group_cfg.get("weight_decay", self.hparams.weight_decay)
            if hasattr(group_cfg, "get")
            else self.hparams.weight_decay
        )
        if lr is None:
            lr = self.hparams.lr
        if weight_decay is None:
            weight_decay = self.hparams.weight_decay
        return float(lr), float(weight_decay)

    def _maybe_load_reference_fields(self) -> None:
        if self.reference_x_axis1 is not None:
            return
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        latent = getattr(datamodule, "reference_latent", None)
        velocities = getattr(datamodule, "reference_velocity_sources", None)
        if latent is None or velocities is None:
            return
        dtype = next(self.parameters()).dtype
        reference_x = torch.as_tensor(latent, device=self.device, dtype=dtype)
        self.reference_x_axis1 = reference_x
        self.reference_x_axis2 = reference_x
        axis1_v = velocities.get("axis1")
        axis2_v = velocities.get("axis2")
        self.reference_v_axis1 = (
            torch.as_tensor(axis1_v, device=self.device, dtype=dtype) if axis1_v is not None else None
        )
        self.reference_v_axis2 = (
            torch.as_tensor(axis2_v, device=self.device, dtype=dtype) if axis2_v is not None else None
        )

    def _maybe_log_phase_transition(self) -> None:
        if self._warmup_finished_logged:
            return
        if not self._in_bridge_warmup():
            rank_zero_info(
                f"Bridge warmup finished at global_step={self.global_step}, "
                f"bridge_warmup_progress={self._bridge_warmup_progress()}. "
                "Switching to product-field training."
            )
            self._warmup_finished_logged = True

    def _log_metric_snapshot(self, prefix: str, keys: list[str]) -> None:
        metrics = getattr(self.trainer, "callback_metrics", {})
        items: list[str] = []
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().cpu().item()
            elif hasattr(value, "item"):
                value = float(value.item())
            else:
                value = float(value)
            items.append(f"{key}={value:.6f}")
        if items:
            rank_zero_info(f"{prefix}: " + ", ".join(items))