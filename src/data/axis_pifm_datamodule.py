from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, Dataset, random_split


Point = tuple[str, str]


@dataclass(frozen=True)
class AxisEdge:
    point: Point
    axis1_target: Point
    axis2_target: Point


class AxisProductTensorDataset(Dataset):
    """Axis-aware tensor dataset for product flow matching."""

    def __init__(
        self,
        x0: torch.Tensor,
        x_axis1: torch.Tensor,
        x_axis2: torch.Tensor,
        v_axis1: Optional[torch.Tensor] = None,
        v_axis2: Optional[torch.Tensor] = None,
        axis1_start: Optional[torch.Tensor] = None,
        axis2_start: Optional[torch.Tensor] = None,
        axis1_delta: Optional[torch.Tensor] = None,
        axis2_delta: Optional[torch.Tensor] = None,
    ):
        if not (x0.ndim == x_axis1.ndim == x_axis2.ndim == 2):
            raise ValueError("x0, x_axis1, and x_axis2 must be 2D tensors [n, dim].")
        tensors = [x0, x_axis1, x_axis2]
        if v_axis1 is not None:
            tensors.append(v_axis1)
        if v_axis2 is not None:
            tensors.append(v_axis2)
        if axis1_start is not None:
            tensors.append(axis1_start)
        if axis2_start is not None:
            tensors.append(axis2_start)
        if axis1_delta is not None:
            tensors.append(axis1_delta)
        if axis2_delta is not None:
            tensors.append(axis2_delta)
        n = min(t.shape[0] for t in tensors)
        if n == 0:
            raise ValueError("PiFM tensors must contain at least one sample.")
        self.x0 = x0[:n].float()
        self.x_axis1 = x_axis1[:n].float()
        self.x_axis2 = x_axis2[:n].float()
        self.v_axis1 = v_axis1[:n].float() if v_axis1 is not None else None
        self.v_axis2 = v_axis2[:n].float() if v_axis2 is not None else None
        self.axis1_start = axis1_start[:n].float() if axis1_start is not None else torch.zeros(n)
        self.axis2_start = axis2_start[:n].float() if axis2_start is not None else torch.zeros(n)
        self.axis1_delta = axis1_delta[:n].float() if axis1_delta is not None else torch.ones(n)
        self.axis2_delta = axis2_delta[:n].float() if axis2_delta is not None else torch.ones(n)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "x0": self.x0[index],
            "x_axis1": self.x_axis1[index],
            "x_axis2": self.x_axis2[index],
            "axis1_start": self.axis1_start[index],
            "axis2_start": self.axis2_start[index],
            "axis1_delta": self.axis1_delta[index],
            "axis2_delta": self.axis2_delta[index],
        }
        if self.v_axis1 is not None:
            item["v_axis1"] = self.v_axis1[index]
        if self.v_axis2 is not None:
            item["v_axis2"] = self.v_axis2[index]
        return item


class AxisHoldoutTensorDataset(Dataset):
    """Dataset for anchor-to-heldout prediction metrics."""

    def __init__(
        self,
        x_anchor: torch.Tensor,
        x_real: torch.Tensor,
        axis1_start: torch.Tensor,
        axis2_start: torch.Tensor,
        axis1_end: torch.Tensor,
        axis2_end: torch.Tensor,
        axis1_steps: torch.Tensor,
        axis2_steps: torch.Tensor,
    ):
        if not (x_anchor.ndim == x_real.ndim == 2):
            raise ValueError("x_anchor and x_real must be 2D tensors [n, dim].")
        n = min(
            x_anchor.shape[0],
            x_real.shape[0],
            axis1_start.shape[0],
            axis2_start.shape[0],
            axis1_end.shape[0],
            axis2_end.shape[0],
            axis1_steps.shape[0],
            axis2_steps.shape[0],
        )
        if n == 0:
            raise ValueError("Holdout tensors must contain at least one sample.")
        self.x_anchor = x_anchor[:n].float()
        self.x_real = x_real[:n].float()
        self.axis1_start = axis1_start[:n].float()
        self.axis2_start = axis2_start[:n].float()
        self.axis1_end = axis1_end[:n].float()
        self.axis2_end = axis2_end[:n].float()
        self.axis1_steps = axis1_steps[:n].float()
        self.axis2_steps = axis2_steps[:n].float()

    def __len__(self) -> int:
        return self.x_anchor.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x_anchor": self.x_anchor[index],
            "x_real": self.x_real[index],
            "axis1_start": self.axis1_start[index],
            "axis2_start": self.axis2_start[index],
            "axis1_end": self.axis1_end[index],
            "axis2_end": self.axis2_end[index],
            "axis1_steps": self.axis1_steps[index],
            "axis2_steps": self.axis2_steps[index],
        }


class AxisPiFMDataModule(LightningDataModule):
    """Build generic two-axis PiFM tensors from an AnnData object.

    The two axes are configured by obs columns. Holdout points are removed from
    training edges and kept only for prediction-vs-real test metrics.
    """

    def __init__(
        self,
        data_path: str,
        axis1_key: str,
        axis2_key: str,
        axis1_values: Optional[Sequence[Any]] = None,
        axis2_values: Optional[Sequence[Any]] = None,
        anchor: Optional[dict[str, Any]] = None,
        holdout_points: Optional[Sequence[dict[str, Any]]] = None,
        axis1_method: str = "conditional_fm",
        axis2_method: str = "conditional_fm",
        axis1_velocity_key: Optional[str] = None,
        axis2_velocity_key: Optional[str] = None,
        axis1_pseudo_velocity_method: str = "nearest_forward",
        axis2_pseudo_velocity_method: str = "nearest_forward",
        pseudo_velocity_knn_k: int = 8,
        pseudo_velocity_ot_reg: float = 0.05,
        pseudo_velocity_ot_max_iter: int = 200,
        pseudo_velocity_ot_tol: float = 1e-4,
        obsm_key: Optional[str] = "auto",
        layer: Optional[str] = None,
        latent_dim: int = 50,
        raw_dim_threshold: int = 50,
        n_samples: Optional[int] = None,
        max_cells_per_point: Optional[int] = None,
        batch_size: int = 256,
        train_val_test_split: Sequence[float] = (0.8, 0.1, 0.1),
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.dim: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is not None:
            return

        self._validate_methods()
        adata = self._read_adata()
        axis1_labels = self._obs_values(adata, self.hparams.axis1_key)
        axis2_labels = self._obs_values(adata, self.hparams.axis2_key)
        axis1_values = self._resolve_axis_values(axis1_labels, self.hparams.axis1_values)
        axis2_values = self._resolve_axis_values(axis2_labels, self.hparams.axis2_values)
        holdouts = self._parse_points(self.hparams.holdout_points or [])
        anchor = self._parse_point(self.hparams.anchor or {"axis1": axis1_values[0], "axis2": axis2_values[0]})

        rng = np.random.default_rng(self.hparams.seed)
        pools = self._collect_pools(axis1_labels, axis2_labels, axis1_values, axis2_values, rng)
        latent, velocity_sources = self._make_latent_and_velocities(adata)
        self.dim = int(latent.shape[1])
        axis1_order = {value: idx for idx, value in enumerate(axis1_values)}
        axis2_order = {value: idx for idx, value in enumerate(axis2_values)}
        axis1_coord = self._numeric_axis_coordinates(axis1_values)
        axis2_coord = {value: float(idx) for value, idx in axis2_order.items()}
        velocity_sources = self._resolve_velocity_sources(
            pools,
            latent,
            velocity_sources,
            axis1_values,
            axis2_values,
            axis1_order,
            axis2_order,
            axis1_coord,
            axis2_coord,
            holdouts,
        )
        self.reference_latent = latent
        self.reference_velocity_sources = velocity_sources

        edges = self._build_edges(axis1_values, axis2_values, holdouts, pools)
        if not edges:
            raise ValueError("No observed local axis1/axis2 training edges remain after holdout removal.")
        train_dataset = self._make_training_dataset(
            edges,
            pools,
            latent,
            velocity_sources,
            axis1_coord,
            axis2_coord,
            rng,
        )
        lengths = self._split_lengths(len(train_dataset), self.hparams.train_val_test_split)
        generator = torch.Generator().manual_seed(self.hparams.seed)
        self.data_train, self.data_val, _ = random_split(train_dataset, lengths, generator=generator)
        self.data_test = self._make_holdout_dataset(
            anchor,
            holdouts,
            pools,
            latent,
            axis1_values,
            axis2_values,
            axis1_coord,
            axis2_coord,
            rng,
        )

    def train_dataloader(self):
        return self._dataloader(self.data_train, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.data_val, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.data_test, shuffle=False)

    def _dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def _read_adata(self):
        try:
            import anndata as ad
        except ImportError as exc:
            raise ImportError("anndata is required for AxisPiFMDataModule.") from exc
        path = Path(self.hparams.data_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"AnnData file not found: {path}")
        return ad.read_h5ad(path)

    def _validate_methods(self) -> None:
        allowed = {"conditional_fm", "curly_fm"}
        pseudo_allowed = {"nearest_forward", "knn_forward", "ot_barycentric"}
        for name in ("axis1_method", "axis2_method"):
            method = getattr(self.hparams, name)
            if method not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got '{method}'.")
        for name in ("axis1_pseudo_velocity_method", "axis2_pseudo_velocity_method"):
            method = getattr(self.hparams, name)
            if method not in pseudo_allowed:
                raise ValueError(f"{name} must be one of {sorted(pseudo_allowed)}, got '{method}'.")
        if int(self.hparams.pseudo_velocity_knn_k) < 1:
            raise ValueError("pseudo_velocity_knn_k must be >= 1.")
        if float(self.hparams.pseudo_velocity_ot_reg) <= 0.0:
            raise ValueError("pseudo_velocity_ot_reg must be > 0.")
        if int(self.hparams.pseudo_velocity_ot_max_iter) < 1:
            raise ValueError("pseudo_velocity_ot_max_iter must be >= 1.")
        if float(self.hparams.pseudo_velocity_ot_tol) <= 0.0:
            raise ValueError("pseudo_velocity_ot_tol must be > 0.")

    @staticmethod
    def _obs_values(adata, key: str) -> np.ndarray:
        if key not in adata.obs:
            available = ", ".join(map(str, adata.obs.columns))
            raise KeyError(f"obs column '{key}' not found. Available columns: {available}")
        return adata.obs[key].astype(str).to_numpy()

    @staticmethod
    def _resolve_axis_values(labels: np.ndarray, configured: Optional[Sequence[Any]]) -> list[str]:
        if configured is not None:
            return [str(value) for value in configured]
        return sorted(set(labels.tolist()))

    @staticmethod
    def _parse_point(point: dict[str, Any]) -> Point:
        return str(point["axis1"]), str(point["axis2"])

    def _parse_points(self, points: Sequence[dict[str, Any]]) -> set[Point]:
        return {self._parse_point(point) for point in points}

    def _collect_pools(
        self,
        axis1_labels: np.ndarray,
        axis2_labels: np.ndarray,
        axis1_values: Sequence[str],
        axis2_values: Sequence[str],
        rng: np.random.Generator,
    ) -> dict[Point, np.ndarray]:
        pools: dict[Point, np.ndarray] = {}
        for axis1 in axis1_values:
            for axis2 in axis2_values:
                idx = np.flatnonzero((axis1_labels == axis1) & (axis2_labels == axis2))
                if idx.size == 0:
                    continue
                max_cells = self.hparams.max_cells_per_point
                if max_cells is not None and idx.size > max_cells:
                    idx = rng.choice(idx, size=max_cells, replace=False)
                pools[(axis1, axis2)] = idx
        return pools

    def _make_latent_and_velocities(self, adata) -> tuple[np.ndarray, dict[str, Optional[np.ndarray]]]:
        svd = None

        if self.hparams.obsm_key and self.hparams.obsm_key != "auto":
            latent = self._obsm_matrix(adata, self.hparams.obsm_key)
        else:
            raw = self._expression_matrix(adata, np.arange(adata.n_obs))
            raw_dim = int(raw.shape[1])
            if raw_dim <= self.hparams.raw_dim_threshold:
                latent = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
            elif "X_pca" in adata.obsm:
                latent = np.asarray(adata.obsm["X_pca"])
            else:
                svd = TruncatedSVD(n_components=self.hparams.latent_dim, random_state=self.hparams.seed)
                latent = svd.fit_transform(raw)

        latent_dim = min(int(self.hparams.latent_dim), int(latent.shape[1]))
        latent = np.asarray(latent[:, :latent_dim], dtype=np.float32)
        velocities = {
            "axis1": self._velocity_matrix(adata, self.hparams.axis1_velocity_key, latent.shape[1], svd),
            "axis2": self._velocity_matrix(adata, self.hparams.axis2_velocity_key, latent.shape[1], svd),
        }
        return latent, velocities

    def _expression_matrix(self, adata, indices: np.ndarray):
        if self.hparams.layer:
            if self.hparams.layer not in adata.layers:
                raise KeyError(f"layer '{self.hparams.layer}' not found. Available: {list(adata.layers.keys())}")
            return adata.layers[self.hparams.layer][indices]
        return adata.X[indices]

    @staticmethod
    def _obsm_matrix(adata, key: str) -> np.ndarray:
        if key not in adata.obsm:
            raise KeyError(f"obsm_key '{key}' not found. Available obsm keys: {list(adata.obsm.keys())}")
        return np.asarray(adata.obsm[key])

    def _velocity_matrix(
        self,
        adata,
        key: Optional[str],
        latent_dim: int,
        svd: Optional[TruncatedSVD],
    ) -> Optional[np.ndarray]:
        if not key:
            return None
        if key in adata.obsm:
            velocity = np.asarray(adata.obsm[key])
        elif key in adata.layers:
            velocity = adata.layers[key]
            velocity = velocity.toarray() if sparse.issparse(velocity) else np.asarray(velocity)
        else:
            raise KeyError(f"velocity_key '{key}' not found in adata.obsm or adata.layers.")

        if velocity.shape[1] == latent_dim:
            return velocity.astype(np.float32, copy=False)
        if svd is not None and velocity.shape[1] == svd.components_.shape[1]:
            return (velocity @ svd.components_.T).astype(np.float32, copy=False)
        if "PCs" in adata.varm and velocity.shape[1] == adata.varm["PCs"].shape[0]:
            pcs = np.asarray(adata.varm["PCs"])[:, :latent_dim]
            return (velocity @ pcs).astype(np.float32, copy=False)
        raise ValueError(
            f"velocity_key '{key}' has dim {velocity.shape[1]}, which cannot be aligned "
            f"to latent dim {latent_dim}. Provide velocity in latent space or fit SVD in this run."
        )

    def _build_edges(
        self,
        axis1_values: Sequence[str],
        axis2_values: Sequence[str],
        holdouts: set[Point],
        pools: dict[Point, np.ndarray],
    ) -> list[AxisEdge]:
        observed = set(pools) - holdouts
        edges: list[AxisEdge] = []
        axis1_order = {value: idx for idx, value in enumerate(axis1_values)}
        axis2_order = {value: idx for idx, value in enumerate(axis2_values)}
        for axis1 in axis1_values[:-1]:
            for axis2 in axis2_values[:-1]:
                point = (axis1, axis2)
                if point not in observed:
                    continue
                axis1_target = self._next_observed_point(point, 0, observed, axis1_values, axis2_values, axis1_order, axis2_order)
                axis2_target = self._next_observed_point(point, 1, observed, axis1_values, axis2_values, axis1_order, axis2_order)
                if axis1_target is not None and axis2_target is not None:
                    edges.append(AxisEdge(point, axis1_target, axis2_target))
        return edges

    def _make_training_dataset(
        self,
        edges: Sequence[AxisEdge],
        pools: dict[Point, np.ndarray],
        latent: np.ndarray,
        velocity_sources: dict[str, Optional[np.ndarray]],
        axis1_coord: dict[str, float],
        axis2_coord: dict[str, float],
        rng: np.random.Generator,
    ) -> AxisProductTensorDataset:
        n_per_edge = self._resolve_n_samples(edges, pools)
        x0_parts = []
        x1_parts = []
        x2_parts = []
        v1_parts = []
        v2_parts = []
        axis1_start_parts = []
        axis2_start_parts = []
        axis1_delta_parts = []
        axis2_delta_parts = []
        axis1_is_curly = self.hparams.axis1_method == "curly_fm"
        axis2_is_curly = self.hparams.axis2_method == "curly_fm"

        for edge in edges:
            source_pool = pools[edge.point]
            idx0 = self._sample_indices(source_pool, n_per_edge, rng)
            x0 = latent[idx0]
            axis1_delta = self._axis_delta(edge.point[0], edge.axis1_target[0], axis1_coord)
            axis2_delta = self._axis_delta(edge.point[1], edge.axis2_target[1], axis2_coord)
            idx1 = self._paired_target_indices(
                x0,
                source_pool,
                pools[edge.axis1_target],
                latent,
                n_per_edge,
                rng,
                self.hparams.axis1_method,
                velocity_sources["axis1"],
                idx0,
                axis1_delta,
            )
            idx2 = self._paired_target_indices(
                x0,
                source_pool,
                pools[edge.axis2_target],
                latent,
                n_per_edge,
                rng,
                self.hparams.axis2_method,
                velocity_sources["axis2"],
                idx0,
                axis2_delta,
            )
            x1 = latent[idx1]
            x2 = latent[idx2]
            x0_parts.append(x0)
            x1_parts.append(x1)
            x2_parts.append(x2)
            axis1_start_parts.append(np.full(n_per_edge, axis1_coord[edge.point[0]], dtype=np.float32))
            axis2_start_parts.append(np.full(n_per_edge, axis2_coord[edge.point[1]], dtype=np.float32))
            axis1_delta_parts.append(np.full(n_per_edge, axis1_delta, dtype=np.float32))
            axis2_delta_parts.append(np.full(n_per_edge, axis2_delta, dtype=np.float32))
            if axis1_is_curly:
                v1_parts.append(self._edge_velocity("axis1", idx0, x0, x1, velocity_sources, axis1_delta))
            if axis2_is_curly:
                v2_parts.append(self._edge_velocity("axis2", idx0, x0, x2, velocity_sources, axis2_delta))

        return AxisProductTensorDataset(
            torch.from_numpy(np.concatenate(x0_parts)),
            torch.from_numpy(np.concatenate(x1_parts)),
            torch.from_numpy(np.concatenate(x2_parts)),
            torch.from_numpy(np.concatenate(v1_parts)) if axis1_is_curly else None,
            torch.from_numpy(np.concatenate(v2_parts)) if axis2_is_curly else None,
            torch.from_numpy(np.concatenate(axis1_start_parts)),
            torch.from_numpy(np.concatenate(axis2_start_parts)),
            torch.from_numpy(np.concatenate(axis1_delta_parts)),
            torch.from_numpy(np.concatenate(axis2_delta_parts)),
        )

    def _make_holdout_dataset(
        self,
        anchor: Point,
        holdouts: set[Point],
        pools: dict[Point, np.ndarray],
        latent: np.ndarray,
        axis1_values: Sequence[str],
        axis2_values: Sequence[str],
        axis1_coord: dict[str, float],
        axis2_coord: dict[str, float],
        rng: np.random.Generator,
    ) -> AxisHoldoutTensorDataset:
        if anchor not in pools:
            raise ValueError(f"anchor point {anchor} is not present in the dataset.")
        if not holdouts:
            n = self.hparams.n_samples or len(pools[anchor])
            idx_anchor = self._sample_indices(pools[anchor], n, rng)
            idx_real = self._sample_indices(pools[anchor], n, rng)
            return AxisHoldoutTensorDataset(
                torch.from_numpy(latent[idx_anchor]),
                torch.from_numpy(latent[idx_real]),
                torch.zeros(n),
                torch.zeros(n),
                torch.zeros(n),
                torch.zeros(n),
                torch.zeros(n),
                torch.zeros(n),
            )

        n = self.hparams.n_samples or min(len(pools[anchor]), *(len(pools[p]) for p in holdouts if p in pools))
        x_anchor_parts = []
        x_real_parts = []
        axis1_start = []
        axis2_start = []
        axis1_end = []
        axis2_end = []
        axis1_steps = []
        axis2_steps = []
        for point in sorted(holdouts):
            if point not in pools:
                raise ValueError(f"holdout point {point} is not present in the dataset.")
            idx_anchor = self._sample_indices(pools[anchor], n, rng)
            idx_real = self._sample_indices(pools[point], n, rng)
            x_anchor_parts.append(latent[idx_anchor])
            x_real_parts.append(latent[idx_real])
            start1 = float(axis1_coord[anchor[0]])
            start2 = float(axis2_coord[anchor[1]])
            end1 = float(axis1_coord[point[0]])
            end2 = float(axis2_coord[point[1]])
            axis1_start.append(np.full(n, start1, dtype=np.float32))
            axis2_start.append(np.full(n, start2, dtype=np.float32))
            axis1_end.append(np.full(n, end1, dtype=np.float32))
            axis2_end.append(np.full(n, end2, dtype=np.float32))
            axis1_steps.append(np.full(n, end1 - start1, dtype=np.float32))
            axis2_steps.append(np.full(n, end2 - start2, dtype=np.float32))

        return AxisHoldoutTensorDataset(
            torch.from_numpy(np.concatenate(x_anchor_parts)),
            torch.from_numpy(np.concatenate(x_real_parts)),
            torch.from_numpy(np.concatenate(axis1_start)),
            torch.from_numpy(np.concatenate(axis2_start)),
            torch.from_numpy(np.concatenate(axis1_end)),
            torch.from_numpy(np.concatenate(axis2_end)),
            torch.from_numpy(np.concatenate(axis1_steps)),
            torch.from_numpy(np.concatenate(axis2_steps)),
        )

    def _resolve_n_samples(self, edges: Sequence[AxisEdge], pools: dict[Point, np.ndarray]) -> int:
        if self.hparams.n_samples is not None:
            return int(self.hparams.n_samples)
        sizes = []
        for edge in edges:
            sizes.extend([len(pools[edge.point]), len(pools[edge.axis1_target]), len(pools[edge.axis2_target])])
        return min(sizes)

    @staticmethod
    def _sample_indices(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(indices, size=n, replace=len(indices) < n)

    def _paired_target_indices(
        self,
        x0: np.ndarray,
        source_pool: np.ndarray,
        target_pool: np.ndarray,
        latent: np.ndarray,
        n: int,
        rng: np.random.Generator,
        method: str,
        velocity: Optional[np.ndarray],
        idx0: np.ndarray,
        delta: float,
    ) -> np.ndarray:
        if method == "conditional_fm":
            return self._sample_ot_targets(source_pool, target_pool, latent, idx0, rng)
        if method != "curly_fm":
            raise ValueError(f"Unsupported axis method for pairing: {method}")
        target = latent[target_pool]
        if velocity is None:
            distances = ((x0[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
            return target_pool[np.argmin(distances, axis=1)]
        predicted = x0 + velocity[idx0] * float(delta)
        cost = ((predicted[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        if len(row_ind) < n:
            return target_pool[np.argmin(cost, axis=1)]
        return target_pool[np.asarray(col_ind, dtype=np.int64)]

    def _sample_ot_targets(
        self,
        source_pool: np.ndarray,
        target_pool: np.ndarray,
        latent: np.ndarray,
        idx0: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        src = latent[source_pool]
        dst = latent[target_pool]
        cost = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=-1)
        plan = self._sinkhorn_plan(cost)
        source_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(source_pool.tolist())}
        local_rows = np.asarray([source_lookup[int(global_idx)] for global_idx in idx0], dtype=np.int64)
        row_probs = plan[local_rows]
        row_sums = row_probs.sum(axis=1, keepdims=True)
        sampled = np.empty(idx0.shape[0], dtype=np.int64)
        valid = row_sums[:, 0] > 0.0
        if np.any(valid):
            probs = row_probs[valid] / row_sums[valid]
            cdf = np.cumsum(probs, axis=1)
            draws = rng.random(probs.shape[0])[:, None]
            sampled[valid] = np.argmax(draws <= cdf, axis=1)
        if np.any(~valid):
            fallback_rows = cost[local_rows[~valid]]
            sampled[~valid] = np.argmin(fallback_rows, axis=1)
        return target_pool[sampled]

    @staticmethod
    def _edge_velocity(
        axis_name: str,
        idx0: np.ndarray,
        x0: np.ndarray,
        x_target: np.ndarray,
        velocity_sources: dict[str, Optional[np.ndarray]],
        delta: float,
    ) -> np.ndarray:
        velocity = velocity_sources[axis_name]
        if velocity is not None:
            return velocity[idx0]
        denom = max(float(delta), 1e-6)
        return (x_target - x0) / denom

    def _resolve_velocity_sources(
        self,
        pools: dict[Point, np.ndarray],
        latent: np.ndarray,
        velocity_sources: dict[str, Optional[np.ndarray]],
        axis1_values: Sequence[str],
        axis2_values: Sequence[str],
        axis1_order: dict[str, int],
        axis2_order: dict[str, int],
        axis1_coord: dict[str, float],
        axis2_coord: dict[str, float],
        holdouts: set[Point],
    ) -> dict[str, Optional[np.ndarray]]:
        resolved = dict(velocity_sources)
        if self.hparams.axis1_method == "curly_fm" and resolved["axis1"] is None:
            resolved["axis1"] = self._estimate_axis_velocity(
                0,
                self.hparams.axis1_pseudo_velocity_method,
                pools,
                latent,
                axis1_values,
                axis2_values,
                axis1_order,
                axis2_order,
                axis1_coord,
                axis2_coord,
                holdouts,
            )
        if self.hparams.axis2_method == "curly_fm" and resolved["axis2"] is None:
            resolved["axis2"] = self._estimate_axis_velocity(
                1,
                self.hparams.axis2_pseudo_velocity_method,
                pools,
                latent,
                axis1_values,
                axis2_values,
                axis1_order,
                axis2_order,
                axis1_coord,
                axis2_coord,
                holdouts,
            )
        return resolved

    def _estimate_axis_velocity(
        self,
        axis: int,
        method: str,
        pools: dict[Point, np.ndarray],
        latent: np.ndarray,
        axis1_values: Sequence[str],
        axis2_values: Sequence[str],
        axis1_order: dict[str, int],
        axis2_order: dict[str, int],
        axis1_coord: dict[str, float],
        axis2_coord: dict[str, float],
        holdouts: set[Point],
    ) -> np.ndarray:
        velocity = np.zeros_like(latent, dtype=np.float32)
        counts = np.zeros(latent.shape[0], dtype=np.float32)
        observed = set(pools) - holdouts
        for point, idx_src in pools.items():
            if point not in observed:
                continue
            target_point = self._next_observed_point(
                point,
                axis,
                observed,
                axis1_values,
                axis2_values,
                axis1_order,
                axis2_order,
            )
            if target_point is None:
                continue
            idx_dst = pools[target_point]
            src = latent[idx_src]
            dst = latent[idx_dst]
            delta = self._axis_delta(
                point[axis],
                target_point[axis],
                axis1_coord if axis == 0 else axis2_coord,
            )
            delta = max(float(delta), 1e-6)
            drift = self._estimate_pairwise_drift(src, dst, delta, method)
            velocity[idx_src] += drift
            counts[idx_src] += 1.0
        nonzero = counts > 0
        velocity[nonzero] /= counts[nonzero, None]
        return velocity

    def _estimate_pairwise_drift(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        delta: float,
        method: str,
    ) -> np.ndarray:
        if src.size == 0:
            return np.zeros_like(src, dtype=np.float32)
        distances = ((src[:, None, :] - dst[None, :, :]) ** 2).sum(axis=-1)
        if method == "nearest_forward":
            matched = dst[np.argmin(distances, axis=1)]
            return ((matched - src) / delta).astype(np.float32, copy=False)
        if method == "knn_forward":
            k = min(int(self.hparams.pseudo_velocity_knn_k), dst.shape[0])
            nn_idx = np.argpartition(distances, kth=max(k - 1, 0), axis=1)[:, :k]
            nn_dist = np.take_along_axis(distances, nn_idx, axis=1)
            weights = 1.0 / np.maximum(nn_dist, 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            matched = (dst[nn_idx] * weights[..., None]).sum(axis=1)
            return ((matched - src) / delta).astype(np.float32, copy=False)
        if method == "ot_barycentric":
            plan = self._sinkhorn_plan(distances)
            row_mass = np.maximum(plan.sum(axis=1, keepdims=True), 1e-8)
            matched = (plan @ dst) / row_mass
            return ((matched - src) / delta).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported pseudo-velocity method: {method}")

    def _sinkhorn_plan(self, cost: np.ndarray) -> np.ndarray:
        n_src, n_dst = cost.shape
        a = np.full(n_src, 1.0 / max(n_src, 1), dtype=np.float64)
        b = np.full(n_dst, 1.0 / max(n_dst, 1), dtype=np.float64)
        reg = float(self.hparams.pseudo_velocity_ot_reg)
        scaled = cost.astype(np.float64, copy=False)
        scaled -= scaled.min()
        kernel = np.exp(-scaled / reg)
        kernel = np.maximum(kernel, 1e-12)
        u = np.ones_like(a)
        v = np.ones_like(b)
        tol = float(self.hparams.pseudo_velocity_ot_tol)
        max_iter = int(self.hparams.pseudo_velocity_ot_max_iter)
        for _ in range(max_iter):
            u_prev = u
            kv = kernel @ v
            kv = np.maximum(kv, 1e-12)
            u = a / kv
            ktu = kernel.T @ u
            ktu = np.maximum(ktu, 1e-12)
            v = b / ktu
            if np.max(np.abs(u - u_prev)) < tol:
                break
        return (u[:, None] * kernel) * v[None, :]

    @staticmethod
    def _next_observed_point(
        point: Point,
        axis: int,
        observed: set[Point],
        axis1_values: Sequence[str],
        axis2_values: Sequence[str],
        axis1_order: dict[str, int],
        axis2_order: dict[str, int],
    ) -> Optional[Point]:
        axis1_idx = axis1_order[point[0]]
        axis2_idx = axis2_order[point[1]]
        if axis == 0:
            for next_idx in range(axis1_idx + 1, len(axis1_values)):
                candidate = (axis1_values[next_idx], point[1])
                if candidate in observed:
                    return candidate
            return None
        for next_idx in range(axis2_idx + 1, len(axis2_values)):
            candidate = (point[0], axis2_values[next_idx])
            if candidate in observed:
                return candidate
        return None

    @staticmethod
    def _numeric_axis_coordinates(values: Sequence[str]) -> dict[str, float]:
        try:
            return {value: float(value) for value in values}
        except ValueError:
            return {value: float(idx) for idx, value in enumerate(values)}

    @staticmethod
    def _axis_delta(start: str, end: str, order: dict[str, float]) -> float:
        if start in order and end in order:
            return float(order[end] - order[start])
        try:
            return float(end) - float(start)
        except ValueError:
            return 1.0 if start != end else 0.0

    @staticmethod
    def _split_lengths(n: int, split: Sequence[float]) -> list[int]:
        if len(split) != 3:
            raise ValueError("train_val_test_split must contain three values.")
        if all(isinstance(v, float) and v <= 1.0 for v in split):
            train = int(n * split[0])
            val = int(n * split[1])
            test = n - train - val
            return [train, val, test]
        lengths = [int(v) for v in split]
        if sum(lengths) != n:
            raise ValueError("Integer train_val_test_split must sum to dataset size.")
        return lengths