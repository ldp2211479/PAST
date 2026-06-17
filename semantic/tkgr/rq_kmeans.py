from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


EPS = 1e-12


def _as_float32_2d(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError("Expected a numpy.ndarray.")
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={array.shape}.")
    return array.astype(np.float32, copy=False)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def _squared_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2x@y^T
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    distances = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(distances, 0.0)


def _argmin_in_chunks(
    x: np.ndarray,
    centroids: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    assignments = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        dists = _squared_distance_matrix(x[start:end], centroids)
        assignments[start:end] = np.argmin(dists, axis=1)
    return assignments


@dataclass
class LayerMetrics:
    coverage: float
    entropy: float
    mean_residual_norm: float


class MiniBatchKMeans:
    """
    Numpy mini-batch K-Means following the update rule used in GRID's implementation:
    centroid <- centroid + eta * (batch_mean - centroid), eta = n_batch_cluster / n_seen_cluster.
    """

    def __init__(
        self,
        n_clusters: int,
        batch_size: int = 1024,
        max_epochs: int = 20,
        max_steps: int | None = None,
        init_buffer_size: int = 4096,
        seed: int = 42,
        distance_chunk_size: int = 4096,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.init_buffer_size = init_buffer_size
        self.seed = seed
        self.distance_chunk_size = distance_chunk_size

        self.centroids: np.ndarray | None = None
        self.cluster_counts: np.ndarray | None = None

    def _init_kmeans_pp(self, buffer: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, dim = buffer.shape
        if n_samples < self.n_clusters:
            raise ValueError(
                "Initialization buffer is smaller than number of clusters: "
                f"{n_samples} < {self.n_clusters}."
            )

        centroids = np.zeros((self.n_clusters, dim), dtype=np.float32)
        first_idx = int(rng.integers(0, n_samples))
        centroids[0] = buffer[first_idx]

        min_distances = _squared_distance_matrix(buffer, centroids[0:1]).reshape(-1)
        for idx in range(1, self.n_clusters):
            probs = np.maximum(min_distances, 0.0)
            total = float(probs.sum())
            if total <= EPS:
                chosen = int(rng.integers(0, n_samples))
            else:
                probs = probs / total
                chosen = int(rng.choice(n_samples, p=probs))
            centroids[idx] = buffer[chosen]
            new_dist = _squared_distance_matrix(buffer, centroids[idx : idx + 1]).reshape(-1)
            min_distances = np.minimum(min_distances, new_dist)

        return centroids

    def fit(self, x: np.ndarray) -> "MiniBatchKMeans":
        x = _as_float32_2d(x)
        n_samples = x.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_clusters ({self.n_clusters})."
            )

        rng = np.random.default_rng(self.seed)
        if n_samples <= self.init_buffer_size:
            init_buffer = x
        else:
            indices = rng.choice(n_samples, size=self.init_buffer_size, replace=False)
            init_buffer = x[indices]

        self.centroids = self._init_kmeans_pp(init_buffer, rng)
        self.cluster_counts = np.zeros(self.n_clusters, dtype=np.float64)

        if self.max_steps is not None:
            n_steps = self.max_steps
        else:
            batches_per_epoch = int(math.ceil(n_samples / max(self.batch_size, 1)))
            n_steps = max(1, self.max_epochs * batches_per_epoch)

        for _ in range(n_steps):
            batch_indices = rng.integers(0, n_samples, size=self.batch_size, endpoint=False)
            batch = x[batch_indices]
            assignments = self.predict(batch)
            unique_ids = np.unique(assignments)
            for cluster_id in unique_ids:
                mask = assignments == cluster_id
                n_points = int(mask.sum())
                if n_points == 0:
                    continue
                target = batch[mask].mean(axis=0)
                self.cluster_counts[cluster_id] += n_points
                eta = n_points / self.cluster_counts[cluster_id]
                self.centroids[cluster_id] += eta * (target - self.centroids[cluster_id])

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Model is not fitted yet.")
        x = _as_float32_2d(x)
        return _argmin_in_chunks(x, self.centroids, chunk_size=self.distance_chunk_size)

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.predict(x)


class RQKMeans:
    """
    Residual quantization with per-layer mini-batch K-Means.

    This mirrors the core logic used in GRID's `ResidualQuantization` + `MiniBatchKMeans`
    path, while keeping dependencies lightweight (numpy only).
    """

    def __init__(
        self,
        num_levels: int = 4,
        codebook_size: int = 256,
        batch_size: int = 1024,
        max_epochs: int = 20,
        max_steps_per_level: int | None = None,
        init_buffer_size: int = 4096,
        normalize_residuals: bool = True,
        seed: int = 42,
        distance_chunk_size: int = 4096,
    ) -> None:
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1.")
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_steps_per_level = max_steps_per_level
        self.init_buffer_size = init_buffer_size
        self.normalize_residuals = normalize_residuals
        self.seed = seed
        self.distance_chunk_size = distance_chunk_size

        self.codebooks: List[np.ndarray] = []
        self.layer_metrics: List[LayerMetrics] = []
        self.fitted_: bool = False

    def _build_layer_model(self, level: int) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=self.codebook_size,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps_per_level,
            init_buffer_size=self.init_buffer_size,
            seed=self.seed + level,
            distance_chunk_size=self.distance_chunk_size,
        )

    def _compute_layer_metrics(self, ids: np.ndarray, residuals: np.ndarray) -> LayerMetrics:
        unique_ids, counts = np.unique(ids, return_counts=True)
        coverage = float(unique_ids.size / self.codebook_size)
        probs = counts.astype(np.float64) / counts.sum()
        entropy = float(-(probs * np.log(np.maximum(probs, EPS))).sum())
        mean_residual_norm = float(np.linalg.norm(residuals, axis=1).mean())
        return LayerMetrics(
            coverage=coverage,
            entropy=entropy,
            mean_residual_norm=mean_residual_norm,
        )

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = _as_float32_2d(x)
        n_samples = x.shape[0]

        self.codebooks = []
        self.layer_metrics = []
        codes = np.zeros((n_samples, self.num_levels), dtype=np.int64)

        residuals = x.copy()
        for level in range(self.num_levels):
            current = _l2_normalize_rows(residuals) if self.normalize_residuals else residuals
            layer_model = self._build_layer_model(level)
            layer_ids = layer_model.fit_predict(current)
            codebook = layer_model.centroids
            if codebook is None:
                raise RuntimeError(f"Layer {level} failed to produce centroids.")
            quantized = codebook[layer_ids]

            residuals = current - quantized
            codes[:, level] = layer_ids

            self.codebooks.append(codebook.astype(np.float32, copy=True))
            self.layer_metrics.append(self._compute_layer_metrics(layer_ids, residuals))

        self.fitted_ = True
        return codes

    def fit(self, x: np.ndarray) -> "RQKMeans":
        self.fit_transform(x)
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        x = _as_float32_2d(x)
        codes = np.zeros((x.shape[0], self.num_levels), dtype=np.int64)
        residuals = x.copy()
        for level, codebook in enumerate(self.codebooks):
            current = _l2_normalize_rows(residuals) if self.normalize_residuals else residuals
            layer_ids = _argmin_in_chunks(
                current,
                codebook,
                chunk_size=self.distance_chunk_size,
            )
            quantized = codebook[layer_ids]
            residuals = current - quantized
            codes[:, level] = layer_ids
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        if codes.ndim != 2 or codes.shape[1] != self.num_levels:
            raise ValueError(
                f"Expected codes shape (n, {self.num_levels}), got {codes.shape}."
            )
        n_samples = codes.shape[0]
        dim = self.codebooks[0].shape[1]
        reconstructed = np.zeros((n_samples, dim), dtype=np.float32)
        for level, codebook in enumerate(self.codebooks):
            reconstructed += codebook[codes[:, level]]
        return reconstructed

    def save(self, output_dir: str | Path) -> None:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        codebook_tensor = np.stack(self.codebooks, axis=0)
        np.savez_compressed(output_dir / "rqkmeans_model.npz", codebooks=codebook_tensor)

        metadata = {
            "num_levels": self.num_levels,
            "codebook_size": self.codebook_size,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "max_steps_per_level": self.max_steps_per_level,
            "init_buffer_size": self.init_buffer_size,
            "normalize_residuals": self.normalize_residuals,
            "seed": self.seed,
            "distance_chunk_size": self.distance_chunk_size,
            "layer_metrics": [asdict(metric) for metric in self.layer_metrics],
        }
        (output_dir / "rqkmeans_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> "RQKMeans":
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "rqkmeans_metadata.json").read_text(encoding="utf-8"))
        stored = np.load(model_dir / "rqkmeans_model.npz")
        codebook_tensor = stored["codebooks"]

        model = cls(
            num_levels=int(metadata["num_levels"]),
            codebook_size=int(metadata["codebook_size"]),
            batch_size=int(metadata["batch_size"]),
            max_epochs=int(metadata["max_epochs"]),
            max_steps_per_level=metadata["max_steps_per_level"],
            init_buffer_size=int(metadata["init_buffer_size"]),
            normalize_residuals=bool(metadata["normalize_residuals"]),
            seed=int(metadata["seed"]),
            distance_chunk_size=int(metadata["distance_chunk_size"]),
        )
        model.codebooks = [codebook_tensor[level].astype(np.float32) for level in range(codebook_tensor.shape[0])]
        model.layer_metrics = [LayerMetrics(**metric) for metric in metadata.get("layer_metrics", [])]
        model.fitted_ = True
        return model


def codes_to_strings(codes: np.ndarray, sep: str = "-") -> List[str]:
    if codes.ndim != 2:
        raise ValueError("codes must be a 2D array.")
    return [sep.join(str(int(token)) for token in row) for row in codes]


def build_code_to_entity_index(
    entity_ids: Sequence[int],
    code_strings: Sequence[str],
) -> Dict[str, List[int]]:
    if len(entity_ids) != len(code_strings):
        raise ValueError("entity_ids and code_strings must have the same length.")
    mapping: Dict[str, List[int]] = {}
    for entity_id, code in zip(entity_ids, code_strings):
        mapping.setdefault(code, []).append(int(entity_id))
    return mapping


def unique_code_rate(codes: np.ndarray) -> float:
    if codes.ndim != 2:
        raise ValueError("codes must be a 2D array.")
    if codes.shape[0] == 0:
        return 0.0
    unique = np.unique(codes, axis=0).shape[0]
    return float(unique / codes.shape[0])

