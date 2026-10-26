from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from ..contracts import GraphArtifact
from ..graph import array_sha256, as_coo_tensor, as_csr, interaction_sha256


def binary_interactions(train: sp.spmatrix) -> sp.csr_matrix:
    result = sp.csr_matrix(train, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError("Interaction matrix must be nonempty")
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError("Interactions must be finite and nonnegative")
    result.data[:] = 1.0
    result.sort_indices()
    return result


def ease_graph(coefficients: np.ndarray, fallback: GraphArtifact, k: int) -> tuple:
    coefficients = np.asarray(coefficients)
    n_items = coefficients.shape[0]
    if coefficients.shape != (n_items, n_items) or not np.isfinite(coefficients).all():
        raise ValueError("EASE coefficients must be a finite square matrix")
    if k < 1 or k >= n_items:
        raise ValueError("EASE k must be positive and smaller than catalogue size")
    graph = as_csr(fallback.operator)
    if graph.shape != coefficients.shape or (graph.data < 0).any():
        raise ValueError("Fallback graph shape/values are invalid")
    if not np.isfinite(graph.data).all():
        raise ValueError("Fallback graph contains nonfinite values")
    rows, columns, values = [], [], []
    zero_rows = partial_rows = fallback_edges = 0
    for receiver in range(n_items):
        incoming = np.asarray(coefficients[:, receiver], dtype=np.float64)
        candidates = np.flatnonzero(incoming > 0)
        candidates = candidates[candidates != receiver]
        order = np.lexsort((candidates, -incoming[candidates]))[:k]
        chosen = candidates[order].tolist()
        weights = incoming[chosen].tolist()
        if len(chosen) < k:
            zero_rows += len(chosen) == 0
            partial_rows += 0 < len(chosen) < k
            lo, hi = graph.indptr[receiver : receiver + 2]
            ids, raw_weights = graph.indices[lo:hi], graph.data[lo:hi]
            order = np.lexsort((ids, -raw_weights))
            seen = set(chosen)
            for offset in order:
                sender = int(ids[offset])
                if sender != receiver and sender not in seen and raw_weights[offset] > 0:
                    chosen.append(sender)
                    weights.append(float(raw_weights[offset]))
                    fallback_edges += 1
                    seen.add(sender)
                if len(chosen) >= k:
                    break
        if not chosen:
            raise ValueError(f"No positive/fallback neighbours for item {receiver}")
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        rows.extend([receiver] * len(chosen))
        columns.extend(chosen)
        values.extend(weights.astype(np.float32))
    matrix = sp.csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)),
                           shape=(n_items, n_items))
    return as_coo_tensor(matrix), {
        "zero_positive_rows": zero_rows, "partial_positive_rows": partial_rows,
        "fallback_edges": fallback_edges,
        "rows_below_requested_k": int((np.diff(matrix.indptr) < k).sum()),
    }


class BehaviorEstimator:
    def __init__(self, regularization: float = 100.0, k: int = 10):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("EASE regularization must be finite and positive")
        self.regularization = float(regularization)
        self.k = k

    def fit(self, train: sp.spmatrix) -> "BehaviorEstimator":
        interactions = binary_interactions(train)
        gram = (interactions.T @ interactions).toarray()
        gram.flat[:: len(gram) + 1] += self.regularization
        precision = la.cho_solve(la.cho_factor(gram, check_finite=False),
                                 np.eye(len(gram)), check_finite=False)
        self.coefficients_ = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.coefficients_, 0.0)
        self.n_items_ = interactions.shape[1]
        self.training_sha256_ = interaction_sha256(interactions)
        return self

    def predict(self, train: sp.spmatrix) -> np.ndarray:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the estimator before predicting")
        matrix = sp.csr_matrix(train)
        if matrix.shape[1] != self.n_items_:
            raise ValueError("Prediction catalogue does not match fitted coefficients")
        return np.asarray(matrix @ self.coefficients_)

    def build(self, item_ids: Sequence[str], fallback: GraphArtifact) -> GraphArtifact:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Fit the estimator before building its graph")
        ids = tuple(str(i) for i in item_ids)
        if len(ids) != self.n_items_ or tuple(fallback.item_ids) != ids:
            raise ValueError("Fitted coefficients and fallback item order must match")
        operator, statistics = ease_graph(self.coefficients_, fallback, self.k)
        return GraphArtifact(operator=operator, item_ids=ids, metadata={
            "estimator": "Frieren-B", "recipe": "ease_clean_v2",
            "regularization": self.regularization, "k": self.k,
            "coefficient_orientation": "source_row_target_column",
            "orientation": "receiver_sender", "include_self": False,
            "normalization": "positive_row_sum", "symmetrized": False,
            "fallback": "explicit supplied multimodal graph; descending weight then item index",
            "fallback_order_sha256": fallback.order_sha256,
            "selection": "descending positive coefficient then item index",
            "training_matrix_sha256": getattr(self, "training_sha256_", None),
            "coefficient_sha256": array_sha256(self.coefficients_),
            **statistics,
        })
