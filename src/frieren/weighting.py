from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .contracts import GraphArtifact
from .estimators.behavior import binary_interactions
from .graph import as_coo_tensor, as_csr, interaction_sha256


def simplex_qp(hessian: np.ndarray, linear: np.ndarray, mass: float) -> tuple:
    
    n = len(linear)
    if n < 1 or mass <= 0:
        raise ValueError("QP requires positive free mass and at least one variable")
    z = np.full(n, mass / n)
    active = np.zeros(n, dtype=bool)
    for iteration in range(128):
        free = np.flatnonzero(~active)
        inverse = np.linalg.solve(hessian[np.ix_(free, free)],
                                  np.column_stack((linear[free], np.ones(len(free)))))
        multiplier = -(mass + inverse[:, 0].sum()) / inverse[:, 1].sum()
        proposal = np.zeros(n)
        proposal[free] = -inverse[:, 0] - multiplier * inverse[:, 1]
        if proposal[free].min() < -1e-11:
            bad = free[proposal[free] < 0]
            step = min(1.0, float(np.min(z[bad] / (z[bad] - proposal[bad]))))
            z += step * (proposal - z)
            hits = (z < 1e-11) & (~active)
            z[hits] = 0.0
            active[hits] = True
            if active.all():
                raise RuntimeError("QP has no feasible free variable")
            continue
        z = np.maximum(proposal, 0.0)
        gradient = hessian @ z + linear + multiplier
        violating = np.flatnonzero(active & (gradient < -1e-9))
        if len(violating):
            active[violating[np.argmin(gradient[violating])]] = False
            continue
        positive = z > 1e-9
        residual = max(
            abs(z.sum() - mass),
            float(np.abs(gradient[positive]).max()) if positive.any() else 0.0,
            float(np.maximum(-gradient[~positive], 0).max()) if (~positive).any() else 0.0,
        )
        if residual >= 1e-7:
            raise RuntimeError(f"QP KKT residual exceeds tolerance: {residual}")
        return z, residual, iteration + 1
    raise RuntimeError("QP did not satisfy KKT within 128 iterations")


def neighborhood_coefficients(
    gram: np.ndarray, cooccurrence: np.ndarray, receiver_degree: float,
    base: np.ndarray, cold: np.ndarray, alpha: float = 1.0, floor: float = 1e-5,
) -> tuple:
    warm = np.flatnonzero(~cold)
    weights = np.asarray(base, dtype=np.float64).copy()
    if len(warm) < 2:
        return weights, 0.0, 0
    hessian = np.diag(np.diag(gram)) / receiver_degree + alpha * np.eye(len(base))
    linear = -cooccurrence / receiver_degree
    fixed = np.flatnonzero(cold)
    mass = 1.0 - weights[fixed].sum()
    warm_hessian = hessian[np.ix_(warm, warm)]
    warm_linear = linear[warm] + warm_hessian @ np.full(len(warm), floor)
    if len(fixed):
        warm_linear += hessian[np.ix_(warm, fixed)] @ weights[fixed]
    solution, residual, iterations = simplex_qp(warm_hessian, warm_linear,
                                               mass - floor * len(warm))
    weights[warm] = solution + floor
    return weights, residual, iterations


def refine_weights(artifact: GraphArtifact, train: sp.spmatrix, alpha: float = 1.0) -> GraphArtifact:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("BGNW alpha must be finite and positive")
    base = as_csr(artifact.operator)
    interactions = binary_interactions(train).tocsc()
    if base.shape != (interactions.shape[1], interactions.shape[1]):
        raise ValueError("Graph and interaction catalogue dimensions differ")
    if not np.isfinite(base.data).all() or (base.data <= 0).any():
        raise ValueError("BGNW requires strictly positive fixed-support edges")
    if not np.allclose(np.asarray(base.sum(1)).ravel(), 1.0, atol=3e-7, rtol=0):
        raise ValueError("BGNW requires a row-normalized input graph")
    if np.count_nonzero(base.diagonal()):
        raise ValueError("BGNW is defined for learned graphs without self edges")
    degree = np.asarray(interactions.sum(axis=0)).ravel()
    cold_items = degree == 0
    result = base.copy()
    changed = optimized = 0
    max_residual = 0.0
    for receiver in range(base.shape[0]):
        lo, hi = base.indptr[receiver : receiver + 2]
        neighbours = base.indices[lo:hi]
        cold = cold_items[neighbours]
        if degree[receiver] == 0 or (~cold).sum() < 2:
            continue
        neighbour_histories = interactions[:, neighbours]
        cooccurrence = np.asarray((neighbour_histories.T @ interactions[:, receiver]).toarray()).ravel()
        if not cooccurrence.any():
            continue
        gram = (neighbour_histories.T @ neighbour_histories).toarray()
        weights, residual, _ = neighborhood_coefficients(
            gram, cooccurrence, degree[receiver], base.data[lo:hi], cold, alpha,
        )
        result.data[lo:hi] = weights.astype(np.float32)
        max_residual = max(max_residual, residual)
        optimized += 1
        changed += bool((np.abs(result.data[lo:hi] - base.data[lo:hi]) > 1e-7).any())
    metadata = dict(artifact.metadata)
    metadata["weight_refinement"] = {
        "method": "BGNW", "alpha": float(alpha), "floor": 1e-5,
        "objective": "0.5*w.T*(diag(Xc.T@Xc)/d+alpha*I)*w - (Xc.T@xi/d).T*w",
        "constraints": "row sum 1; warm weights >= floor; cold-source weights fixed",
        "support_unchanged": True, "cold_receivers_unchanged": True,
        "optimized_rows": optimized, "changed_rows": changed,
        "max_kkt_residual": max_residual, "training_interactions_only": True,
        "training_matrix_sha256": interaction_sha256(interactions),
    }
    return GraphArtifact(operator=as_coo_tensor(result), item_ids=artifact.item_ids,
                         metadata=metadata)
