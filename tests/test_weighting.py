import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize

from frieren.contracts import GraphArtifact
from frieren.graph import as_coo_tensor, as_csr
from frieren.weighting import neighborhood_coefficients, refine_weights


def test_bgnw_matches_independent_constrained_optimizer():
    rng = np.random.default_rng(10606)
    for _ in range(8):
        x = rng.integers(0, 2, size=(24, 10)).astype(float)
        x[:, :2] = 0
        target = rng.integers(0, 2, size=24).astype(float)
        gram, c, degree = x.T @ x, x.T @ target, target.sum()
        base, cold = np.full(10, .1, np.float32), np.r_[True, True, np.zeros(8, dtype=bool)]
        weights, residual, _ = neighborhood_coefficients(gram, c, degree, base, cold)
        hessian, linear = np.diag(np.diag(gram)) / degree + np.eye(10), -c / degree
        warm = np.flatnonzero(~cold)
        fixed = np.where(cold, base, 0.0).astype(np.float64)
        def objective(free):
            w = fixed.copy()
            w[warm] = free
            return .5 * w @ hessian @ w + linear @ w
        expected = minimize(objective, np.full(8, .1), method="SLSQP",
                            bounds=[(1e-5, None)] * 8,
                            constraints={"type": "eq", "fun": lambda w: w.sum() - (1 - fixed[cold].sum())},
                            options={"ftol": 1e-12, "maxiter": 250})
        assert expected.success
        assert abs(objective(weights[warm]) - expected.fun) < 1e-8
        assert residual < 1e-7
        np.testing.assert_array_equal(weights[cold], base[cold])
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-8)


def test_bgnw_preserves_support_cold_receivers_and_cold_source_weights():
    n = 12
    rows = np.repeat(np.arange(n), 10)
    columns = np.concatenate([np.delete(np.arange(n), i)[:10] for i in range(n)])
    base = sp.csr_matrix((np.full(len(rows), .1, np.float32), (rows, columns)), shape=(n, n))
    graph = GraphArtifact(as_coo_tensor(base), tuple(str(i) for i in range(n)))
    x = np.random.default_rng(54).integers(0, 2, size=(16, n)).astype(float)
    x[:, -2:] = 0
    weighted = as_csr(refine_weights(graph, sp.csr_matrix(x)).operator)
    np.testing.assert_array_equal(weighted.indptr, base.indptr)
    np.testing.assert_array_equal(weighted.indices, base.indices)
    np.testing.assert_array_equal(weighted[-2:].toarray(), base[-2:].toarray())
    np.testing.assert_array_equal(weighted.data[weighted.indices >= 10], base.data[base.indices >= 10])
    np.testing.assert_allclose(np.asarray(weighted.sum(1)), 1, atol=3e-7)

    source = Path(__file__).resolve().parents[3] / "experiments/u2_conditional_weights_20261006/prepare.py"
    if not source.exists():
        pytest.skip("Archived solver parity is available only in the source workspace")
    spec = importlib.util.spec_from_file_location("canonical_qp", source)
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    degree = x.sum(0)
    for receiver in range(10):
        neighbours = base.indices[base.indptr[receiver]:base.indptr[receiver + 1]]
        z = x[:, neighbours]
        c = z.T @ x[:, receiver]
        expected, _, _ = canonical.coefficients(z.T @ z, c, degree[receiver],
                                                np.full(10, .1, np.float32), degree[neighbours] == 0, True)
        np.testing.assert_array_equal(weighted.data[weighted.indptr[receiver]:weighted.indptr[receiver + 1]],
                                      expected.astype(np.float32))
