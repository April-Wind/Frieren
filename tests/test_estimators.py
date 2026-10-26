import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from frieren.contracts import GraphArtifact
from frieren.estimators import BehaviorEstimator, KnowledgeEstimator, align_entity_embeddings
from frieren.graph import as_coo_tensor


def fallback(n=12):
    rows = np.repeat(np.arange(n), n - 1)
    columns = np.concatenate([np.delete(np.arange(n), row) for row in range(n)])
    matrix = sp.csr_matrix((np.full(len(rows), 1.0 / (n - 1)), (rows, columns)), shape=(n, n))
    return GraphArtifact(as_coo_tensor(matrix), tuple(str(i) for i in range(n)))


def test_ease_constrained_solution_and_orientation():
    x = sp.csr_matrix([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0]], dtype=float)
    model = BehaviorEstimator(regularization=7.0, k=2).fit(x)
    b = model.coefficients_
    gram = (x.T @ x).toarray() + 7 * np.eye(4)
    for target in range(4):
        sources = np.delete(np.arange(4), target)
        expected = np.linalg.solve(gram[np.ix_(sources, sources)], (x.T @ x).toarray()[sources, target])
        np.testing.assert_allclose(b[sources, target], expected, atol=1e-14)
    np.testing.assert_array_equal(np.diag(b), 0)
    np.testing.assert_allclose(model.predict(x), x.toarray() @ b)


    model.coefficients_ = np.array([[0, .2, .6, 0], [.8, 0, .1, 0], [.1, .9, 0, 0], [0, 0, 0, 0]])
    graph = model.build(tuple(str(i) for i in range(4)), fallback(4)).operator.to_dense().numpy()
    assert graph[0, 1] > graph[0, 2]
    assert graph[1, 2] > graph[1, 0]
    np.testing.assert_allclose(graph.sum(1), 1)
    np.testing.assert_array_equal(np.diag(graph), 0)


def test_behavior_partial_and_zero_rows_match_archived_clean_recipe():
    n = 12
    rng = np.random.default_rng(52)
    coefficients = rng.normal(size=(n, n))
    np.fill_diagonal(coefficients, 0)
    coefficients[:, 0] = 0
    coefficients[:, 1] = 0
    coefficients[3, 1] = .7
    model = BehaviorEstimator(k=10)
    model.coefficients_, model.n_items_ = coefficients, n
    fb = fallback(n)
    actual = model.build(fb.item_ids, fb)
    assert actual.metadata["zero_positive_rows"] == 1
    assert actual.metadata["partial_positive_rows"] >= 1
    assert actual.operator._nnz() == n * 10
    np.testing.assert_allclose(actual.operator.to_dense().sum(1).numpy(), 1, atol=1e-7)
    source = Path(__file__).resolve().parents[3] / "experiments/paper_complete_20261010/main_stage/build_graphs.py"
    if not source.exists():
        pytest.skip("Research-snapshot parity is available only in the source workspace")
    spec = importlib.util.spec_from_file_location("canonical_build", source)
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    expected, _ = canonical.ease_operator(coefficients,
                                          sp.csr_matrix(fb.operator.to_dense().numpy()), 10)
    np.testing.assert_array_equal(actual.operator.to_dense().numpy(), expected.toarray())


def test_knowledge_item_mapping_and_self_recipe():
    vectors = np.random.default_rng(6).normal(size=(12, 5)).astype(np.float32)
    ids = tuple(str(i) for i in range(12))
    entities = tuple("entity_" + i for i in ids)
    aligned = align_entity_embeddings(vectors[::-1], entities[::-1], dict(zip(ids, entities)), ids)
    np.testing.assert_array_equal(aligned, vectors)
    included = KnowledgeEstimator(4, True).build(vectors, ids)
    removed = KnowledgeEstimator(4, False).build(vectors, ids)
    assert torch.count_nonzero(included.operator.to_dense().diag()).item() == 12
    assert torch.count_nonzero(removed.operator.to_dense().diag()).item() == 0
    assert torch.unique(included.operator.values()).tolist() == [.25]
    with pytest.raises(ValueError, match="Missing"):
        align_entity_embeddings(vectors, entities, {}, ids)
    with pytest.raises(ValueError, match="order"):
        BehaviorEstimator(k=2).fit(sp.eye(12)).build(ids[::-1], fallback())
