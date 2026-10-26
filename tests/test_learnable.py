from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from frieren.contracts import InteractionData
from frieren.estimators import learnable
from frieren.estimators.learnable import (
    LearnableEstimatorConfig, SidePredictor, batch_candidates, direct_item_vectors,
    estimator_losses, export_learned_graph, prepare_multimodal_features, train_learnable,
)


def fixture():
    pairs = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 3], [1, 4], [1, 5]])
    x = sp.csr_matrix((np.ones(len(pairs)), pairs.T), shape=(2, 24))
    data = InteractionData(x, pairs, {0: np.array([4]), 1: np.array([0])},
                           {0: np.array([5]), 1: np.array([2])}, tuple(str(i) for i in range(24)))
    features = np.random.default_rng(12).normal(size=(24, 6)).astype(np.float32)
    return data, features


def test_frozen_features_joint_export_and_task_mask():
    data, features = fixture()
    model = SidePredictor(2, 24, features, LearnableEstimatorConfig())
    _, behavior, content = model.vectors()
    fused = direct_item_vectors(behavior, content, data.train)
    torch.testing.assert_close(fused[6:], content[6:])
    torch.testing.assert_close(fused[:6], torch.nn.functional.normalize(behavior[:6] + content[:6], dim=1))
    users = torch.tensor([0, 1])
    candidates = torch.tensor([[0, 4, 7], [1, 0, 8]])
    scores = model.candidate_scores(users, candidates, torch.ones(2, dtype=torch.bool))
    a, _, z = model.vectors()
    torch.testing.assert_close(scores, (a[users, None] * z[candidates]).sum(-1) / .2)
    estimator_losses(model, users, candidates)["total"].backward()
    assert not model.feature.requires_grad
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())


def test_initialization_sampling_scores_and_gradients_match_canonical_source():
    source = Path(__file__).resolve().parents[3] / "experiments/paper_complete_20261010/est_stage/train_estimator.py"
    if not source.exists():
        pytest.skip("Canonical source parity available only in the source workspace")
    spec = importlib.util.spec_from_file_location("canonical_estimator", source)
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    data, features = fixture()
    torch.manual_seed(999)
    expected = canonical.SidePredictor(2, 24, features)
    torch.manual_seed(999)
    actual = SidePredictor(2, 24, features, LearnableEstimatorConfig())
    for key, parameter in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], parameter, atol=0, rtol=0)
    users = np.array([0, 1, 0, 1])
    rng1, rng2 = np.random.default_rng(999), np.random.default_rng(999)
    c1 = canonical.batch_candidates_original(data.train, users, rng1)
    c2 = batch_candidates(data.train, users, rng2)
    np.testing.assert_array_equal(c1, c2)
    assert rng1.bit_generator.state == rng2.bit_generator.state
    uid, candidates, mask = torch.from_numpy(users), torch.from_numpy(c1), torch.tensor([True, False, True, False])
    score = expected.candidate_scores(uid, candidates, mask)
    unique = torch.unique(candidates[:, 0])
    z = torch.nn.functional.normalize(expected.encoder(expected.feature[unique]), dim=-1, eps=1e-8)
    v = torch.nn.functional.normalize(expected.item(unique), dim=-1, eps=1e-8)
    logits = z @ v.T / .2
    target = torch.arange(len(unique))
    align = (torch.nn.functional.cross_entropy(logits, target) +
             torch.nn.functional.cross_entropy(logits.T, target)) / 2
    reg = 1e-4 * (expected.user(uid).square().mean() + expected.item(candidates).square().mean() +
                  sum(p.square().mean() for p in expected.encoder.parameters()))
    loss = torch.nn.functional.cross_entropy(score, torch.zeros(len(users), dtype=torch.long)) + .1 * align + reg
    values = estimator_losses(actual, uid, candidates, mask)
    torch.testing.assert_close(values["total"], loss, atol=0, rtol=0)
    loss.backward()
    values["total"].backward()
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


def test_training_resume_matches_uninterrupted_run_and_exports_direct_only(tmp_path, monkeypatch):
    data, features = fixture()
    config = LearnableEstimatorConfig(dim=4, hidden=8, batch=3, negative=4, epochs=3, patience=10)
    evaluation = learnable._evaluate
    test_calls = []
    def observed_evaluation(model, train, targets, mode, *args, **kwargs):
        if targets is data.test:
            test_calls.append(mode)
        return evaluation(model, train, targets, mode, *args, **kwargs)
    monkeypatch.setattr(learnable, "_evaluate", observed_evaluation)
    uninterrupted = train_learnable(data, features, config, tmp_path / "full")
    assert test_calls == ["direct"]
    original = learnable.batch_candidates
    counter = 0
    def interrupted(*args, **kwargs):
        nonlocal counter
        counter += 1
        if counter == 4:
            raise RuntimeError("simulated process interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(learnable, "batch_candidates", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        train_learnable(data, features, config, tmp_path / "resumed")
    monkeypatch.setattr(learnable, "batch_candidates", original)
    resumed = train_learnable(data, features, config, tmp_path / "resumed", resume=True)
    assert test_calls == ["direct", "direct"]
    assert resumed["history"] == uninterrupted["history"]
    assert resumed["test_at_best_valid"] == uninterrupted["test_at_best_valid"]
    assert resumed["test_evaluations"] == 1
    for name in ("last.pt", "direct_best.pt"):
        a = torch.load(tmp_path / "full" / name, weights_only=False)
        b = torch.load(tmp_path / "resumed" / name, weights_only=False)
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key], atol=0, rtol=0)
    graph = export_learned_graph(data, features, tmp_path / "full/direct_best.pt")
    assert graph.operator._nnz() == 24 * 10
    assert graph.operator.to_dense().diag().count_nonzero() == 0
    assert graph.metadata["representation"] == "warm_normalize_v_plus_z_cold_z"
    with pytest.raises(ValueError, match="direct-selected"):
        export_learned_graph(data, features, tmp_path / "full/side_best.pt")
    with pytest.raises(ValueError, match="identity"):
        train_learnable(data, features + .01, config, tmp_path / "full", resume=True)
    with pytest.raises(ValueError, match="identity"):
        train_learnable(data, features, replace(config, lr=.01), tmp_path / "full", resume=True)


def test_multimodal_input_preparation():
    image = np.array([[3, 4], [0, 0]], dtype=np.float32)
    text = np.array([[0, 2], [2, 0]], dtype=np.float32)
    prepared = prepare_multimodal_features(image, text)
    np.testing.assert_allclose(prepared[0], np.array([.6, .8, 0, 1]) / np.sqrt(2), atol=1e-7)
    assert np.isfinite(prepared).all()
