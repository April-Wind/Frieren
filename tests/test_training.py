from dataclasses import replace
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from frieren.contracts import GraphArtifact, InteractionData
from frieren.evaluation import evaluate_scores
from frieren.training import TrainConfig, replay_freedom, train_freedom
from frieren import training


def fixture_data_graph():
    pairs = np.asarray([[u, i] for u in range(4) for i in (u, u + 4)], dtype=np.int64)
    matrix = sp.csr_matrix((np.ones(len(pairs)), pairs.T), shape=(4, 8))
    validation = {u: np.asarray([(u + 1) % 4]) for u in range(4)}
    test = {u: np.asarray([(u + 2) % 4]) for u in range(4)}
    data = InteractionData(matrix, pairs, validation, test, tuple(map(str, range(8))))
    rows = torch.arange(8)
    graph = GraphArtifact(torch.sparse_coo_tensor(torch.stack((rows, (rows + 1) % 8)), torch.ones(8), (8, 8)).coalesce(), data.item_ids)
    return data, graph


def test_exact_cpu_resume_and_no_update_replay(tmp_path):
    data, graph = fixture_data_graph()
    config = TrainConfig(embedding_dim=8, epochs=4, batch_size=3, eval_batch_size=2)
    full = train_freedom(data, graph, config, tmp_path / "full")
    train_freedom(data, graph, replace(config, epochs=2), tmp_path / "resume")
    resumed = train_freedom(data, graph, config, tmp_path / "resume", resume=True)
    assert full["validation"] == resumed["validation"]
    assert full["test"] == resumed["test"]
    complete_state = torch.load(tmp_path / "full/last.pt", weights_only=True)
    resumed_state = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert complete_state["history"] == resumed_state["history"]
    for name, value in complete_state["model_state"].items():
        assert torch.equal(value, resumed_state["model_state"][name])
    assert torch.equal(complete_state["rng"]["dropout"], resumed_state["rng"]["dropout"])
    alternate = GraphArtifact(torch.eye(8).to_sparse(), data.item_ids)
    replay = replay_freedom(data, {"trained": graph, "alternate": alternate}, tmp_path / "full/best.pt")
    assert replay["parameter_updates"] == 0
    assert replay["graphs"]["trained"]["validation"] == full["validation"]
    assert replay["graphs"]["trained"]["test"] == full["test"]
    with pytest.raises(FileExistsError):
        train_freedom(data, graph, config, tmp_path / "full")
    with pytest.raises(ValueError, match="only increasing"):
        train_freedom(data, graph, replace(config, dropout=0.1), tmp_path / "resume", resume=True)


def test_es40_resume_preserves_stale_and_stops_at_the_same_epoch(tmp_path, monkeypatch):
    data, graph = fixture_data_graph()
    config = TrainConfig(embedding_dim=4, epochs=1000, patience=40, batch_size=4)
    calls = 0
    interrupt_at = None

    def evaluate(model, data, split, **kwargs):
        nonlocal calls
        if split == "validation":
            calls += 1
            if calls == interrupt_at:
                raise RuntimeError("simulated evaluation interruption")
        return {"metrics": {"recall@20": 0.25}, "evaluated_users": 0}

    monkeypatch.setattr(training, "evaluate_freedom", evaluate)
    full = train_freedom(data, graph, config, tmp_path / "full")
    assert full["epochs_completed"] == 41
    assert full["best_epoch"] == 0
    assert full["stale"] == 40
    assert full["stop_reason"] == "early_stopping"
    calls, interrupt_at = 0, 4
    with pytest.raises(RuntimeError, match="interruption"):
        train_freedom(data, graph, config, tmp_path / "resume")
    stopped = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert stopped["epoch"] == 2 and stopped["stale"] == 2
    interrupt_at = None
    resumed = train_freedom(data, graph, config, tmp_path / "resume", resume=True)
    assert resumed["epochs_completed"] == full["epochs_completed"]
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/last.pt", weights_only=True)
    assert a["stale"] == b["stale"] == 40
    assert a["history"] == b["history"]
    assert a["scheduler_state"] == b["scheduler_state"]
    for name in a["model_state"]:
        assert torch.equal(a["model_state"][name], b["model_state"][name])
    for name in ("torch", "dropout"):
        assert torch.equal(a["rng"][name], b["rng"][name])
    calls = 0
    train_freedom(data, graph, config, tmp_path / "resume", resume=True)
    assert calls == 0


def test_multi_target_metrics_and_train_mask():
    train = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 5))
    scores = torch.tensor([[100.0, 4.0, 3.0, 2.0, 1.0]])
    result = evaluate_scores(lambda users: scores[users], train, {0: np.asarray([1, 3])}, topk=(2, 3), batch_size=1)
    assert result["predictions"] == [[1, 2, 3]]
    assert result["metrics"]["recall@2"] == 0.5
    assert result["metrics"]["recall@3"] == 1.0
    assert result["metrics"]["precision@3"] == pytest.approx(2 / 3)
    assert result["metrics"]["map@3"] == pytest.approx((1 + 2 / 3) / 2)
    expected_ndcg = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert result["metrics"]["ndcg@3"] == pytest.approx(expected_ndcg)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_scores(lambda users: scores[users], train, {0: np.asarray([0])})


def test_training_without_test_and_rejecting_graph_order_drift(tmp_path):
    data, graph = fixture_data_graph()
    without_test = InteractionData(data.train, data.train_pairs, data.validation, {}, data.item_ids)
    result = train_freedom(without_test, graph, TrainConfig(embedding_dim=8, epochs=1, batch_size=4), tmp_path / "no_test")
    assert result["test"] == {}
    assert result["test_evaluated_users"] == 0
    bad = GraphArtifact(graph.operator, tuple(reversed(graph.item_ids)))
    with pytest.raises(ValueError, match="item order"):
        train_freedom(data, bad, TrainConfig(embedding_dim=8, epochs=1), tmp_path / "bad")


def test_training_pair_and_negative_stream_match_canonical_dataloader():
    source = Path(__file__).resolve().parents[3] / "experiments/paper_complete_20261010/main_stage/payload/src/utils/dataloader.py"
    if not source.exists():
        pytest.skip("Historical dataloader is not distributed with the clean release")
    import pandas as pd
    spec = importlib.util.spec_from_file_location("canonical_dataloader", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, _ = fixture_data_graph()
    class Dataset:
        def __init__(self, frame):
            self.df = frame
            self.uid_field, self.iid_field = "user", "item"
            self.inter_num = len(frame)
            self.user_num, self.item_num = data.n_users, data.n_items
        def copy(self, frame):
            return Dataset(frame.copy())
        def shuffle(self):
            self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)
        def __len__(self):
            return len(self.df)
        def __getitem__(self, index):
            return self.df.iloc[index]
    frame = pd.DataFrame(data.train_pairs, columns=("user", "item"))
    config = {"device": "cpu", "use_full_sampling": False, "use_neg_sampling": True,
              "use_neighborhood_loss": False, "USER_ID_FIELD": "user", "ITEM_ID_FIELD": "item"}
    random.seed(999)
    np.random.seed(999)
    loader = module.TrainDataLoader(config, Dataset(frame), batch_size=3, shuffle=True)
    loader.pretrain_setup()


    reference = []
    for _ in range(4):
        epoch_batches = []
        for batch in loader:
            epoch_batches.append(batch.numpy())
        reference.append(epoch_batches)
    random.seed(999)
    np.random.seed(999)
    pool = sorted(set(map(int, data.train_pairs[:, 1])))
    random.shuffle(pool)
    histories = {u: set(data.train[u].indices) for u in range(data.n_users)}
    pairs = data.train_pairs.copy()
    for expected_epoch in reference:
        pairs = pairs[np.random.choice(len(pairs), len(pairs), replace=False)]
        for start, expected in zip(range(0, len(pairs), 3), expected_epoch):
            batch = pairs[start:start + 3]
            negatives = []
            for user in batch[:, 0]:
                item = random.sample(pool, 1)[0]
                while item in histories[int(user)]:
                    item = random.sample(pool, 1)[0]
                negatives.append(item)
            actual = np.vstack((batch.T, np.asarray(negatives)))
            np.testing.assert_array_equal(actual, expected)
