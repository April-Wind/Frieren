from pathlib import Path
import importlib.util
import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch import nn

from frieren.backbones import FreedomConsumer
from frieren.contracts import GraphArtifact, InteractionData


def fixture_data_graph():
    pairs = np.asarray([[0, 0], [0, 2], [1, 1], [1, 3], [2, 0], [2, 4]], dtype=np.int64)
    matrix = sp.csr_matrix((np.ones(len(pairs)), pairs.T), shape=(3, 5))
    data = InteractionData(matrix, pairs, {0: np.array([1]), 1: np.array([2]), 2: np.array([3])}, {}, tuple(map(str, range(5))))
    indices = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    graph = GraphArtifact(torch.sparse_coo_tensor(indices, torch.ones(5), (5, 5)).coalesce(), data.item_ids)
    return data, graph


def _canonical_model(tmp_path, monkeypatch, data, graph):
    source_root = Path(__file__).resolve().parents[3] / "experiments/paper_complete_20261010/main_stage/payload/src"
    source = source_root / "models/freedom.py"
    if not source.exists():
        pytest.skip("Historical source snapshot is not distributed with the clean release")
    class GeneralRecommender(nn.Module):
        def __init__(self, config, dataset):
            super().__init__()
            self.n_users, self.n_items, self.device = data.n_users, data.n_items, torch.device("cpu")
            self.v_feat = self.t_feat = None
    abstract = types.ModuleType("common.abstract_recommender")
    abstract.GeneralRecommender = GeneralRecommender
    content = types.ModuleType("models.content_adapter")
    content.ContentMessageAdapter = object
    monkeypatch.setitem(sys.modules, "common", types.ModuleType("common"))
    monkeypatch.setitem(sys.modules, "common.abstract_recommender", abstract)
    monkeypatch.setitem(sys.modules, "models", types.ModuleType("models"))
    monkeypatch.setitem(sys.modules, "models.content_adapter", content)
    fusion_spec = importlib.util.spec_from_file_location("models.graph_fusion", source_root / "models/graph_fusion.py")
    fusion = importlib.util.module_from_spec(fusion_spec)
    fusion_spec.loader.exec_module(fusion)
    monkeypatch.setitem(sys.modules, "models.graph_fusion", fusion)
    model_spec = importlib.util.spec_from_file_location("canonical_freedom", source)
    canonical = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(canonical)
    graph_path = tmp_path / "graph.pt"
    torch.save(graph.operator, graph_path)
    config = {
        "embedding_size": 8, "feat_embed_dim": 8, "knn_k": 1, "n_mm_layers": 2,
        "n_ui_layers": 2, "reg_weight": 0, "mm_image_weight": 0.1, "dropout": 0.3,
        "graph_fusion_mode": "single", "fusion_hidden_dim": 16, "data_path": str(tmp_path),
        "dataset": "fixture", "mm_adj_files": [str(graph_path)], "mm_adj_file": "",
        "content_mode": "M0", "seed": 999, "dropout_seed_offset": 1000003,
    }
    dataset = types.SimpleNamespace(inter_matrix=lambda form: data.train.tocoo())
    torch.manual_seed(999)
    return canonical.FREEDOM(config, dataset)


def test_forward_gradients_and_dropout_match_canonical_m0(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data, graph = fixture_data_graph()
    reference = _canonical_model(tmp_path, monkeypatch, data, graph)
    torch.manual_seed(999)
    model = FreedomConsumer(data, graph, embedding_dim=8, n_item_layers=2, dropout=0.3)
    for ref_parameter, parameter in zip(reference.parameters(), model.parameters()):
        torch.testing.assert_close(parameter, ref_parameter, rtol=0, atol=0)
    torch.testing.assert_close(model.norm_adj.to_dense(), reference.norm_adj.to_dense(), rtol=0, atol=0)
    batch = torch.tensor([[0, 1, 2], [0, 1, 4], [4, 0, 1]])
    for _ in range(3):
        reference.pre_epoch_processing()
        model.pre_epoch_processing()
        assert torch.equal(model.last_dropout_edge_indices, reference.last_dropout_edge_indices)
        torch.testing.assert_close(model.masked_adj.to_dense(), reference.masked_adj.to_dense(), rtol=0, atol=0)
        expected = reference.forward(reference.masked_adj)
        actual = model(model.masked_adj)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=0, atol=0)
        reference.zero_grad()
        model.zero_grad()
        reference.calculate_loss(batch).backward()
        model.calculate_loss(batch).backward()
        for ref_parameter, parameter in zip(reference.parameters(), model.parameters()):
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)


def test_graph_swap_changes_propagation_without_parameters_or_dropout_state():
    data, graph = fixture_data_graph()
    torch.manual_seed(999)
    model = FreedomConsumer(data, graph, embedding_dim=8)
    before = model()[1].detach().clone()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    rng = model.dropout_generator.get_state().clone()
    replacement = GraphArtifact(torch.eye(5).to_sparse(), data.item_ids)
    model.replace_graph(replacement)
    after = model()[1]
    assert not torch.equal(before, after)
    for name, value in model.named_parameters():
        assert torch.equal(parameters[name], value)
    assert torch.equal(rng, model.dropout_generator.get_state())
