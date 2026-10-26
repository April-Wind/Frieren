from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from frieren.adapters import adapt_graph, apply_graph, operator_signature
from frieren.contracts import GraphArtifact


def fixture_graph():
    tensor = torch.sparse_coo_tensor(torch.tensor([[0, 1, 2], [1, 0, 1]]), torch.tensor([0.8, 0.3, 1.0]), (3, 3)).coalesce()
    return GraphArtifact(tensor, ("a", "b", "c"), {"recipe": "fixture"})


def test_identity_and_transfer_match_frozen_source(tmp_path, monkeypatch):
    artifact = fixture_graph()
    identity = adapt_graph(artifact, "FREEDOM")
    assert operator_signature(identity.operator) == operator_signature(artifact.operator)
    source = Path(__file__).resolve().parents[3] / "experiments/paper_complete_20261010/repair_stage/payload/sources/msca/src/models/behavior_graph.py"
    if not source.exists():
        pytest.skip("Historical source snapshot is not distributed with the clean release")
    spec = importlib.util.spec_from_file_location("canonical_transfer_adapter", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "source.pt"
    torch.save(artifact.operator, path)
    monkeypatch.delenv("SOTA_RUN_DIR", raising=False)
    reference = module.replace_one_view({"behavior_graph_path": str(path), "device": "cpu"}, artifact.operator, 3)
    for name, view in (("MENTOR", "mm_adj"), ("MSCA", "struct_original_adj"), ("SMORE", "fusion_adj")):
        output = adapt_graph(artifact, name)
        torch.testing.assert_close(output.operator.to_dense(), reference.to_dense(), rtol=0, atol=0)
        assert output.metadata["adapter"]["view"] == view
        assert not output.operator.to_dense().diagonal().any()
        with pytest.raises(ValueError, match="already adapted"):
            adapt_graph(output, name)


def test_apply_replaces_only_the_evaluated_view_and_clears_cache():
    artifact = fixture_graph()
    image = torch.eye(3).to_sparse()
    model = SimpleNamespace(n_items=3, item_ids=("a", "b", "c"), fusion_adj=image, image_original_adj=image, restore_item_e=torch.ones(3, 2))
    applied = apply_graph(model, artifact, "SMORE")
    assert model.image_original_adj is image
    torch.testing.assert_close(model.fusion_adj.to_dense(), applied.operator.to_dense())
    assert model.restore_item_e is None
    with pytest.raises(AttributeError):
        apply_graph(model, artifact, "MSCA")
    model.item_ids = ("c", "b", "a")
    with pytest.raises(ValueError, match="item order"):
        apply_graph(model, artifact, "SMORE")


def test_wrong_shapes_and_multigraph_consumer_are_rejected_before_mutation():
    artifact = fixture_graph()
    original = torch.eye(3).to_sparse()
    model = SimpleNamespace(n_items=3, mm_adj=original, mm_adjs=[original, original])
    with pytest.raises(ValueError, match="single-graph"):
        apply_graph(model, artifact, "FREEDOM")
    assert model.mm_adj is original
    with pytest.raises(ValueError, match="dimensions"):
        apply_graph(SimpleNamespace(n_items=4, fusion_adj=torch.eye(4)), artifact, "SMORE")
