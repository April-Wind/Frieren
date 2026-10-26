from __future__ import annotations

import copy
import hashlib

import torch

from .contracts import GraphArtifact


VIEW_NAMES = {
    "FREEDOM": "mm_adj",
    "MENTOR": "mm_adj",
    "MSCA": "struct_original_adj",
    "SMORE": "fusion_adj",
}


def operator_signature(operator: torch.Tensor) -> str:
    
    graph = operator.detach().cpu().coalesce()
    digest = hashlib.sha256()
    digest.update(str(tuple(graph.shape)).encode("ascii"))
    digest.update(graph.indices().numpy().tobytes())
    digest.update(graph.values().numpy().tobytes())
    return digest.hexdigest()


def adapt_graph(artifact: GraphArtifact, backbone: str) -> GraphArtifact:


    name = backbone.upper()
    if name not in VIEW_NAMES:
        raise ValueError(f"Unknown backbone {backbone!r}; choose {tuple(VIEW_NAMES)}")
    artifact.validate()
    if artifact.metadata.get("adapter") is not None:
        raise ValueError("Graph already adapted; pass its original estimator artifact")
    graph = artifact.operator.detach().cpu().coalesce()
    metadata = copy.deepcopy(artifact.metadata)
    source_signature = operator_signature(graph)
    if name == "FREEDOM":
        result = graph
        recipe = "identity"
    else:
        symmetric = (graph + graph.transpose(0, 1)).coalesce()
        degree = torch.sparse.sum(symmetric, dim=1).to_dense().clamp_min(1e-12)
        row, col = symmetric.indices()
        values = symmetric.values() * degree[row].rsqrt() * degree[col].rsqrt()
        result = torch.sparse_coo_tensor(
            symmetric.indices(), values, symmetric.shape
        ).coalesce()
        recipe = "D^-1/2 (A + A.T) D^-1/2; degree floor 1e-12"
    metadata["adapter"] = {
        "backbone": name,
        "view": VIEW_NAMES[name],
        "recipe": recipe,
        "source_operator_signature": source_signature,
        "output_operator_signature": operator_signature(result),
    }
    
    metadata["orientation"] = "receiver_sender"
    output = GraphArtifact(result, artifact.item_ids, metadata)
    output.validate()
    return output


def apply_graph(model, artifact: GraphArtifact, backbone: str) -> GraphArtifact:


    name = backbone.upper()
    if name not in VIEW_NAMES:
        raise ValueError(f"Unknown backbone {backbone!r}")
    view = VIEW_NAMES[name]
    if not hasattr(model, view):
        raise AttributeError(f"{name} does not expose evaluated view {view!r}")
    count = getattr(model, "n_items", getattr(model, "num_item", None))
    original = getattr(model, view)
    if count is None and torch.is_tensor(original):
        count = original.shape[0]
    if count != artifact.n_items or tuple(original.shape) != (count, count):
        raise ValueError("Graph and model item dimensions disagree")
    model_ids = getattr(model, "item_ids", None)
    if model_ids is not None and tuple(map(str, model_ids)) != artifact.item_ids:
        raise ValueError("Model item order differs from the graph artifact")
    if name == "FREEDOM" and hasattr(model, "mm_adjs") and len(model.mm_adjs) != 1:
        raise ValueError("FREEDOM adapter supports the evaluated single-graph mode")
    adapted = adapt_graph(artifact, name)
    device = original.device
    replacement = adapted.operator.to(device)
    setattr(model, view, replacement)
    
    if name == "FREEDOM" and hasattr(model, "mm_adjs"):
        model.mm_adjs = [replacement]
    if hasattr(model, "clear_prediction_cache"):
        model.clear_prediction_cache()
    for cache in ("restore_user_e", "restore_item_e"):
        if hasattr(model, cache):
            setattr(model, cache, None)
    return adapted
