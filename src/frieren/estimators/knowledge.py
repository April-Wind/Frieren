from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from ..contracts import GraphArtifact
from ..graph import array_sha256, cosine_knn


def align_entity_embeddings(
    embeddings: np.ndarray,
    entity_ids: Sequence[str],
    item_to_entity: Mapping[str, str],
    item_ids: Sequence[str],
) -> np.ndarray:
    
    vectors = np.asarray(embeddings)
    entities = tuple(str(i) for i in entity_ids)
    items = tuple(str(i) for i in item_ids)
    if vectors.ndim != 2 or len(vectors) != len(entities):
        raise ValueError("Entity IDs and embedding rows do not match")
    if len(set(entities)) != len(entities) or len(set(items)) != len(items):
        raise ValueError("Entity/item IDs must be unique")
    index = {entity: row for row, entity in enumerate(entities)}
    missing_items = [i for i in items if i not in item_to_entity]
    if missing_items:
        raise ValueError(f"Missing item-to-entity mappings: {missing_items[:5]}")
    mapped = [str(item_to_entity[i]) for i in items]
    missing_entities = [i for i in mapped if i not in index]
    if missing_entities:
        raise ValueError(f"Mapped entities absent from embeddings: {missing_entities[:5]}")
    return np.asarray(vectors[[index[i] for i in mapped]], dtype=np.float32)


class KnowledgeEstimator:
    def __init__(self, k: int = 10, include_self: bool = True):
        self.k = k
        self.include_self = include_self

    def build(self, embeddings: np.ndarray, item_ids: Sequence[str]) -> GraphArtifact:
        ids = tuple(str(i) for i in item_ids)
        if len(embeddings) != len(ids):
            raise ValueError("Knowledge vectors and item IDs do not match")
        operator = cosine_knn(embeddings, self.k, self.include_self)
        return GraphArtifact(operator=operator, item_ids=ids, metadata={
            "estimator": "Frieren-K", "recipe": "knowledge_cosine_equal_v1",
            "orientation": "receiver_sender", "k": self.k,
            "include_self": self.include_self, "edge_weight": "1/k",
            "selection": "torch_topk", "symmetrized": False,
            "embedding_sha256": array_sha256(np.asarray(embeddings, dtype=np.float32)),
            "upstream": "externally pretrained, item-aligned knowledge embeddings",
        })
