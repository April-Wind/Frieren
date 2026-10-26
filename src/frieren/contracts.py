from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from scipy import sparse
import torch
from .io import atomic_json, atomic_torch_save, file_sha256, item_order_sha256


@dataclass
class InteractionData:
    train: sparse.csr_matrix
    train_pairs: np.ndarray
    validation: dict
    test: dict
    item_ids: tuple
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.train = sparse.csr_matrix(self.train, dtype=np.float32)
        self.train.sum_duplicates(); self.train.sort_indices()
        self.train_pairs = np.asarray(self.train_pairs, dtype=np.int64)
        self.item_ids = tuple(str(x) for x in self.item_ids)
        if self.train.shape[1] != len(self.item_ids) or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("Item mapping must uniquely match the interaction columns")
        if self.train_pairs.ndim != 2 or self.train_pairs.shape[1] != 2:
            raise ValueError("train_pairs must have shape [number of interactions, 2]")
        if not np.isfinite(self.train.data).all() or (self.train.data <= 0).any():
            raise ValueError("Training interactions must be finite and positive")
        self.train.data[:] = 1
        if len(self.train_pairs):
            users, items = self.train_pairs.T
            if (users < 0).any() or (users >= self.n_users).any() or (items < 0).any() or (items >= self.n_items).any():
                raise ValueError("Training pair outside the declared index range")
            pairs = sparse.csr_matrix((np.ones(len(users)), (users, items)), shape=self.train.shape)
            pairs.data[:] = 1
            if (pairs != self.train).nnz:
                raise ValueError("train_pairs and training matrix differ")
        elif self.train.nnz:
            raise ValueError("train_pairs are missing")
        for split in (self.validation, self.test):
            for user, items in split.items():
                ids = np.asarray(items, dtype=np.int64)
                if not 0 <= int(user) < self.n_users or ids.ndim != 1 or (ids < 0).any() or (ids >= self.n_items).any():
                    raise ValueError("Held-out target outside the declared index range")
                if self.train[int(user), ids].nnz:
                    raise ValueError("Held-out targets overlap training interactions")

    @property
    def n_users(self): return self.train.shape[0]
    @property
    def n_items(self): return self.train.shape[1]
    @property
    def warm_items(self): return np.asarray(self.train.sum(axis=0)).ravel() > 0
    @property
    def order_sha256(self): return item_order_sha256(self.item_ids)


@dataclass
class GraphArtifact:
    operator: torch.Tensor
    item_ids: tuple
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.item_ids = tuple(str(x) for x in self.item_ids)
        self.metadata = dict(self.metadata)
        if self.operator.layout == torch.sparse_coo:
            self.operator = self.operator.detach().coalesce()
        self.validate()

    @property
    def n_items(self): return len(self.item_ids)
    @property
    def order_sha256(self): return item_order_sha256(self.item_ids)

    def validate(self):
        if self.operator.layout != torch.sparse_coo or self.operator.ndim != 2:
            raise ValueError("A graph artifact must be a sparse COO matrix")
        if tuple(self.operator.shape) != (self.n_items, self.n_items):
            raise ValueError("Graph dimensions do not match the item mapping")
        if len(set(self.item_ids)) != self.n_items:
            raise ValueError("Item identifiers must be unique")
        values = self.operator.coalesce().values()
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Frozen propagation graphs require finite nonnegative weights")
        orientation = self.metadata.get("orientation", "receiver_sender")
        if orientation != "receiver_sender":
            raise ValueError("Graph rows must be message receivers, columns senders")
        return self

    def save(self, path):
        path = Path(path)
        self.validate()
        atomic_torch_save(path, self.operator.detach().cpu().coalesce())
        atomic_json(path.with_suffix(path.suffix + ".json"), {
            "schema": "frieren.graph.v1", "item_ids": list(self.item_ids),
            "item_order_sha256": self.order_sha256, "operator_sha256": file_sha256(path),
            "shape": list(self.operator.shape), "orientation": "receiver_sender",
            "metadata": self.metadata,
        })
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if meta.get("schema") != "frieren.graph.v1" or meta.get("orientation") != "receiver_sender":
            raise ValueError("Unsupported graph metadata")
        if file_sha256(path) != meta["operator_sha256"]:
            raise ValueError("Graph tensor hash does not match its metadata")
        graph = cls(torch.load(path, map_location="cpu", weights_only=True), tuple(meta["item_ids"]), meta["metadata"])
        if graph.order_sha256 != meta["item_order_sha256"] or list(graph.operator.shape) != meta["shape"]:
            raise ValueError("Graph metadata and tensor dimensions/order differ")
        return graph
