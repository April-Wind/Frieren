from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .contracts import GraphArtifact, InteractionData


def _to_torch(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
    values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


class FreedomConsumer(nn.Module):


    def __init__(
        self,
        data: InteractionData,
        artifact: GraphArtifact,
        embedding_dim: int = 64,
        n_ui_layers: int = 2,
        n_item_layers: int = 1,
        dropout: float = 0.3,
        seed: int = 999,
    ):
        super().__init__()
        artifact.validate()
        if data.n_items != artifact.n_items or data.item_ids != artifact.item_ids:
            raise ValueError("Graph shape or item order differs from interactions")
        if embedding_dim <= 0 or n_ui_layers < 0 or n_item_layers < 0:
            raise ValueError("Invalid embedding dimension or propagation layer count")
        if not 0 <= dropout < 1:
            raise ValueError("UI dropout must lie in [0, 1)")
        self.n_users, self.n_items = data.n_users, data.n_items
        self.item_ids = data.item_ids
        self.embedding_dim = int(embedding_dim)
        self.n_ui_layers, self.n_layers = int(n_ui_layers), int(n_item_layers)
        self.dropout, self.seed = float(dropout), int(seed)
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = data.train.astype(np.float32).tocoo()
        self.register_buffer("norm_adj", self._normalized_ui(), persistent=False)
        edge_indices = torch.from_numpy(np.vstack((self.interaction_matrix.row, self.interaction_matrix.col))).long()
        self.register_buffer("edge_indices", edge_indices, persistent=False)
        self.register_buffer("edge_values", self._normalize_bipartite(edge_indices), persistent=False)
        self.register_buffer("mm_adj", artifact.operator.coalesce().float().detach().clone(), persistent=False)
        self.register_buffer("masked_adj", self.norm_adj, persistent=False)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.dropout_generator = torch.Generator(device="cpu")
        self.dropout_generator.manual_seed(self.seed + 1000003)
        self.last_dropout_edge_indices = None

    @property
    def device(self) -> torch.device:
        return self.user_embedding.weight.device

    def _apply(self, fn, recurse=True):


        previous = self.dropout_generator.device
        super()._apply(fn, recurse)
        if self.device != previous:
            if self.last_dropout_edge_indices is not None:
                raise RuntimeError("Move FREEDOM to its training device before sampling dropout")
            self.dropout_generator = torch.Generator(device=self.device)
            self.dropout_generator.manual_seed(self.seed + 1000003)
        return self

    def _normalized_ui(self) -> torch.Tensor:
        inter = self.interaction_matrix
        rows = np.concatenate((inter.row, inter.col + self.n_users))
        cols = np.concatenate((inter.col + self.n_users, inter.row))
        values = np.ones_like(rows, dtype=np.float32)
        adj = sp.coo_matrix((values, (rows, cols)), shape=(self.n_nodes, self.n_nodes), dtype=np.float32).tocsr()
        degree = np.asarray((adj > 0).sum(axis=1)).ravel() + 1e-7
        inv = np.power(degree, -0.5)
        return _to_torch(sp.diags(inv) @ adj @ sp.diags(inv))

    def _normalize_bipartite(self, indices: torch.Tensor) -> torch.Tensor:
        values = torch.ones_like(indices[0], dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, (self.n_users, self.n_items), device=indices.device)
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=-1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), dim=-1).to_dense()
        return row_sum.pow(-0.5)[indices[0]] * col_sum.pow(-0.5)[indices[1]]

    def pre_epoch_processing(self) -> None:
        if self.dropout <= 0:
            self.masked_adj = self.norm_adj
            self.last_dropout_edge_indices = torch.arange(self.edge_values.numel())
            return
        probability = self.edge_values / (self.edge_values.max() + 1e-8)
        count = max(1, int(probability.numel() * (1.0 - self.dropout)))
        selected = torch.multinomial(probability, count, replacement=False, generator=self.dropout_generator)
        self.last_dropout_edge_indices = selected.detach().cpu()
        retained = self.edge_indices[:, selected].clone()
        weights = self._normalize_bipartite(retained)
        retained[1] += self.n_users
        indices = torch.cat((retained, torch.flip(retained, [0])), dim=1)
        self.masked_adj = torch.sparse_coo_tensor(indices, torch.cat((weights, weights)), self.norm_adj.shape, device=self.device).coalesce()

    def forward(self, adj: torch.Tensor | None = None):
        item_graph = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            item_graph = torch.sparse.mm(self.mm_adj, item_graph)
        embedding = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        layers = [embedding]
        adj = self.norm_adj if adj is None else adj
        for _ in range(self.n_ui_layers):
            embedding = torch.sparse.mm(adj, embedding)
            layers.append(embedding)
        aggregate = torch.stack(layers, dim=1).mean(dim=1)
        users, items = torch.split(aggregate, (self.n_users, self.n_items), dim=0)
        return users, items + item_graph

    @staticmethod
    def bpr_loss(users, positives, negatives):
        difference = torch.sum(users * positives, dim=1) - torch.sum(users * negatives, dim=1)
        return -F.logsigmoid(difference).mean()

    def calculate_loss(self, interaction: torch.Tensor):
        users, positives, negatives = interaction[:3]
        user_vectors, item_vectors = self(self.masked_adj)
        return self.bpr_loss(user_vectors[users], item_vectors[positives], item_vectors[negatives])

    def full_sort_predict(self, users: torch.Tensor):
        user_vectors, item_vectors = self(self.norm_adj)
        return user_vectors[users] @ item_vectors.T

    def replace_graph(self, artifact: GraphArtifact) -> None:
        
        artifact.validate()
        if artifact.n_items != self.n_items or artifact.item_ids != self.item_ids:
            raise ValueError("Replacement graph shape or item order differs")
        self.mm_adj = artifact.operator.coalesce().float().to(self.device)
