from __future__ import annotations

import hashlib
import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def interaction_sha256(matrix: sp.spmatrix) -> str:
    value = matrix.tocsr(copy=True)
    value.sort_indices()
    digest = hashlib.sha256(str(value.shape).encode())
    for array in (value.indptr, value.indices, value.data):
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def as_csr(operator: torch.Tensor) -> sp.csr_matrix:
    if not isinstance(operator, torch.Tensor) or operator.layout != torch.sparse_coo:
        raise TypeError("Expected a sparse COO propagation operator")
    graph = operator.detach().cpu().coalesce()
    indices = graph.indices().numpy()
    result = sp.csr_matrix((graph.values().numpy(), indices), shape=tuple(graph.shape))
    result.sort_indices()
    return result


def as_coo_tensor(matrix: sp.spmatrix) -> torch.Tensor:
    graph = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack((graph.row, graph.col)).astype(np.int64)),
        torch.from_numpy(np.asarray(graph.data, dtype=np.float32)),
        graph.shape,
    ).coalesce()


def cosine_knn(
    embeddings: np.ndarray,
    k: int = 10,
    include_self: bool = True,
    *,
    block_size: int = 128,
    selection: str = "torch_topk",
) -> torch.Tensor:


    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite, nonempty 2D array")
    n_items = len(array)
    maximum = n_items if include_self else n_items - 1
    if not isinstance(k, int) or k < 1 or k > maximum:
        raise ValueError(f"k must lie in [1, {maximum}]")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if selection == "torch_topk":
        z = F.normalize(torch.from_numpy(array), dim=1)
        neighbours = []
        for start in range(0, n_items, block_size):
            score = z[start : start + block_size] @ z.T
            if not include_self:
                local = torch.arange(len(score))
                score[local, local + start] = -torch.inf
            neighbours.append(torch.topk(score, k, dim=1).indices.numpy())
        columns = np.concatenate(neighbours).ravel()
    elif selection == "numpy_argpartition":


        columns = []
        for receiver in range(n_items):
            score = np.asarray(array @ array[receiver], dtype=np.float64)
            if not include_self:
                score[receiver] = -np.inf
            chosen = np.argpartition(score, -k)[-k:]
            columns.extend(sorted(int(i) for i in chosen))
        columns = np.asarray(columns, dtype=np.int64)
    else:
        raise ValueError("Unknown neighbour selection rule")
    rows = np.repeat(np.arange(n_items), k)
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack((rows, columns))),
        torch.full((n_items * k,), 1.0 / k, dtype=torch.float32),
        (n_items, n_items),
    ).coalesce()
