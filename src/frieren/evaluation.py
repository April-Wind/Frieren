from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import scipy.sparse as sp
import torch


@torch.no_grad()
def evaluate_scores(
    score_batch: Callable[[torch.Tensor], torch.Tensor],
    train: sp.csr_matrix,
    targets: Mapping[int, np.ndarray],
    *,
    topk: tuple[int, ...] = (10, 20),
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> dict:


    ks = tuple(sorted(set(int(k) for k in topk)))
    if not ks or min(ks) < 1 or batch_size < 1:
        raise ValueError("Positive top-K values and batch size are required")
    matrix = train.tocsr()
    users = np.asarray(sorted(int(u) for u, items in targets.items() if len(items)), dtype=np.int64)
    if not users.size:
        return {"metrics": {}, "users": [], "predictions": [], "evaluated_users": 0}
    if users.min() < 0 or users.max() >= matrix.shape[0]:
        raise ValueError("Held-out user outside interaction shape")
    unique_targets = {}
    for user in users:
        values = np.unique(np.asarray(targets[int(user)], dtype=np.int64))
        if values.min() < 0 or values.max() >= matrix.shape[1]:
            raise ValueError("Held-out item outside interaction shape")
        history = matrix.indices[matrix.indptr[user] : matrix.indptr[user + 1]]
        if np.intersect1d(values, history).size:
            raise ValueError("Held-out targets overlap training history")
        unique_targets[int(user)] = set(values.tolist())
    totals = {f"{metric}@{k}": 0.0 for k in ks for metric in ("recall", "ndcg", "precision", "map")}
    max_k = min(max(ks), matrix.shape[1])
    predictions = []
    for start in range(0, len(users), batch_size):
        selected = users[start : start + batch_size]
        user_tensor = torch.as_tensor(selected, dtype=torch.long, device=device)
        scores = score_batch(user_tensor)
        if tuple(scores.shape) != (len(selected), matrix.shape[1]):
            raise ValueError("Score provider returned the wrong user-by-item shape")
        if not torch.isfinite(scores).all():
            raise ValueError("Score provider returned nonfinite unmasked values")
        scores = scores.clone()
        seen = matrix[selected].tocoo()
        scores[torch.as_tensor(seen.row, device=scores.device), torch.as_tensor(seen.col, device=scores.device)] = -torch.inf
        values, ranking = torch.topk(scores, max_k, dim=1)
        ranking[~torch.isfinite(values)] = -1
        rows = ranking.cpu().numpy()
        predictions.extend(rows.tolist())
        for user, ranked in zip(selected, rows):
            gold = unique_targets[int(user)]
            for k in ks:
                hits = np.fromiter((int(item) in gold for item in ranked[:k]), dtype=np.float64)
                total_hits = hits.sum()
                discount = 1.0 / np.log2(np.arange(len(hits)) + 2)
                ideal = (1.0 / np.log2(np.arange(min(len(gold), k)) + 2)).sum()
                totals[f"recall@{k}"] += float(total_hits / len(gold))
                totals[f"ndcg@{k}"] += float(np.dot(hits, discount) / ideal)
                totals[f"precision@{k}"] += float(total_hits / k)
                ranks = np.arange(len(hits)) + 1
                totals[f"map@{k}"] += float(np.sum(np.cumsum(hits) / ranks * hits) / min(len(gold), k))
    return {
        "metrics": {name: value / len(users) for name, value in totals.items()},
        "users": users.tolist(),
        "predictions": predictions,
        "evaluated_users": int(len(users)),
        "mask": "training history only",
    }


@torch.no_grad()
def evaluate_freedom(model, data, split="validation", *, topk=(10, 20), batch_size=4096):
    if split not in {"validation", "test"}:
        raise ValueError("Expected validation or test")
    was_training = model.training
    model.eval()
    try:
        users, items = model(model.norm_adj)
        return evaluate_scores(
            lambda selected: users[selected] @ items.T,
            data.train,
            getattr(data, split),
            topk=topk,
            batch_size=batch_size,
            device=model.device,
        )
    finally:
        model.train(was_training)
