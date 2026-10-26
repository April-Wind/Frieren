from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import GraphArtifact, InteractionData
from ..graph import cosine_knn
from ..io import atomic_json, atomic_torch_save, file_sha256, run_lock
from .behavior import binary_interactions


@dataclass(frozen=True)
class LearnableEstimatorConfig:
    dim: int = 64
    hidden: int = 256
    mask: float = 0.5
    align: float = 0.1
    tau: float = 0.2
    batch: int = 256
    negative: int = 128
    lr: float = 0.001
    reg: float = 0.0001
    epochs: int = 200
    patience: int = 20
    seed: int = 999
    behavior_only: bool = False

    def __post_init__(self):
        for name in ("dim", "hidden", "batch", "negative", "epochs", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.mask <= 1 or self.align < 0 or self.reg < 0:
            raise ValueError("Invalid masking/alignment/regularization setting")
        for name in ("mask", "align", "tau", "lr", "reg"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.tau <= 0 or self.lr <= 0:
            raise ValueError("Temperature and learning rate must be positive")
        if self.behavior_only and (self.mask != 0 or self.align != 0):
            raise ValueError("Behavior-only ablations require mask=0 and align=0")


def prepare_multimodal_features(image: np.ndarray, text: np.ndarray) -> np.ndarray:
    
    image = np.asarray(image, dtype=np.float32)
    text = np.asarray(text, dtype=np.float32)
    if image.ndim != 2 or text.ndim != 2 or len(image) != len(text):
        raise ValueError("Image and text feature rows must align")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("Features contain nonfinite values")
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(np.concatenate((image, text), axis=1) / np.sqrt(2),
                                dtype=np.float32)


class SidePredictor(nn.Module):
    
    def __init__(self, n_users: int, n_items: int, features: np.ndarray | None,
                 config: LearnableEstimatorConfig):
        super().__init__()
        self.config = config
        self.behavior_only = config.behavior_only
        self.user = nn.Embedding(n_users, config.dim)
        self.item = nn.Embedding(n_items, config.dim)
        nn.init.normal_(self.user.weight, std=0.08)
        nn.init.normal_(self.item.weight, std=0.08)
        if not self.behavior_only:
            if features is None:
                raise ValueError("Content-aware estimation requires fixed features")
            self.register_buffer("feature", torch.from_numpy(features), persistent=False)
            self.encoder = nn.Sequential(
                nn.Linear(features.shape[1], config.hidden), nn.LeakyReLU(0.2),
                nn.Linear(config.hidden, config.dim),
            )

    def vectors(self):
        users = F.normalize(self.user.weight, dim=1, eps=1e-8)
        behavior = F.normalize(self.item.weight, dim=1, eps=1e-8)
        content = None if self.behavior_only else F.normalize(self.encoder(self.feature),
                                                             dim=1, eps=1e-8)
        return users, behavior, content

    def candidate_scores(self, users: torch.Tensor, candidates: torch.Tensor,
                         mask: torch.Tensor | None = None) -> torch.Tensor:
        user = F.normalize(self.user(users), dim=-1, eps=1e-8)
        behavior = F.normalize(self.item(candidates), dim=-1, eps=1e-8)
        if self.behavior_only:
            item = behavior
        else:
            content = F.normalize(self.encoder(self.feature[candidates]), dim=-1, eps=1e-8)
            fused = F.normalize(behavior + content, dim=-1, eps=1e-8)
            item = torch.where(mask[:, None, None], content, fused) if mask is not None else fused
        return (user[:, None, :] * item).sum(dim=-1) / self.config.tau


def direct_item_vectors(behavior: torch.Tensor, content: torch.Tensor | None,
                        train: sp.spmatrix) -> torch.Tensor:
    if content is None:
        return behavior
    warm = np.asarray(train.sum(axis=0)).ravel() > 0
    warm = torch.from_numpy(warm).to(behavior.device)
    return torch.where(warm[:, None], F.normalize(behavior + content, dim=1, eps=1e-8), content)


def batch_candidates(train: sp.csr_matrix, users: np.ndarray,
                     rng: np.random.Generator, negative: int = 128) -> np.ndarray:


    positives = np.fromiter(
        (rng.choice(train.indices[train.indptr[u] : train.indptr[u + 1]]) for u in users),
        dtype=np.int64, count=len(users),
    )
    candidates = np.empty((len(users), negative + 1), dtype=np.int64)
    candidates[:, 0] = positives
    excluded = np.zeros(train.shape[1], dtype=bool)
    for row, user in enumerate(users):
        seen = train.indices[train.indptr[user] : train.indptr[user + 1]]
        if len(seen) == train.shape[1]:
            raise ValueError("Cannot draw negatives for a user who saw every item")
        excluded[seen] = True
        draws = []
        while len(draws) < negative:
            values = rng.integers(0, train.shape[1], size=160, dtype=np.int64)
            draws.extend(values[~excluded[values]].tolist())
        candidates[row, 1:] = draws[:negative]
        excluded[seen] = False
    return candidates


def estimator_losses(model: SidePredictor, users: torch.Tensor,
                     candidates: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    config = model.config
    scores = model.candidate_scores(users, candidates, mask)
    target = torch.zeros(len(users), dtype=torch.long, device=users.device)
    recommendation = F.cross_entropy(scores, target)
    alignment = scores.new_zeros(())
    if config.align > 0:
        unique = torch.unique(candidates[:, 0])
        if len(unique) > 1:
            content = F.normalize(model.encoder(model.feature[unique]), dim=-1, eps=1e-8)
            behavior = F.normalize(model.item(unique), dim=-1, eps=1e-8)
            logits = content @ behavior.T / config.tau
            target = torch.arange(len(unique), device=users.device)
            alignment = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    regularization = model.user(users).square().mean() + model.item(candidates).square().mean()
    if not model.behavior_only:
        regularization += sum(p.square().mean() for p in model.encoder.parameters())
    regularization *= config.reg
    return {
        "recommendation": recommendation, "alignment": alignment,
        "regularization": regularization,
        "total": recommendation + config.align * alignment + regularization,
    }


def _feature_array(features, n_items, behavior_only):
    if behavior_only:
        return None
    array = np.ascontiguousarray(features, dtype=np.float32)
    if array.ndim != 2 or len(array) != n_items or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("Features must be finite and match the catalogue")
    return array


def _array_hash(array):
    if array is None:
        return None
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _training_hash(train):
    digest = hashlib.sha256(str(train.shape).encode())
    for value in (train.indptr, train.indices, train.data):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _targets_hash(targets):
    digest = hashlib.sha256()
    for user in sorted(targets):
        digest.update(np.asarray([user], dtype=np.int64).tobytes())
        items = np.sort(np.asarray(targets[user], dtype=np.int64))
        digest.update(np.asarray([len(items)], dtype=np.int64).tobytes())
        digest.update(items.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _evaluate(model, train, targets, mode, batch_size=64):
    model.eval()
    users, behavior, content = model.vectors()
    items = content if mode == "side" else direct_item_vectors(behavior, content, train)
    if items is None:
        raise ValueError("Side evaluation is unavailable for a behavior-only model")
    catalogue = items.T.contiguous()
    selected_users = np.asarray(sorted(targets), dtype=np.int64)
    if not len(selected_users):
        raise ValueError("Evaluation requires held-out users")
    metrics = {f"{metric}@{k}": 0.0 for k in (10, 20) for metric in ("recall", "ndcg")}
    predictions = np.empty((len(selected_users), min(20, train.shape[1])), dtype=np.int64)
    for start in range(0, len(selected_users), batch_size):
        batch_users = selected_users[start : start + batch_size]
        scores = np.asarray((users[torch.from_numpy(batch_users).to(users.device)] @ catalogue /
                             model.config.tau).cpu().numpy(), dtype=np.float32).copy()
        seen = train[batch_users].tocoo()
        scores[seen.row, seen.col] = -np.inf
        order = torch.topk(torch.from_numpy(scores), predictions.shape[1], dim=1).indices.numpy()
        predictions[start : start + len(batch_users)] = order
        for row, user in enumerate(batch_users):
            relevant = set(int(i) for i in np.asarray(targets[int(user)]).ravel())
            if not relevant:
                raise ValueError("Every evaluated user needs a held-out target")
            hits = np.asarray([int(item) in relevant for item in order[row]], dtype=float)
            for k in (10, 20):
                cutoff = min(k, len(hits))
                metrics[f"recall@{k}"] += hits[:cutoff].sum() / len(relevant)
                discounts = 1.0 / np.log2(np.arange(cutoff) + 2)
                ideal = (1.0 / np.log2(np.arange(min(cutoff, len(relevant))) + 2)).sum()
                metrics[f"ndcg@{k}"] += float((hits[:cutoff] * discounts).sum() / ideal)
    return {key: value / len(selected_users) for key, value in metrics.items()}, predictions


def _rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def train_learnable(data: InteractionData, features: np.ndarray | None,
                    config: LearnableEstimatorConfig, output_dir: str | Path,
                    device: str = "cpu", resume: bool = False) -> dict:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(output_dir):
        return _train(data, features, config, output_dir, device, resume)


def _train(data, features, config, output_dir, device, resume):
    train = binary_interactions(data.train)
    if (np.diff(train.indptr) == 0).any():
        raise ValueError("Canonical uniform-user sampling requires a history for every user")
    if not data.validation or not data.test:
        raise ValueError("Training needs independent validation and test splits")
    features = _feature_array(features, data.n_items, config.behavior_only)
    identity = {
        "item_order_sha256": data.order_sha256, "feature_sha256": _array_hash(features),
        "train_sha256": _training_hash(train), "validation_sha256": _targets_hash(data.validation),
        "test_sha256": _targets_hash(data.test), "config": asdict(config),
    }
    result_path = output_dir / "result.json"
    if result_path.exists():
        import json
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError("Output already contains a completed run; use resume or a new directory")
        if result["identity"] != identity:
            raise ValueError("Completed run identity differs from the requested run")
        return result
    last_path = output_dir / "last.pt"
    if last_path.exists() and not resume:
        raise FileExistsError("An existing checkpoint requires explicit resume")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    target_device = torch.device(device)
    model = SidePredictor(data.n_users, data.n_items, features, config).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    modes = ("direct",) if config.behavior_only else ("side", "direct")
    best = {mode: -1.0 for mode in modes}
    best_epochs = {mode: None for mode in modes}
    stale = {mode: 0 for mode in modes}
    start_epoch, elapsed = 0, 0.0
    history = []
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=target_device, weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("Resume data/configuration differs from the saved run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best, best_epochs, stale = checkpoint["best"], checkpoint["best_epochs"], checkpoint["stale"]
        history, elapsed = checkpoint["history"], checkpoint["seconds"]
        start_epoch = checkpoint["epoch"] + 1
        _restore_rng(checkpoint["rng"])
    start = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        if epoch and all(value >= config.patience for value in stale.values()):
            break
        rng = np.random.default_rng(config.seed + 1000003 * epoch)
        mask_rng = np.random.default_rng(config.seed + 29001 + epoch)
        users = rng.integers(0, data.n_users, size=train.nnz, dtype=np.int64)
        losses = []
        model.train()
        for lo in range(0, len(users), config.batch):
            ids = users[lo : lo + config.batch]
            candidates = batch_candidates(train, ids, rng, config.negative)
            user_tensor = torch.from_numpy(ids).to(target_device)
            candidate_tensor = torch.from_numpy(candidates).to(target_device)
            mask = (torch.from_numpy(mask_rng.random(len(ids)) < config.mask).to(target_device)
                    if config.mask > 0 else None)
            values = estimator_losses(model, user_tensor, candidate_tensor, mask)
            optimizer.zero_grad(set_to_none=True)
            values["total"].backward()
            optimizer.step()
            losses.append(float(values["total"].detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        for mode in modes:
            valid, _ = _evaluate(model, train, data.validation, mode)
            row.update({f"{mode}/{key}": value for key, value in valid.items()})
            if valid["recall@20"] > best[mode] + 1e-12:
                best[mode], best_epochs[mode], stale[mode] = valid["recall@20"], epoch, 0
                atomic_torch_save(output_dir / f"{mode}_best.pt", {
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": epoch, "mode": mode, "identity": identity,
                    "config": asdict(config), "checkpoint_format": "frieren-u2-v1",
                })
            else:
                stale[mode] += 1
        history.append(row)
        atomic_torch_save(last_path, {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "epoch": epoch,
            "best": best, "best_epochs": best_epochs, "stale": stale, "history": history,
            "identity": identity, "config": asdict(config), "rng": _rng_state(),
            "seconds": elapsed + time.monotonic() - start,
        })
        atomic_json(output_dir / "progress.json", {"epoch": epoch, "best": best, "stale": stale})
    selected_path = output_dir / "direct_best.pt"
    checkpoint = torch.load(selected_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation, validation_predictions = _evaluate(model, train, data.validation, "direct")
    if abs(validation["recall@20"] - best["direct"]) > 1e-10:
        raise RuntimeError("Direct-selected checkpoint validation does not reproduce")
    test, test_predictions = _evaluate(model, train, data.test, "direct")
    np.savez_compressed(output_dir / "direct_validation_top20.npz",
                        users=np.asarray(sorted(data.validation)), top20=validation_predictions)
    np.savez_compressed(output_dir / "direct_test_top20.npz",
                        users=np.asarray(sorted(data.test)), top20=test_predictions)
    result = {
        "status": "COMPLETE", "identity": identity, "effective_config": asdict(config),
        "history": history, "best_epoch": best_epochs["direct"],
        "valid": validation, "test_at_best_valid": test,
        "checkpoint": selected_path.name, "checkpoint_sha256": file_sha256(selected_path),
        "selection_uses_test": False, "test_evaluations": 1,
        "seconds": elapsed + time.monotonic() - start,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_trainable": False, "graph_representation": "warm_normalize_v_plus_z_cold_z",
    }
    atomic_json(result_path, result)
    return result


def export_learned_graph(data: InteractionData, features: np.ndarray | None,
                         checkpoint_path: str | Path, k: int = 10,
                         device: str = "cpu") -> GraphArtifact:
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_format") != "frieren-u2-v1" or checkpoint.get("mode") != "direct":
        raise ValueError("Export requires a Frieren direct-selected checkpoint")
    config = LearnableEstimatorConfig(**checkpoint["config"])
    features = _feature_array(features, data.n_items, config.behavior_only)
    train = binary_interactions(data.train)
    identity = checkpoint["identity"]
    if identity["item_order_sha256"] != data.order_sha256 or identity["feature_sha256"] != _array_hash(features):
        raise ValueError("Checkpoint feature or item order mismatch")
    if identity["train_sha256"] != _training_hash(train):
        raise ValueError("Checkpoint training interactions mismatch")
    model = SidePredictor(data.n_users, data.n_items, features, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        _, behavior, content = model.vectors()
        item_vectors = direct_item_vectors(behavior, content, train).cpu().numpy()
    operator = cosine_knn(item_vectors, k, False, selection="numpy_argpartition")
    return GraphArtifact(operator=operator, item_ids=data.item_ids, metadata={
        "estimator": "Frieren-L", "recipe": "u2_fused_direct_numpy_partition_v1",
        "orientation": "receiver_sender", "k": k, "include_self": False,
        "edge_weight": "1/k", "selection": "numpy_argpartition",
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_selection": "validation direct Recall@20",
        "representation": "behavior_v" if config.behavior_only else "warm_normalize_v_plus_z_cold_z",
        "feature_sha256": identity["feature_sha256"], "symmetrized": False,
        "config": asdict(config),
    })
