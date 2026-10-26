from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .adapters import operator_signature
from .backbones import FreedomConsumer
from .contracts import GraphArtifact, InteractionData
from .evaluation import evaluate_freedom
from .io import atomic_json, atomic_torch_save, file_sha256, run_lock


@dataclass(frozen=True)
class TrainConfig:


    embedding_dim: int = 64
    n_ui_layers: int = 2
    n_item_layers: int = 1
    dropout: float = 0.3
    seed: int = 999
    learning_rate: float = 0.001
    epochs: int = 500
    batch_size: int = 2048
    eval_batch_size: int = 4096
    topk: tuple[int, ...] = (5, 10, 20, 50)
    cpu_threads: int = 1
    patience: int | None = None

    def validate(self):
        if self.patience is not None and (not isinstance(self.patience, int) or isinstance(self.patience, bool) or self.patience < 1):
            raise ValueError("Early-stopping patience must be a positive integer or null")
        if min(self.embedding_dim, self.epochs, self.batch_size, self.eval_batch_size, self.cpu_threads) < 1:
            raise ValueError("Dimensions, budget, batch sizes and CPU threads must be positive")
        if self.n_ui_layers < 0 or self.n_item_layers < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid propagation count or UI dropout")
        if self.learning_rate <= 0 or not np.isfinite(self.learning_rate):
            raise ValueError("Learning rate must be finite and positive")
        if 20 not in self.topk or any(int(k) != k or k < 1 for k in self.topk):
            raise ValueError("Ranking cutoffs must be positive integers including Recall@20")
        return self


def _data_signature(data: InteractionData) -> str:
    digest = hashlib.sha256()
    digest.update(str(data.train.shape).encode("ascii"))
    digest.update(data.train.indptr.tobytes())
    digest.update(data.train.indices.tobytes())
    digest.update(data.train_pairs.tobytes())
    digest.update(data.order_sha256.encode("ascii"))
    for name in ("validation", "test"):
        serialized = {str(int(u)): sorted(set(map(int, items))) for u, items in sorted(getattr(data, name).items())}
        digest.update(json.dumps(serialized, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _model_kwargs(config: TrainConfig) -> dict:
    return {name: getattr(config, name) for name in ("embedding_dim", "n_ui_layers", "n_item_layers", "dropout", "seed")}


def _portable_config(config: TrainConfig) -> dict:
    values = asdict(config)
    values["topk"] = list(config.topk)
    return values


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _rng_state(model):
    numpy = np.random.get_state()


    return {
        "python": random.getstate(),
        "numpy": (numpy[0], torch.from_numpy(numpy[1].astype(np.int64)), int(numpy[2]), int(numpy[3]), float(numpy[4])),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else None,
        "dropout": model.dropout_generator.get_state(),
    }


def _restore_rng(model, rng):
    random.setstate(rng["python"])
    name, keys, position, has_gauss, cached_gauss = rng["numpy"]
    np.random.set_state((name, keys.cpu().numpy().astype(np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(rng["torch"].cpu())
    if model.device.type == "cuda" and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    model.dropout_generator.set_state(rng["dropout"].cpu())


def _environment(device):
    return {"torch": str(torch.__version__), "numpy": str(np.__version__), "device": str(device)}


def train_freedom(
    data: InteractionData,
    artifact: GraphArtifact,
    config: TrainConfig,
    output_dir,
    device: str | torch.device = "cpu",
    resume: bool = False,
) -> dict:


    config.validate()
    artifact.validate()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not data.validation or not any(len(values) for values in data.validation.values()):
        raise ValueError("Validation targets are required for checkpoint selection")
    if not data.train.nnz or not len(data.train_pairs):
        raise ValueError("At least one training interaction is required")
    folder = Path(output_dir)
    with run_lock(folder):
        return _train_locked(data, artifact, config, folder, device, resume)


def _train_locked(data, artifact, config, folder, device, resume):
    last_path, best_path = folder / "last.pt", folder / "best.pt"
    if not resume and any(path.exists() for path in (last_path, best_path, folder / "result.json")):
        raise FileExistsError("Output contains an existing run; explicitly resume or choose a new directory")
    if resume and not last_path.is_file():
        raise FileNotFoundError("Resume requires last.pt")
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)


    item_pool = sorted(set(map(int, data.train_pairs[:, 1])))
    random.shuffle(item_pool)
    histories = [set(data.train.indices[data.train.indptr[u] : data.train.indptr[u + 1]].tolist()) for u in range(data.n_users)]
    active_set = set(item_pool)
    for user in np.unique(data.train_pairs[:, 0]):
        if not active_set.difference(histories[int(user)]):
            raise ValueError("A training user has consumed every active item; canonical negative sampling cannot proceed")
    model = FreedomConsumer(data, artifact, **_model_kwargs(config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    signature = _data_signature(data)
    graph_signature = operator_signature(artifact.operator)
    environment = _environment(device)
    common = {
        "schema": "frieren.freedom.v1",
        "configuration": _portable_config(config),
        "data_signature": signature,
        "item_ids": list(data.item_ids),
        "item_order_sha256": data.order_sha256,
        "graph_signature": graph_signature,
        "graph_metadata": artifact.metadata,
        "environment": environment,
    }
    pairs = data.train_pairs.copy()
    history, start_epoch, best_score, best_epoch, best_valid = [], 0, -1.0, None, {}
    elapsed_before = 0.0
    stale = 0
    if resume:
        previous = torch.load(last_path, map_location="cpu", weights_only=True)
        if previous.get("schema") != common["schema"]:
            raise ValueError("Unsupported resume checkpoint")
        for key in ("data_signature", "item_order_sha256", "graph_signature", "environment"):
            if previous[key] != common[key]:
                raise ValueError(f"Resume input/environment changed: {key}")
        old_config = dict(previous["configuration"])
        old_config.setdefault("patience", None)
        new_config = dict(common["configuration"])
        old_budget, new_budget = old_config.pop("epochs"), new_config.pop("epochs")
        if old_config != new_config or new_budget < old_budget:
            raise ValueError("Resume permits only increasing the epoch budget")
        model.load_state_dict(previous["model_state"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state"])
        if "scheduler_state" in previous:
            scheduler.load_state_dict(previous["scheduler_state"])
        
        pairs = previous["training_pair_order"].numpy().copy()
        item_pool = previous["training_item_pool"]
        history = previous["history"]
        start_epoch = int(previous["epoch"]) + 1
        best_score, best_epoch, best_valid = previous["best_score"], previous["best_epoch"], previous["best_validation"]
        stale = int(previous.get("stale", previous["epoch"] - best_epoch))
        elapsed_before = float(previous["training_seconds"])
        if "best_checkpoint" not in previous:
            raise ValueError("Resume checkpoint is missing its selected model")
        atomic_torch_save(best_path, previous["best_checkpoint"])
        _restore_rng(model, previous["rng"])
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        if config.patience is not None and stale >= config.patience:
            break
        model.train()
        model.pre_epoch_processing()
        pairs = pairs[np.random.choice(len(pairs), size=len(pairs), replace=False)]
        losses = []
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start : start + config.batch_size]
            negatives = []
            for user in batch[:, 0]:
                negative = random.sample(item_pool, 1)[0]
                while negative in histories[int(user)]:
                    negative = random.sample(item_pool, 1)[0]
                negatives.append(negative)
            interaction = torch.as_tensor(np.vstack((batch.T, np.asarray(negatives))), dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = model.calculate_loss(interaction)
            if not torch.isfinite(loss):
                raise FloatingPointError("FREEDOM training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation = evaluate_freedom(model, data, "validation", topk=config.topk, batch_size=config.eval_batch_size)["metrics"]
        score = float(validation["recall@20"])
        if score > best_score:
            best_score, best_epoch, best_valid = score, epoch, validation
            stale = 0
            best = {**common, "epoch": epoch, "model_state": _cpu_state(model), "validation": validation}
            atomic_torch_save(best_path, best)
        else:
            stale += 1
        history.append({"epoch": epoch, "loss_sum": sum(losses), "loss_mean": float(np.mean(losses)), "validation": validation})
        atomic_json(folder / "history.json", history)
        latest = {
            **common,
            "epoch": epoch,
            "model_state": _cpu_state(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "stale": stale,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_validation": best_valid,
            "best_checkpoint": torch.load(best_path, map_location="cpu", weights_only=True),
            "training_pair_order": torch.from_numpy(pairs.copy()),
            "training_item_pool": item_pool,
            "history": history,
            "rng": _rng_state(model),
            "training_seconds": elapsed_before + time.perf_counter() - started,
        }
        atomic_torch_save(last_path, latest)
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state"], strict=True)
    test = evaluate_freedom(model, data, "test", topk=config.topk, batch_size=config.eval_batch_size)
    result = {
        "status": "COMPLETE",
        "model": "FREEDOM M0 single fixed graph",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "stale": stale,
        "stop_reason": "early_stopping" if config.patience is not None and stale >= config.patience else "epoch_budget",
        "validation": best_valid,
        "test": test["metrics"],
        "selection": "maximum validation Recall@20; strict > tie handling",
        "protocol": {"max_epochs": config.epochs, "patience": config.patience, "metric": "validation recall@20"},
        "configuration": common["configuration"],
        "data_signature": signature,
        "graph_signature": graph_signature,
        "item_order_sha256": data.order_sha256,
        "environment": environment,
        "best_checkpoint_sha256": file_sha256(best_path),
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "test_evaluated_users": test["evaluated_users"],
        "negative_pool": "training-active items excluding each user's training history",
    }
    atomic_json(folder / "result.json", result)
    if test["evaluated_users"]:
        atomic_json(folder / "test_predictions.json", {"users": test["users"], "items": test["predictions"]})
    return result


def replay_freedom(
    data: InteractionData,
    artifacts: dict[str, GraphArtifact],
    checkpoint_path,
    device: str | torch.device = "cpu",
) -> dict:
    
    if not artifacts:
        raise ValueError("At least one graph is required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "frieren.freedom.v1":
        raise ValueError("Replay expects a release-framework FREEDOM checkpoint")
    if checkpoint["data_signature"] != _data_signature(data) or checkpoint["item_ids"] != list(data.item_ids):
        raise ValueError("Replay data or item order differs from the trained model")
    config = TrainConfig(**{**checkpoint["configuration"], "topk": tuple(checkpoint["configuration"]["topk"])}).validate()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(config.cpu_threads)
    first = next(iter(artifacts.values()))
    model = FreedomConsumer(data, first, **_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    result = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "parameter_updates": 0,
        "graphs": {},
    }
    for name, artifact in artifacts.items():
        model.replace_graph(artifact)
        result["graphs"][name] = {
            "graph_signature": operator_signature(artifact.operator),
            "validation": evaluate_freedom(model, data, "validation", topk=config.topk, batch_size=config.eval_batch_size)["metrics"],
            "test": evaluate_freedom(model, data, "test", topk=config.topk, batch_size=config.eval_batch_size)["metrics"],
        }
    return result
