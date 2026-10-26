from pathlib import Path
import csv
import json
import numpy as np
from scipy import sparse
from .contracts import InteractionData
from .io import file_sha256


def load_item_ids(path, n_items=None):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"item_id", "itemID"}.issubset(reader.fieldnames or []):
            raise ValueError("Mapping needs TSV columns item_id (external ID) and itemID (zero-based index)")
        mapping = {}
        for row in reader:
            index = int(row["itemID"])
            if index in mapping:
                raise ValueError("Duplicate item mapping index")
            mapping[index] = row["item_id"]
    count = len(mapping) if n_items is None else int(n_items)
    if set(mapping) != set(range(count)):
        raise ValueError("Item mapping indices must be contiguous and cover all items")
    ids = tuple(mapping[i] for i in range(count))
    if len(set(ids)) != count:
        raise ValueError("Duplicate external item identifier")
    return ids


def dataset_spec(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "frieren.dataset.v1":
        raise ValueError("Expected frieren.dataset.v1 manifest")
    for key in ("interaction_file", "item_mapping_file", "image_features", "text_features", "knowledge_embeddings", "knowledge_mapping"):
        if spec.get(key):
            spec[key] = str((path.parent / spec[key]).resolve())
    return spec


def load_dataset(path):
    spec = dataset_spec(path)
    ids = load_item_ids(spec["item_mapping_file"], spec.get("n_items"))
    pairs, validation, test = [], {}, {}
    users_seen = set()
    counts = {0: 0, 1: 0, 2: 0}
    with Path(spec["interaction_file"]).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"userID", "itemID", "x_label"}.issubset(reader.fieldnames or []):
            raise ValueError("Interaction TSV needs userID, itemID and x_label")
        seen = set()
        for row in reader:
            u, i, split = int(row["userID"]), int(row["itemID"]), int(row["x_label"])
            if u < 0 or not 0 <= i < len(ids) or split not in counts:
                raise ValueError("Invalid interaction ID or split (0=train, 1=validation, 2=test)")
            if (u, i) in seen:
                raise ValueError("Repeated user-item pair, including overlap across splits")
            seen.add((u, i)); users_seen.add(u); counts[split] += 1
            if split == 0:
                pairs.append((u, i))
            else:
                target = validation if split == 1 else test
                target.setdefault(u, []).append(i)
    inferred = max(users_seen, default=-1) + 1
    n_users = int(spec.get("n_users", inferred))
    if n_users < inferred or n_users <= 0 or not pairs:
        raise ValueError("Declared user count is invalid or training split is empty")
    pair_array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    x = sparse.csr_matrix((np.ones(len(pairs), np.float32), pair_array.T), shape=(n_users, len(ids)))
    return InteractionData(x, pair_array,
        {u: np.asarray(v, np.int64) for u, v in validation.items()},
        {u: np.asarray(v, np.int64) for u, v in test.items()}, ids,
        {"name": spec.get("name", "dataset"), "split_counts": counts,
         "interaction_sha256": file_sha256(spec["interaction_file"]),
         "mapping_sha256": file_sha256(spec["item_mapping_file"]),
         "manifest_sha256": file_sha256(path)})


def load_content_features(path, n_items):
    
    spec = dataset_spec(path)
    arrays = []
    for key in ("image_features", "text_features"):
        if spec.get(key):
            a = np.asarray(np.load(spec[key], allow_pickle=False), dtype=np.float32)
            if a.ndim != 2 or len(a) != n_items or not np.isfinite(a).all():
                raise ValueError("Feature rows must match the item mapping and be finite")
            arrays.append(a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12))
    if not arrays:
        raise ValueError("Provide at least one content feature file")
    return np.ascontiguousarray(np.concatenate(arrays, axis=1) / np.sqrt(len(arrays)), dtype=np.float32)
