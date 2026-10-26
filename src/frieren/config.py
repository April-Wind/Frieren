from pathlib import Path
from dataclasses import fields
import json


def read_config(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a JSON object")
    return result


def make_dataclass_config(cls, path):
    raw = read_config(path) if path else {}
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError("Unknown configuration settings: " + ", ".join(sorted(unknown)))
    return cls(**raw)
