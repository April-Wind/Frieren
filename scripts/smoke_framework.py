from pathlib import Path
import argparse
import contextlib
import importlib.util
import io
import json
import time

import torch

from frieren.cli import main as cli
from frieren.io import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs" / "smoke"))
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    if folder.exists():
        raise FileExistsError("Choose a fresh smoke output directory")
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    dataset_folder = folder / "data"
    spec = importlib.util.spec_from_file_location("frieren_make_demo", ROOT / "scripts" / "make_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    
    import sys
    original = sys.argv
    try:
        sys.argv = ["make_demo.py", "--output", str(dataset_folder)]
        demo.main()
    finally:
        sys.argv = original
    dataset = str(dataset_folder / "dataset.json")
    graphs = folder / "graphs"
    report = {"synthetic": True, "device": "cpu", "steps": []}

    def run(name, argv):
        started = time.perf_counter()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            cli(argv)
        report["steps"].append({"name": name, "status": "PASS", "seconds": time.perf_counter() - started})
        print(name + ": PASS", flush=True)

    run("multimodal_reference", ["build-reference", "--dataset", dataset, "--output", str(graphs / "F.pt")])
    run("knowledge", ["build-k", "--dataset", dataset, "--embeddings", str(dataset_folder / "knowledge.npy"), "--output", str(graphs / "K.pt")])
    run("behavior", ["build-b", "--dataset", dataset, "--fallback", str(graphs / "F.pt"), "--coefficients", str(folder / "B.npy"), "--output", str(graphs / "B.pt")])
    run("learnable_training", ["train-l", "--dataset", dataset, "--config", str(ROOT / "configs" / "demo_L.json"), "--output-dir", str(folder / "L")])
    run("learnable_export", ["export-l", "--dataset", dataset, "--checkpoint", str(folder / "L" / "direct_best.pt"), "--output", str(graphs / "L.pt")])
    run("bgnw", ["refine-bgnw", "--dataset", dataset, "--graph", str(graphs / "L.pt"), "--output", str(graphs / "L_BGNW.pt")])
    run("freedom_training", ["train-freedom", "--dataset", dataset, "--graph", str(graphs / "L_BGNW.pt"), "--config", str(ROOT / "configs" / "demo_FREEDOM.json"), "--output-dir", str(folder / "FREEDOM")])
    for backbone in ("MENTOR", "SMORE", "MSCA"):
        run("adapter_" + backbone, ["adapt", "--graph", str(graphs / "L_BGNW.pt"), "--backbone", backbone, "--output", str(graphs / (backbone + ".pt"))])
    atomic_json(graphs / "replay_inputs.json", {name: name + ".pt" for name in ("F", "K", "B", "L", "L_BGNW")})
    run("fixed_parameter_replay", ["replay", "--dataset", dataset, "--graphs", str(graphs / "replay_inputs.json"), "--checkpoint", str(folder / "FREEDOM" / "best.pt"), "--output", str(folder / "replay.json")])
    replay = json.loads((folder / "replay.json").read_text(encoding="utf-8"))
    trained = json.loads((folder / "FREEDOM" / "result.json").read_text(encoding="utf-8"))
    if replay["parameter_updates"] != 0 or replay["graphs"]["L_BGNW"]["test"] != trained["test"]:
        raise AssertionError("Frozen replay must exactly reproduce the trained model on its native graph")
    report["native_graph_replay_matches_training"] = True
    report["parameter_updates_during_replay"] = 0
    report["status"] = "PASS"
    atomic_json(folder / "SMOKE_REPORT.json", report)
    print(folder / "SMOKE_REPORT.json")


if __name__ == "__main__":
    main()
