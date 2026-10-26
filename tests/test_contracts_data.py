from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from frieren.contracts import GraphArtifact
from frieren.data import load_dataset, load_content_features
from frieren.io import run_lock


class DataContractsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        (self.root / "items.tsv").write_text("item_id\titemID\na\t0\nb\t1\nc\t2\n")
        (self.root / "data.inter").write_text("userID\titemID\tx_label\n0\t0\t0\n0\t1\t1\n0\t2\t2\n")
        np.save(self.root / "v.npy", np.ones((3, 2), np.float32))
        (self.root / "dataset.json").write_text(json.dumps({"schema":"frieren.dataset.v1", "name":"toy", "interaction_file":"data.inter", "item_mapping_file":"items.tsv", "image_features":"v.npy"}))
        return self.root / "dataset.json"

    def test_fixed_splits_and_mapping(self):
        path = self.fixture(); data = load_dataset(path)
        self.assertEqual(data.item_ids, ("a", "b", "c"))
        self.assertEqual(data.train.nnz, 1)
        np.testing.assert_array_equal(data.warm_items, [True, False, False])
        np.testing.assert_allclose(np.linalg.norm(load_content_features(path, 3), axis=1), 1, atol=1e-6)

    def test_rejects_heldout_overlap(self):
        path = self.fixture()
        (self.root / "data.inter").write_text("userID\titemID\tx_label\n0\t0\t0\n0\t0\t1\n")
        with self.assertRaises(ValueError): load_dataset(path)

    def test_graph_roundtrip_and_tamper_detection(self):
        operator = torch.eye(3).to_sparse()
        graph = GraphArtifact(operator, ("a", "b", "c"), {"recipe":"test"})
        path = graph.save(self.root / "graph.pt")
        loaded = GraphArtifact.load(path)
        self.assertEqual(loaded.order_sha256, graph.order_sha256)
        self.assertTrue(torch.equal(operator.to_dense(), loaded.operator.to_dense()))
        with path.open("ab") as stream: stream.write(b"tamper")
        with self.assertRaises(ValueError): GraphArtifact.load(path)

    def test_duplicate_writer_is_rejected(self):
        with run_lock(self.root):
            with self.assertRaises(RuntimeError):
                with run_lock(self.root): pass
        with run_lock(self.root): pass

if __name__ == "__main__": unittest.main()
