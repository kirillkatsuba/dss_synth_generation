import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.explore_iops_latency_clouds import load_annotations, prepare_payload


class GroundTruthCloudsTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.csv"
        self.frame = pd.DataFrame({
            "iops": [1000, 20000, 3000], "lat": [1e6, 2e6, 3e6],
            "id": ["a", "b", "a"], "raid": ["8+2", "4+1", "8+2"],
        })
        self.frame.to_csv(self.source, index=False)
        self.payload = prepare_payload(
            self.frame, self.source, {"iops": [0, 14], "lat": [0, 100]}, 1e3, 1e6
        )

    def annotations(self, groups, sha256=None):
        path = self.source.parent / "annotations.json"
        path.write_text(json.dumps({
            "source_sha256": sha256 or self.payload["sha256"],
            "rows": len(self.frame), "clouds": groups,
        }))
        return path

    def test_payload_preserves_all_rows_and_original_order(self):
        restored = []
        for iops, lat, code in self.payload["rows"]:
            restored.append({"iops": iops, "lat": lat, **self.payload["configurations"][code]})
        pd.testing.assert_frame_equal(pd.DataFrame(restored), self.frame)
        self.assertEqual(self.payload["visible_rows"], 2)
        self.assertEqual(len(restored), 3)

    def test_valid_annotation_uses_source_row_indices(self):
        labels, groups = load_annotations(self.annotations([
            {"id": 1, "source_rows": [0, 2]},
        ]), self.payload)
        self.assertEqual(labels.tolist(), [1, 0, 1])
        self.assertEqual(list(groups), [1])

    def test_rejects_invalid_indices_and_overlapping_clouds(self):
        for rows in ([0.5], [True], [-1], [3], [0, 0]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_annotations(self.annotations([{"id": 1, "source_rows": rows}]), self.payload)
        with self.assertRaises(ValueError):
            load_annotations(self.annotations([
                {"id": 1, "source_rows": [0]}, {"id": 2, "source_rows": [0]},
            ]), self.payload)

    def test_rejects_another_source_csv(self):
        with self.assertRaises(ValueError):
            load_annotations(self.annotations([], sha256="different"), self.payload)


if __name__ == "__main__":
    unittest.main()
