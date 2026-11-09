import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hardsynth.families import FAMILIES
from hardsynth.generate import generate


class FamilyContracts(unittest.TestCase):
    def _kwargs(self, name, **over):
        kw = dict(seed=3, difficulty=0.5, n_total=5_000, label_noise=0.3)
        if name != "many_classes":
            kw["n_classes"] = 3
        kw.update(over)
        return kw

    def test_shapes_types_and_bayes_range(self):
        for name, cls in FAMILIES.items():
            with self.subTest(family=name):
                fam = cls(**self._kwargs(name))
                s = fam.sample(2_000, 0)
                self.assertEqual(len(s.y), 2_000)
                self.assertEqual(s.bayes.shape, (2_000,))
                for col, arr in s.X.items():
                    self.assertEqual(arr.shape, (2_000,), col)
                    if col in fam.categorical:
                        self.assertEqual(arr.dtype, np.int32, col)
                    else:
                        self.assertEqual(arr.dtype, np.float32, col)
                self.assertEqual(s.y.dtype, np.int32)
                self.assertGreaterEqual(s.bayes.min(), 1.0 / fam.n_classes - 1e-6)
                self.assertLessEqual(s.bayes.max(), 1.0 + 1e-6)
                self.assertEqual(set(np.unique(s.y)) - set(range(fam.n_classes)), set())

    def test_deterministic_and_chunks_differ(self):
        for name, cls in FAMILIES.items():
            with self.subTest(family=name):
                a = cls(**self._kwargs(name)).sample(500, 1000)
                b = cls(**self._kwargs(name)).sample(500, 1000)
                np.testing.assert_array_equal(a.y, b.y)
                c = cls(**self._kwargs(name)).sample(500, 1500)
                self.assertFalse(np.array_equal(a.y, c.y))

    def test_regression_variant(self):
        for name, cls in FAMILIES.items():
            if name == "many_classes":
                continue
            with self.subTest(family=name):
                fam = cls(**self._kwargs(name, task="regression"))
                s = fam.sample(3_000, 0)
                self.assertEqual(s.y.dtype, np.float32)
                resid = s.y - s.bayes
                self.assertGreater(np.std(resid), 0.2)
                self.assertLess(abs(np.std(s.bayes) - 1.0), 0.35)

    def test_difficulty_moves_the_knobs(self):
        easy = FAMILIES["deep_interaction"](**self._kwargs("deep_interaction", difficulty=0.0))
        hard = FAMILIES["deep_interaction"](**self._kwargs("deep_interaction", difficulty=1.0))
        self.assertLess(easy.n_cells, hard.n_cells)
        self.assertGreater(FAMILIES["many_classes"](**self._kwargs("many_classes", difficulty=1.0)).n_classes, 100)

    def test_many_classes_bayes_calibrated(self):
        fam = FAMILIES["many_classes"](**self._kwargs("many_classes", difficulty=0.3))
        s = fam.sample(20_000, 0)
        self.assertAlmostEqual(float(s.bayes.mean()), 0.75, delta=0.03)

    def test_grouped_test_groups_disjoint(self):
        fam = FAMILIES["grouped_entities"](**self._kwargs("grouped_entities"))
        tr = fam.sample(5_000, 0).X["group_id"]
        te = fam.sample_test(5_000, 5_000).X["group_id"]
        self.assertEqual(set(tr) & set(te), set())

    def test_temporal_time_monotone_and_covers_range(self):
        fam = FAMILIES["temporal_drift"](**self._kwargs("temporal_drift", n_total=4_000))
        t1 = fam.sample(2_000, 0).X["time"]
        t2 = fam.sample(2_000, 2_000).X["time"]
        self.assertTrue(np.all(np.diff(np.concatenate([t1, t2])) >= 0))
        self.assertAlmostEqual(float(t2[-1]), 1.0, places=5)


class GenerateCLI(unittest.TestCase):
    def test_generate_writes_parquet_and_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            meta = generate("high_cardinality", Path(d), n_samples=3_000, test_size=1_000, chunk_size=1_200,
                            verbose=False, difficulty=0.4, n_classes=2, seed=1)
            t = pq.read_table(Path(d) / "data.parquet")
            self.assertEqual(t.num_rows, 3_000)
            split = t.column("split").to_numpy()
            self.assertEqual(int(split.sum()), 1_000)
            self.assertTrue(np.all(split[-1_000:] == 1))
            saved = json.loads((Path(d) / "metadata.json").read_text())
            self.assertEqual(saved["n_train"], 2_000)
            self.assertIn("bayes_accuracy_test", saved)
            self.assertEqual(saved["categorical"], meta["categorical"])


if __name__ == "__main__":
    unittest.main()
