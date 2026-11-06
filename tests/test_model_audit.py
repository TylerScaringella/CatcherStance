from __future__ import annotations

import unittest

import numpy as np

from research.scripts.audit_model import build_group_splits


class ModelAuditTests(unittest.TestCase):
    def test_group_splits_have_no_source_group_leakage(self):
        features = np.zeros((12, 2), dtype=np.float32)
        labels = np.asarray(["Left", "Right", "Squat"] * 4)
        groups = np.asarray(["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])
        for train, test in build_group_splits(features, labels, groups, n_splits=5, test_size=0.25):
            self.assertTrue(set(groups[train]).isdisjoint(set(groups[test])))

    def test_group_split_is_deterministic(self):
        features = np.zeros((12, 2), dtype=np.float32)
        labels = np.asarray(["Left", "Right", "Squat"] * 4)
        groups = np.asarray(["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])
        first = build_group_splits(features, labels, groups, n_splits=3)
        second = build_group_splits(features, labels, groups, n_splits=3)
        self.assertEqual([(a.tolist(), b.tolist()) for a, b in first], [(a.tolist(), b.tolist()) for a, b in second])


if __name__ == "__main__":
    unittest.main()
