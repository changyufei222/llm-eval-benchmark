from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines.benchmark_schedule import build_sampling_groups  # type: ignore[attr-defined]


class BenchmarkScheduleTests(unittest.TestCase):
    def test_build_sampling_groups_splits_rounds_into_short_chains(self) -> None:
        groups = build_sampling_groups(total_rounds=100, group_size=5, max_parallel_groups=4)

        self.assertEqual(len(groups), 20)
        self.assertEqual(groups[0]["group_index"], 1)
        self.assertEqual(groups[0]["round_start"], 1)
        self.assertEqual(groups[0]["round_end"], 5)
        self.assertEqual(groups[0]["chain_index"], 1)
        self.assertEqual(groups[1]["chain_index"], 2)
        self.assertEqual(groups[4]["chain_index"], 1)
        self.assertEqual(groups[-1]["round_start"], 96)
        self.assertEqual(groups[-1]["round_end"], 100)
        self.assertEqual(groups[-1]["chain_index"], 4)

    def test_build_sampling_groups_rejects_non_positive_values(self) -> None:
        with self.assertRaises(ValueError):
            build_sampling_groups(total_rounds=0, group_size=5, max_parallel_groups=4)
        with self.assertRaises(ValueError):
            build_sampling_groups(total_rounds=100, group_size=0, max_parallel_groups=4)
        with self.assertRaises(ValueError):
            build_sampling_groups(total_rounds=100, group_size=5, max_parallel_groups=0)


if __name__ == "__main__":
    unittest.main()
