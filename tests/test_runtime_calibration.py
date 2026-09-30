import sys
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "benchmarks"))

from calibrate_historical_runtime import estimate_pilot_runtime, select_calibration_records


def example(repository, index, parent=None):
    return {"repository": repository, "commit": f"commit-{index}", "parent": parent or f"parent-{index}",
            "timestamp": f"2025-01-{index:02d}T00:00:00Z"}


class RuntimeCalibrationTests(unittest.TestCase):
    def test_selects_deterministic_early_middle_late_unique_parents(self):
        rows = []
        for repository in ("Pinia", "Express", "Flask"):
            rows.extend(example(repository, i) for i in range(1, 8))
            rows.append(example(repository, 8, parent="parent-7"))
        first = select_calibration_records(rows)
        second = select_calibration_records(list(reversed(rows)))
        for repository in ("Pinia", "Express", "Flask"):
            selected = first[repository]
            self.assertEqual([x["parent"] for x in selected], ["parent-1", "parent-4", "parent-7"])
            self.assertEqual([x["parent"] for x in selected], [x["parent"] for x in second[repository]])
            self.assertEqual(len({x["parent"] for x in selected}), 3)

    def test_runtime_projection_uses_unique_states_and_calibration_range(self):
        calibration = []
        for repository, times in (("Pinia", (100, 40, 60)), ("Express", (80, 30, 50)), ("Flask", (90, 20, 40))):
            for order, seconds in enumerate(times, 1):
                calibration.append({"repository": repository, "calibration_order": order,
                    "indexing_seconds": seconds, "total_chunks": 100})
        estimate = estimate_pilot_runtime(calibration,
            {"CourseCompass": 2, "Pinia": 5, "Express": 4, "Flask": 3})
        self.assertEqual(estimate["Pinia"]["unique_parent_states"], 5)
        self.assertEqual(estimate["Pinia"]["optimistic_seconds"], 260)
        self.assertEqual(estimate["Pinia"]["observed_seconds"], 300)
        self.assertEqual(estimate["Pinia"]["conservative_seconds"], 340)
        self.assertEqual(estimate["combined"]["observed_seconds"], sum(
            estimate[name]["observed_seconds"] for name in ("CourseCompass", "Pinia", "Express", "Flask")))


if __name__ == "__main__":
    unittest.main()
