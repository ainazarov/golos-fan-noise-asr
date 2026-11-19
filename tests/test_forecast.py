from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forecast import build_forecast


class ForecastTests(unittest.TestCase):
    def inputs(self):
        golos = {"splits": {"train": {"recordings": 17, "hours": 1.0}}}
        sova = {"splits": {"train_mixed": {"recordings": 33, "hours": 2.0}}}
        benchmark = {"status": "complete", "mel_bins": 128, "results": {
            name: {"status": "passed", "batch_size": 16, "precision": 32, "mean_seconds_per_step": rate}
            for name, rate in (("golos", 2.0), ("mixed", 3.0), ("mixed_noise", 4.0))
        }}
        configs = {phase: {
            "trainer": {"max_epochs": epochs, "precision": 32},
            "model": {"train_ds": {"batch_size": 16}, "preprocessor": {"features": 128}, "encoder": {"feat_in": 128}},
        } for phase, epochs in (("baseline", 10), ("mixed", 25), ("noise", 5))}
        return golos, sova, benchmark, configs

    def test_ceiling_steps_and_sequential_finishes(self):
        start = datetime.fromisoformat("2026-11-19T13:00:00+00:00")
        report = build_forecast(*self.inputs(), start, allowance=0.15)
        self.assertEqual(report["total_epochs"], 40)
        self.assertEqual(report["phase_order"], ["baseline", "mixed", "noise"])
        self.assertEqual(report["stages"]["baseline"]["steps_per_epoch"], 2)
        self.assertEqual(report["stages"]["mixed"]["steps_per_epoch"], 3)
        self.assertEqual(report["stages"]["noise"]["steps_per_epoch"], 3)
        expected_seconds = {"baseline": 10 * 2 * 2, "mixed": 25 * 3 * 3, "noise": 5 * 3 * 4}
        cumulative = 0
        for phase, seconds in expected_seconds.items():
            cumulative += seconds
            stage = report["stages"][phase]
            self.assertAlmostEqual(stage["training_only_hours"], seconds / 3600)
            self.assertEqual(datetime.fromisoformat(stage["training_only_finish_moscow"]), start + timedelta(seconds=cumulative))
            self.assertEqual(datetime.fromisoformat(stage["planning_finish_moscow"]), start + timedelta(seconds=cumulative * 1.15))
        self.assertAlmostEqual(report["total_training_only_hours"], 325 / 3600)
        self.assertAlmostEqual(report["total_planning_hours"], 325 * 1.15 / 3600)
        self.assertEqual(report["planning_finish_moscow"], report["stages"]["noise"]["planning_finish_moscow"])
        self.assertEqual(report["hypothetical_start_moscow"], "2026-11-19T16:00:00+03:00")
        self.assertFalse(report["actual_training_started"])

    def test_rejects_invalid_start_and_allowance(self):
        for start in ("2026-11-19T16:00:00", "not-a-date", datetime(2026, 11, 19)):
            with self.subTest(start=start), self.assertRaises(ValueError):
                build_forecast(*self.inputs(), start)
        for allowance in (-0.01, float("nan"), float("inf")):
            with self.subTest(allowance=allowance), self.assertRaises(ValueError):
                build_forecast(*self.inputs(), "2026-11-19T16:00:00+03:00", allowance)


if __name__ == "__main__":
    unittest.main()
