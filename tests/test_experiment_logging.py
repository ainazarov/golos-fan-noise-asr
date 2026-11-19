import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiment_logging import EpochHistory, ProgressHistory


class EpochHistoryTests(unittest.TestCase):
    def test_checkpoint_association_metrics_and_replayed_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            checkpoint = checkpoints / "epoch=00-wer=0.4000.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            history = EpochHistory(root)
            history.record(epoch_index=0, global_step=10, checkpoint=checkpoint,
                           metrics={"val_wer": 0.4, "val_wer_ctc": 0.5, "train_loss": 2.0})
            history.record(epoch_index=0, global_step=10, checkpoint=checkpoint,
                           metrics={"val_wer": 0.3, "val_wer_ctc": 0.45})
            rows = [json.loads(line) for line in history.jsonl_path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["epoch"], 1)
            self.assertEqual(rows[0]["global_step"], 10)
            self.assertEqual(rows[0]["checkpoint"], "checkpoints/epoch=00-wer=0.4000.ckpt")
            self.assertEqual(rows[0]["checkpoint_bytes"], 10)
            self.assertEqual(rows[0]["val_wer"], 0.3)
            self.assertNotIn("train_loss", rows[0])
            with history.csv_path.open(newline="") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 1)
            self.assertEqual(float(table[0]["val_wer_ctc"]), 0.45)

    def test_older_resume_discards_future_metrics_and_keeps_previous_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "example.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            history = EpochHistory(root)
            for epoch in range(3):
                history.record(epoch_index=epoch, global_step=(epoch + 1) * 10,
                               checkpoint=checkpoint, metrics={"val_wer": 1 / (epoch + 1)})
            history.resume_from_step(20)
            rows = [json.loads(line) for line in history.jsonl_path.read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [1, 2])
            history.resume_from_step(0)
            self.assertEqual(history.jsonl_path.read_text(), "")

    def test_no_record_for_missing_checkpoint_and_nonfinite_metric_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "example.ckpt"
            history = EpochHistory(root)
            with self.assertRaises(FileNotFoundError):
                history.record(epoch_index=0, global_step=1, checkpoint=checkpoint, metrics={})
            self.assertFalse(history.jsonl_path.exists())
            checkpoint.write_bytes(b"checkpoint")
            history.record(epoch_index=0, global_step=1, checkpoint=checkpoint,
                           metrics={"val_wer": float("nan"), "val_vector": [1, 2]})
            row = json.loads(history.jsonl_path.read_text())
            self.assertIsNone(row["val_wer"])
            self.assertNotIn("val_vector", row)

    def test_unvalidated_checkpoint_cannot_inherit_old_validation_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "example.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            history = EpochHistory(root, "checkpoint_history")
            history.record(epoch_index=5, global_step=60, checkpoint=checkpoint,
                           metrics={"val_wer": 0.3}, validation_completed=False)
            row = json.loads(history.jsonl_path.read_text())
            self.assertEqual(row["epoch"], 6)
            self.assertFalse(row["validation_completed"])
            self.assertNotIn("val_wer", row)

    def test_resume_segments_preserve_training_rows_and_update_current_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = ProgressHistory(root, "first")
            first.append({"epoch": 1, "global_step": 50, "train_loss": 0.5,
                          "train_metrics": {"train_ctc_loss": 0.2}})
            original = first.jsonl_path.read_text()
            second = ProgressHistory(root, "resumed")
            second.append({"epoch": 2, "global_step": 100, "train_loss": 0.4,
                           "train_metrics": {"train_ctc_loss": 0.1}})
            self.assertEqual(first.jsonl_path.read_text(), original)
            self.assertEqual(len(second.jsonl_path.read_text().splitlines()), 1)
            status = json.loads(second.status_path.read_text())
            self.assertEqual(status["segment"], "resumed")
            self.assertEqual(status["global_step"], 100)
            self.assertEqual(status["train_metrics"]["train_ctc_loss"], 0.1)


if __name__ == "__main__":
    unittest.main()
