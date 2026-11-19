"""CPU integration check of epoch retention and persistent validation results."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("lightning"), "requires course runtime")
class EpochCheckpointRuntimeTests(unittest.TestCase):
    def test_sampled_nonfinite_training_loss_stops_progress(self):
        from train import TrainingProgress
        import torch
        with tempfile.TemporaryDirectory() as directory:
            progress = TrainingProgress(directory, "nonfinite")
            trainer = SimpleNamespace(global_step=0, current_epoch=0, is_global_zero=True,
                                      callback_metrics={}, num_training_batches=100,
                                      optimizers=[SimpleNamespace(param_groups=[{"lr": 0.01}])])
            module = SimpleNamespace(device=torch.device("cpu"))
            progress.on_train_start(trainer, module)
            trainer.global_step = 50
            with self.assertRaisesRegex(FloatingPointError, "optimizer step 50"):
                progress.on_train_batch_end(trainer, module, {"loss": torch.tensor(float("nan"))}, None, 49)

    def test_all_epoch_checkpoints_have_matching_validation_and_training_logs(self):
        from train import AuditedModelCheckpoint, CheckpointIdentity, TrainingProgress
        import lightning.pytorch as pl
        from lightning.pytorch.loggers import CSVLogger
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class TinyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def training_step(self, batch, batch_idx):
                loss = (self.weight * batch[0]).square().mean()
                self.log("train_loss", loss, on_step=True, on_epoch=False)
                return loss

            def validation_step(self, batch, batch_idx):
                value = 0.3
                self.log("val_wer", value, on_step=False, on_epoch=True, batch_size=2)
                self.log("val_wer_ctc", value + 0.1, on_step=False, on_epoch=True, batch_size=2)

            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.ones(4)), batch_size=2)

            def val_dataloader(self):
                return DataLoader(TensorDataset(torch.ones(2)), batch_size=2)

            def configure_optimizers(self):
                return torch.optim.Adam([self.weight], lr=0.01)

        with tempfile.TemporaryDirectory(prefix="proj1_epoch_log_test_") as directory:
            root = Path(directory)
            checkpoint = AuditedModelCheckpoint(
                run_dir=root, dirpath=str(root / "checkpoints"), monitor=None,
                save_top_k=-1, save_last=True, filename="epoch={epoch:02d}-step={step}",
                auto_insert_metric_name=False, every_n_epochs=1, save_on_train_epoch_end=True)
            logger = CSVLogger(save_dir=str(root), name="csv", version="segment", flush_logs_every_n_steps=1)
            progress = TrainingProgress(root, "segment", interval=1, stdout_interval=100)
            trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=6, logger=logger,
                                 callbacks=[CheckpointIdentity("tokenizer-test", "baseline", "config-test"), progress, checkpoint],
                                 log_every_n_steps=1, num_sanity_val_steps=0,
                                 check_val_every_n_epoch=5,
                                 enable_progress_bar=False, enable_model_summary=False,
                                 default_root_dir=str(root))
            trainer.fit(TinyModel())
            self.assertEqual(len(list((root / "checkpoints").glob("epoch=*.ckpt"))), 6)
            self.assertTrue((root / "checkpoints" / "last.ckpt").is_file())
            history = [json.loads(line) for line in (root / "validation_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in history], [5])
            self.assertEqual([row["global_step"] for row in history], [10])
            for row, expected in zip(history, [0.3]):
                self.assertTrue((root / row["checkpoint"]).is_file())
                self.assertAlmostEqual(row["val_wer"], expected, places=6)
                self.assertAlmostEqual(row["val_wer_ctc"], expected + 0.1, places=6)
                self.assertGreater(row["checkpoint_bytes"], 0)
            checkpoint_history = [json.loads(line) for line in (root / "checkpoint_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in checkpoint_history], list(range(1, 7)))
            self.assertEqual([row["epoch"] for row in checkpoint_history if row["validation_completed"]], [5])
            for row in checkpoint_history:
                if row["epoch"] != 5:
                    self.assertNotIn("val_wer", row)
                    self.assertNotIn("val_wer_ctc", row)
            saved = torch.load(root / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(saved["epoch"], 5)
            self.assertEqual(saved["global_step"], 12)
            self.assertIn("optimizer_states", saved)
            with (root / "csv" / "segment" / "metrics.csv").open(newline="") as stream:
                metrics = list(csv.DictReader(stream))
            self.assertTrue(any(row.get("train_loss") for row in metrics))
            self.assertTrue(any(row.get("val_wer_ctc") for row in metrics))
            progress_rows = [json.loads(line) for line in (root / "progress/segment/progress.jsonl").read_text().splitlines()]
            self.assertEqual([row["global_step"] for row in progress_rows], list(range(1, 13)))
            self.assertTrue(all(row["train_loss"] is not None for row in progress_rows))
            self.assertTrue(all(row["gpu_allocated_gib"] is None for row in progress_rows))
            self.assertTrue(all("val_wer" not in row["train_metrics"] for row in progress_rows))
            status = json.loads((root / "progress_status.json").read_text())
            self.assertEqual(status["status"], "finished")
            self.assertEqual(status["epoch"], 6)
            self.assertEqual(status["global_step"], 12)


if __name__ == "__main__":
    unittest.main()
