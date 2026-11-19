"""CPU-only integration check for the installed Lightning/Torch checkpoint loader."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("lightning"), "requires course runtime")
class TrainingResumeTests(unittest.TestCase):
    def test_optimizer_scheduler_and_omegaconf_checkpoint_restore(self):
        from train import CheckpointIdentity
        import lightning.pytorch as pl
        from omegaconf import OmegaConf
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        class TinyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.save_hyperparameters({"cfg": OmegaConf.create({"resume_probe": True})})
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.restored_state = None

            def training_step(self, batch, batch_idx):
                return (self.weight * batch[0]).square().mean()

            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.ones(4)), batch_size=2)

            def configure_optimizers(self):
                optimizer = torch.optim.Adam([self.weight], lr=0.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

            def on_train_start(self):
                optimizer = self.trainer.optimizers[0]
                self.restored_state = {
                    "global_step": self.trainer.global_step,
                    "optimizer_step": float(optimizer.state.get(self.weight, {}).get("step", 0)),
                    "lr": optimizer.param_groups[0]["lr"],
                    "scheduler_epoch": self.trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                }

        with tempfile.TemporaryDirectory(prefix="proj1_resume_test_") as directory:
            options = dict(accelerator="cpu", devices=1, max_epochs=2, logger=False,
                           enable_checkpointing=False, enable_progress_bar=False,
                           enable_model_summary=False, default_root_dir=directory)
            first = pl.Trainer(**options, max_steps=1,
                               callbacks=[CheckpointIdentity("tokenizer-test", "baseline", "config-test")])
            first.fit(TinyModel())
            path = Path(directory) / "probe.ckpt"
            first.save_checkpoint(path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["proj1"]["phase"], "baseline")
            self.assertEqual(saved["proj1"]["config_sha256"], "config-test")
            second = pl.Trainer(**options, max_steps=2,
                                callbacks=[CheckpointIdentity("tokenizer-test", "baseline", "config-test")])
            model = TinyModel()
            second.fit(model, ckpt_path=path)
            self.assertEqual(model.restored_state["global_step"], 1)
            self.assertEqual(model.restored_state["optimizer_step"], 1)
            self.assertAlmostEqual(model.restored_state["lr"], 0.005)
            self.assertEqual(model.restored_state["scheduler_epoch"], 1)
            self.assertEqual(second.global_step, 2)
            self.assertEqual(float(second.optimizers[0].state[model.weight]["step"]), 2)


if __name__ == "__main__":
    unittest.main()
