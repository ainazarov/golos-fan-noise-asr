import tempfile
import unittest
from pathlib import Path
from checkpoint_utils import configuration_sha256, initialization_source_phase, resume_run_directory, validate_checkpoint_identity


class CheckpointIdentityTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"tokenizer_sha256": "tokenizer", "vocab_size": 112,
                         "phase": "noise", "config_sha256": "config"}

    def test_configuration_hash_ignores_mapping_order_but_tracks_schedule(self):
        first = {"trainer": {"max_epochs": 30}, "seed": 112}
        reordered = {"seed": 112, "trainer": {"max_epochs": 30}}
        changed = {"seed": 112, "trainer": {"max_epochs": 50}}
        self.assertEqual(configuration_sha256(first), configuration_sha256(reordered))
        self.assertNotEqual(configuration_sha256(first), configuration_sha256(changed))

    def test_matching_resume_identity_is_accepted(self):
        validate_checkpoint_identity(self.identity, tokenizer_sha256="tokenizer",
                                     phase="noise", config_sha256="config")

    def test_resume_rejects_each_changed_identity_field(self):
        for field in self.identity:
            with self.subTest(field=field):
                changed = dict(self.identity, **{field: "different"})
                with self.assertRaisesRegex(ValueError, field):
                    validate_checkpoint_identity(changed, tokenizer_sha256="tokenizer",
                                                 phase="noise", config_sha256="config")

    def test_continuation_requires_its_immediate_parent_and_same_tokenizer(self):
        for phase, expected_parent in [('mixed', 'baseline'), ('noise', 'mixed')]:
            with self.subTest(phase=phase):
                self.assertEqual(initialization_source_phase(phase), expected_parent)
                parent = dict(self.identity, phase=expected_parent)
                validate_checkpoint_identity(parent, tokenizer_sha256="tokenizer", phase=initialization_source_phase(phase))
                for wrong_parent in set(('baseline', 'mixed', 'noise')) - {expected_parent}:
                    with self.assertRaisesRegex(ValueError, "phase"):
                        validate_checkpoint_identity(dict(parent, phase=wrong_parent), tokenizer_sha256="tokenizer", phase=initialization_source_phase(phase))
                with self.assertRaisesRegex(ValueError, "tokenizer_sha256"):
                    validate_checkpoint_identity(parent, tokenizer_sha256="other", phase=initialization_source_phase(phase))

    def test_initial_stage_and_unknown_phases_cannot_import_a_previous_stage(self):
        for phase in ('baseline', 'benchmark', 'control', 'unknown'):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                initialization_source_phase(phase)

    def test_missing_identity_is_rejected(self):
        for identity in [None, {}]:
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                validate_checkpoint_identity(identity, tokenizer_sha256="tokenizer", phase="baseline")

    def test_resume_reuses_original_run_and_rejects_other_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "baseline_example"
            self.assertEqual(resume_run_directory(run / "checkpoints" / "last.ckpt", root), run)
            for path in [root / "last.ckpt", root / "outside" / "checkpoints" / "last.ckpt"]:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    resume_run_directory(path, root)


if __name__ == "__main__":
    unittest.main()
