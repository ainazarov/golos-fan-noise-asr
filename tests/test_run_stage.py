import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from run_stage import evaluate_run, parse_args, prepare_test_manifests, run_training_stage, select_checkpoints


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "runs" / "baseline_example"
        (self.run / "checkpoints").mkdir(parents=True)
        (self.root / "data").mkdir()
        self.write_history([0.6, 0.2, 0.3])
        for domain in ("crowd", "farfield"):
            for condition in ("original", "fan_5db", "fan_10db", "fan_20db"):
                (self.root / "data" / f"test_{domain}_{condition}.jsonl").write_text('{}\n')

    def write_history(self, wers):
        rows = []
        for epoch, wer in enumerate(wers, 1):
            checkpoint = f"checkpoints/epoch={epoch:02d}.ckpt"
            (self.run / checkpoint).write_bytes(f"weights-{epoch}".encode())
            rows.append({"epoch": epoch, "global_step": epoch * 10, "checkpoint": checkpoint, "val_wer": wer})
        (self.run / "validation_history.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def evaluation_runner(self, calls):
        def run(command, **kwargs):
            calls.append(command)
            def argument(name):
                return command[command.index(name) + 1]
            output = Path(argument("--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"prediction":"example"}\n')
            output.with_suffix('.summary.json').write_text(json.dumps({
                "checkpoint": argument("--checkpoint"), "manifest": argument("--manifest"),
                "predictions": str(output), "wer": 0.2, "wer_percent": 20.0, "records": 1,
                "reference_words": 5, "word_errors": 1,
            }))
            return subprocess.CompletedProcess(command, 0)
        return run

    def test_selects_validation_best_and_final_and_deduplicates(self):
        selections = select_checkpoints(self.run)
        self.assertEqual([(row['role'], row['epoch']) for row in selections], [('best', 2), ('final', 3)])
        self.write_history([0.6, 0.4, 0.2])
        selection = select_checkpoints(self.run)
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0]['selected_as'], ['best', 'final'])

    def test_final_selection_includes_checkpoint_without_validation(self):
        self.write_history([0.6, 0.2])
        checkpoint = 'checkpoints/epoch=07.ckpt'
        (self.run / checkpoint).write_bytes(b'epoch-seven')
        (self.run / 'checkpoint_history.jsonl').write_text(json.dumps({
            'epoch': 7, 'global_step': 70, 'checkpoint': checkpoint,
        }) + '\n')
        selections = select_checkpoints(self.run)
        self.assertEqual([(row['role'], row['epoch']) for row in selections], [('best', 2), ('final', 7)])

    def test_rejects_missing_and_outside_checkpoint_paths(self):
        for name in ['checkpoints/missing.ckpt', '../outside.ckpt']:
            with self.subTest(name=name):
                (self.run / 'validation_history.jsonl').write_text(json.dumps({'epoch': 1, 'checkpoint': name, 'val_wer': 0.2}))
                with self.assertRaises(ValueError):
                    select_checkpoints(self.run)

    def test_only_missing_noise_manifests_are_prepared(self):
        missing = self.root / 'data/test_crowd_fan_10db.jsonl'
        missing.unlink()
        calls = []
        def creator(root, source, noise, snrs):
            calls.append((source, noise, snrs))
            missing.write_text('{}\n')
        manifests = prepare_test_manifests(self.root, creator)
        self.assertEqual(len(manifests), 8)
        self.assertEqual(calls, [('test_crowd_original.jsonl', 'noise_test.jsonl', [10])])

    def test_sova_test_conditions_are_included_when_reserved(self):
        (self.root / 'data/test_sova_original.jsonl').write_text('{}\n')
        calls = []
        def creator(root, source, noise, snrs):
            calls.append((source, noise, snrs))
            for snr in snrs:
                (root / 'data' / f'test_sova_fan_{snr}db.jsonl').write_text('{}\n')
        manifests = prepare_test_manifests(self.root, creator)
        self.assertEqual(len(manifests), 12)
        self.assertEqual([path.name for path in manifests[-4:]], [
            'test_sova_original.jsonl', 'test_sova_fan_5db.jsonl',
            'test_sova_fan_10db.jsonl', 'test_sova_fan_20db.jsonl',
        ])
        self.assertEqual(calls, [('test_sova_original.jsonl', 'noise_test.jsonl', [snr]) for snr in (5, 10, 20)])

    def test_evaluation_has_sixteen_results_and_reuses_only_matching_outputs(self):
        calls = []
        summary_path = evaluate_run(self.root, self.run, runner=self.evaluation_runner(calls))
        self.assertEqual(len(calls), 16)
        self.assertEqual(len(json.loads(summary_path.read_text())['results']), 16)
        with summary_path.with_suffix('.csv').open(newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 16)
        calls.clear()
        evaluate_run(self.root, self.run, runner=self.evaluation_runner(calls))
        self.assertEqual(calls, [])
        (self.root / 'data/test_crowd_original.jsonl').write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            evaluate_run(self.root, self.run, runner=self.evaluation_runner(calls))
        self.assertEqual(json.loads((self.run / 'evaluation/status.json').read_text())['status'], 'failed')

    def test_retry_after_partial_evaluation_preserves_failure_code(self):
        self.write_history([0.2])
        def fail(command, **kwargs):
            output = Path(command[command.index('--output') + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('partial\n')
            return subprocess.CompletedProcess(command, 9)
        with self.assertRaises(subprocess.CalledProcessError) as error:
            evaluate_run(self.root, self.run, runner=fail)
        self.assertEqual(error.exception.returncode, 9)
        state = json.loads((self.run / 'evaluation/status.json').read_text())
        self.assertEqual(state['returncode'], 9)
        calls = []
        evaluate_run(self.root, self.run, runner=self.evaluation_runner(calls))
        self.assertEqual(len(calls), 8)
        self.assertIn('--overwrite', calls[0])

    def stage_args(self):
        return argparse.Namespace(output_root=self.root, phase='baseline', run_training=True,
                                  init_from=None, resume=None, evaluate_after_training=False)

    def test_training_failure_never_evaluates_and_records_exit_code(self):
        evaluated = []
        def fail(command, **kwargs):
            return subprocess.CompletedProcess(command, 7)
        with self.assertRaises(subprocess.CalledProcessError) as error:
            run_training_stage(self.stage_args(), runner=fail, evaluator=lambda *x: evaluated.append(x))
        self.assertEqual(error.exception.returncode, 7)
        self.assertEqual(evaluated, [])
        state = json.loads(next((self.root / 'logs').glob('stage_*.json')).read_text())
        self.assertEqual(state['returncode'], 7)

    def test_success_does_not_evaluate_by_default(self):
        expected = self.root / 'runs/baseline_new'
        def train(command, **kwargs):
            expected.mkdir()
            return subprocess.CompletedProcess(command, 0)
        evaluated = []
        result = run_training_stage(self.stage_args(), runner=train, evaluator=lambda *x: evaluated.append(x))
        self.assertEqual(evaluated, [])
        state = json.loads(result.read_text())
        self.assertEqual(state['status'], 'training_complete')
        self.assertEqual(state['run_dir'], str(expected))
        self.assertFalse(state['full_test_evaluation_requested'])

    def test_explicit_opt_in_evaluates_only_the_new_run(self):
        expected = self.root / 'runs/baseline_new'
        def train(command, **kwargs):
            expected.mkdir()
            return subprocess.CompletedProcess(command, 0)
        evaluated = []
        def evaluator(root, run):
            evaluated.append(run)
            return run / 'evaluation/summary.json'
        args = self.stage_args()
        args.evaluate_after_training = True
        result = run_training_stage(args, runner=train, evaluator=evaluator)
        self.assertEqual(evaluated, [expected])
        self.assertEqual(result, expected / 'evaluation/summary.json')

    def test_cli_requires_training_authorization_or_evaluation_only(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([])
            with self.assertRaises(SystemExit):
                parse_args(['--evaluate-run', str(self.run), '--run-training'])
        self.assertFalse(parse_args(['--evaluate-run', str(self.run)]).run_training)
        self.assertFalse(parse_args(['--run-training']).evaluate_after_training)
        self.assertTrue(parse_args(['--run-training', '--evaluate-after-training']).evaluate_after_training)


if __name__ == '__main__':
    unittest.main()
