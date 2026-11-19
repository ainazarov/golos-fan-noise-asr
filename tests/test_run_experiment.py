import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from run_experiment import STAGE_EPOCHS, run_experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.calls = []

    def stage(self, args):
        self.calls.append(args)
        run = self.root / 'runs' / (args.phase + '_example')
        (run / 'checkpoints').mkdir(parents=True)
        rows = []
        epochs = [(5, 0.2), (STAGE_EPOCHS[args.phase], 0.3)] if args.phase != 'noise' else [(5, 0.2)]
        for epoch, wer in epochs:
            checkpoint = f'checkpoints/epoch={epoch}.ckpt'
            (run / checkpoint).write_bytes(b'checkpoint')
            rows.append({'epoch': epoch, 'checkpoint': checkpoint, 'val_wer': wer})
        (run / 'validation_history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (run / 'checkpoint_history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        status = self.root / 'logs' / (args.phase + '_status.json')
        status.parent.mkdir(exist_ok=True)
        status.write_text(json.dumps({'status': 'training_complete', 'run_dir': str(run)}))
        return status

    def test_runs_sequential_stages_from_previous_final_not_best_checkpoint(self):
        status_path = run_experiment(self.root, run_training=True, stage_runner=self.stage)
        self.assertEqual([args.phase for args in self.calls], ['baseline', 'mixed', 'noise'])
        self.assertIsNone(self.calls[0].init_from)
        self.assertEqual(self.calls[1].init_from, self.root / 'runs/baseline_example/checkpoints/epoch=10.ckpt')
        self.assertEqual(self.calls[2].init_from, self.root / 'runs/mixed_example/checkpoints/epoch=25.ckpt')
        for args in self.calls:
            self.assertIsNone(args.resume)
            self.assertFalse(args.evaluate_after_training)
        state = json.loads(status_path.read_text())
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['stage_epochs'], {'baseline': 10, 'mixed': 25, 'noise': 5})
        self.assertEqual(list(state['completed_stages']), ['baseline', 'mixed', 'noise'])
        self.assertEqual([stage['final_checkpoint_epoch'] for stage in state['completed_stages'].values()], [10, 25, 5])
        self.assertEqual(state['final_checkpoint'], str(self.root / 'runs/noise_example/checkpoints/epoch=5.ckpt'))
        self.assertFalse(state['full_test_evaluation'])

    def test_failure_stops_sequence_and_preserves_exit_code(self):
        def stage(args):
            if args.phase == 'mixed':
                self.calls.append(args)
                raise subprocess.CalledProcessError(9, ['training'])
            return self.stage(args)
        with self.assertRaises(subprocess.CalledProcessError):
            run_experiment(self.root, run_training=True, stage_runner=stage)
        self.assertEqual([args.phase for args in self.calls], ['baseline', 'mixed'])
        state = json.loads(next((self.root / 'logs').glob('experiment_*.json')).read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['phase'], 'mixed')
        self.assertEqual(state['returncode'], 9)
        self.assertEqual(list(state['completed_stages']), ['baseline'])

    def test_partial_stage_completion_does_not_advance(self):
        def stage(args):
            status = self.stage(args)
            run = self.root / 'runs' / (args.phase + '_example')
            for name in ('validation_history.jsonl', 'checkpoint_history.jsonl'):
                path = run / name
                rows = path.read_text().splitlines()
                path.write_text(rows[0] + '\n')
            return status
        with self.assertRaisesRegex(ValueError, 'expected 10'):
            run_experiment(self.root, run_training=True, stage_runner=stage)
        self.assertEqual([args.phase for args in self.calls], ['baseline'])

    def test_unvalidated_final_checkpoint_does_not_advance(self):
        def stage(args):
            status = self.stage(args)
            run = self.root / 'runs' / (args.phase + '_example')
            path = run / 'validation_history.jsonl'
            path.write_text(path.read_text().splitlines()[0] + '\n')
            path = run / 'checkpoint_history.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1].pop('val_wer')
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            return status
        with self.assertRaisesRegex(ValueError, 'final checkpoint'):
            run_experiment(self.root, run_training=True, stage_runner=stage)
        self.assertEqual([args.phase for args in self.calls], ['baseline'])

    def test_requires_explicit_training_authorization(self):
        with self.assertRaisesRegex(ValueError, '--run-training'):
            run_experiment(self.root, stage_runner=self.stage)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'logs').exists())


if __name__ == '__main__':
    unittest.main()
