from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from prepare_eval import create_noisy, prepare_validation


class ValidationPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.golos = self.speech('golos', ['привет мир', 'ещё одна запись'])
        self.sova = self.speech('sova', ['запись совы', 'тест русского текста'])
        self.write('val_original.jsonl', self.golos)
        self.write('val_sova_original.jsonl', self.sova)
        self.write('val_mixed_original.jsonl', self.golos + self.sova)
        noise = self.root / 'fan.wav'
        rng = np.random.default_rng(112)
        sf.write(noise, rng.normal(0, 0.05, 4000), 16000, subtype='FLOAT')
        self.write('noise_val.jsonl', [{'audio_filepath': str(noise), 'duration': 0.25}])

    def speech(self, domain, texts):
        rows = []
        time = np.arange(8000, dtype=np.float32) / 16000
        for index, text in enumerate(texts):
            identifier = f'{domain}-{index}'
            path = self.root / (identifier + '.wav')
            waveform = 0.2 * np.sin(2 * np.pi * (220 + index * 110) * time)
            sf.write(path, waveform, 16000, subtype='FLOAT')
            rows.append({'id': identifier, 'audio_filepath': str(path), 'duration': 0.5, 'text': text})
        return rows

    def write(self, name, rows):
        (self.data / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

    def read(self, name):
        return [json.loads(line) for line in (self.data / name).read_text(encoding='utf-8').splitlines()]

    def generate_sources(self):
        for name in ('val_original.jsonl', 'val_sova_original.jsonl'):
            create_noisy(self.root, name, 'noise_val.jsonl', [10])

    def test_mixed_validation_contains_aligned_original_and_noisy_versions_of_both_corpora(self):
        selection = prepare_validation(self.root, 'mixed')
        self.assertEqual(selection, self.data / 'val_mixed_selection.jsonl')
        originals = self.golos + self.sova
        rows = self.read(selection.name)
        noisy = self.read('val_mixed_fan_10db.jsonl')
        self.assertEqual(rows[:len(originals)], originals)
        self.assertEqual(rows[len(originals):], noisy)
        self.assertEqual(len(rows), 2 * len(originals))
        self.assertEqual(Counter(row['text'] for row in rows), Counter({row['text']: 2 for row in originals}))
        self.assertEqual(sum(len(row['text'].split()) for row in rows),
                         2 * sum(len(row['text'].split()) for row in originals))
        for original, augmented in zip(originals, noisy):
            self.assertEqual(augmented['source_id'], original['id'])
            self.assertEqual(augmented['text'], original['text'])
            self.assertEqual(augmented['mix']['requested_snr_db'], 10)
            mixed_audio, rate = sf.read(augmented['audio_filepath'])
            source_audio, _ = sf.read(original['audio_filepath'])
            self.assertEqual(rate, 16000)
            self.assertEqual(mixed_audio.shape, source_audio.shape)
            self.assertTrue(np.isfinite(mixed_audio).all())
            self.assertFalse(np.array_equal(mixed_audio, source_audio))

    def test_existing_golos_fixed_noise_is_reused_without_modifying_files(self):
        create_noisy(self.root, 'val_original.jsonl', 'noise_val.jsonl', [10])
        paths = [self.data / 'val_fan_10db.jsonl'] + [Path(row['audio_filepath']) for row in self.read('val_fan_10db.jsonl')]
        before = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths}
        prepare_validation(self.root, 'mixed')
        after = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(before, after)
        self.assertTrue((self.data / 'val_sova_fan_10db.jsonl').is_file())

    def test_stale_noisy_source_ids_and_transcripts_are_rejected(self):
        self.generate_sources()
        for name in ('val_fan_10db.jsonl', 'val_sova_fan_10db.jsonl'):
            expected = self.read(name)
            for field in ('source_id', 'text'):
                with self.subTest(manifest=name, field=field):
                    changed = [dict(row) for row in expected]
                    changed[0][field] = 'stale'
                    self.write(name, changed)
                    try:
                        with self.assertRaisesRegex(ValueError, 'does not match its source'):
                            prepare_validation(self.root, 'mixed')
                        self.assertFalse((self.data / 'val_mixed_selection.jsonl').exists())
                    finally:
                        self.write(name, expected)

    def test_missing_mixed_original_fails_before_generating_noise(self):
        (self.data / 'val_mixed_original.jsonl').unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_validation(self.root, 'mixed')
        self.assertFalse((self.data / 'val_fan_10db.jsonl').exists())
        self.assertFalse((self.data / 'val_sova_fan_10db.jsonl').exists())

    def test_mixed_original_order_and_transcripts_must_match_sources(self):
        self.generate_sources()
        originals = self.golos + self.sova
        altered_text = [dict(row) for row in originals]
        altered_text[0]['text'] = 'устаревшая расшифровка'
        for case, rows in [('order', list(reversed(originals))), ('text', altered_text)]:
            with self.subTest(case=case):
                self.write('val_mixed_original.jsonl', rows)
                with self.assertRaisesRegex(ValueError, 'Mixed validation'):
                    prepare_validation(self.root, 'mixed')
                self.assertFalse((self.data / 'val_mixed_selection.jsonl').exists())

    def test_missing_cached_noisy_audio_is_rejected(self):
        self.generate_sources()
        Path(self.read('val_sova_fan_10db.jsonl')[0]['audio_filepath']).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Noisy validation audio is missing'):
            prepare_validation(self.root, 'mixed')


if __name__ == '__main__':
    unittest.main()
