import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import parse_args, prepare


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "out"
        rows = []
        for condition in ["crowd", "farfield"]:
            for index in range(25):
                rows.append(self.record(f"{condition}-{index}", f"{condition}/{index}.wav", 2.0, " привет   мир "))
        rows.extend([
            self.record("empty", "crowd/empty.wav", 2.0, " \t "),
            self.record("too-short", "crowd/short.wav", 0.1, "слово"),
            self.record("both", "crowd/both.wav", 30.0, ""),
        ])
        self.write_manifest(self.source / "train", rows)
        for condition in ["crowd", "farfield"]:
            self.write_manifest(self.source / "test" / condition, [
                self.record(f"test-{condition}", "files/1.wav", 25.0, ""),
            ])

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def record(identifier, path, duration, text):
        return {"id": identifier, "audio_filepath": path, "duration": duration, "text": text}

    def write_manifest(self, base, rows):
        base.mkdir(parents=True, exist_ok=True)
        (base / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        for row in rows:
            audio = base / row["audio_filepath"]
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"existence-check-fixture")

    def arguments(self, output=None):
        return parse_args([
            "--source-root", str(self.source), "--output-root", str(output or self.output),
            "--dev-hours", "0.002", "--benchmark-size", "16",
        ])

    def read_output(self, name, output=None):
        return [json.loads(line) for line in ((output or self.output) / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_split_is_deterministic_and_recording_disjoint(self):
        report = prepare(self.arguments())
        another = self.root / "other"
        manifest = self.source / "train" / "manifest.jsonl"
        manifest.write_text("\n".join(reversed(manifest.read_text(encoding="utf-8").splitlines())) + "\n", encoding="utf-8")
        prepare(self.arguments(another))
        for name in report["splits"]:
            self.assertEqual(self.read_output(name), self.read_output(name, another))
        train = self.read_output("train")
        dev = self.read_output("val_original")
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in dev})
        self.assertEqual(len(train) + len(dev), 50)
        self.assertEqual(report["excluded_training_pool"]["unique_excluded"], 3)
        self.assertEqual(report["excluded_training_pool"]["both_duration_and_empty"], 1)
        self.assertTrue(all(Path(row["audio_filepath"]).is_absolute() for row in train))
        self.assertTrue(all(row["text"] == "привет мир" for row in train))
        self.assertEqual(len(self.read_output("benchmark_train")), 16)

    def test_official_test_retains_empty_text_and_long_recordings(self):
        prepare(self.arguments())
        row = self.read_output("test_crowd_original")[0]
        self.assertEqual(row["text"], "")
        self.assertEqual(row["duration"], 25)

    def test_refuses_overwrite(self):
        prepare(self.arguments())
        with self.assertRaises(FileExistsError):
            prepare(self.arguments())

    def test_fails_on_official_split_overlap_before_writing_train(self):
        self.write_manifest(self.source / "test" / "crowd", [self.record("crowd-0", "files/overlap.wav", 2.0, "слово")])
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare(self.arguments())
        self.assertFalse((self.output / "data" / "train.jsonl").exists())
        report = json.loads((self.output / "reports" / "data_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["overlap_recording_counts"]["train_pool_vs_test_crowd"], 1)


if __name__ == "__main__":
    unittest.main()
