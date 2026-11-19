import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_sova import mapped_audio_path, parse_args, prepare


class SovaPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "sova.jsonl"
        self.audio = self.root / "sova_audio"
        self.output = self.root / "project"
        self.rows = []
        for index in range(30):
            identifier = str(uuid.UUID(int=index + 1))
            relative = f"collection/0/0/{identifier}.wav"
            self.rows.append({"audio_filepath": "/root/dataset/sova/" + relative,
                              "duration": 2.0, "text": " привет   мир "})
            target = self.audio / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"existence-fixture")
        self.write_source()
        self.golos_bytes = {}
        for name in ("train", "val_original", "test_crowd_original", "test_farfield_original"):
            path = self.output / "data" / (name + ".jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {"id": "golos-" + name, "audio_filepath": "/golos/" + name + ".wav",
                   "duration": 3.0, "text": "пример"}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.golos_bytes[name] = path.read_bytes()
        self.tokenizer = self.output / "tokenizer112/tokenizer.model"
        self.tokenizer.parent.mkdir()
        self.tokenizer.write_bytes(b"existing-tokenizer")
        (self.output / "data/train_text.txt").write_text("original training corpus\n")

    def tearDown(self):
        self.temporary.cleanup()

    def write_source(self):
        self.source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8")

    def arguments(self, *extra):
        return parse_args(["--source-manifest", str(self.source), "--audio-root", str(self.audio),
                           "--output-root", str(self.output), "--dev-hours", "0.002",
                           "--test-hours", "0.002", *extra])

    def output_rows(self, name):
        return [json.loads(line) for line in (self.output / "data" / (name + ".jsonl")).read_text(encoding="utf-8").splitlines()]

    def test_holdouts_are_disjoint_order_independent_and_keep_original_files(self):
        report = prepare(self.arguments())
        names = ("train_sova", "val_sova_original", "test_sova_original")
        first = {name: self.output_rows(name) for name in names}
        ids = [{row["id"] for row in first[name]} for name in names]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(len(set.union(*ids)), 30)
        for name in names[1:]:
            self.assertGreaterEqual(report["splits"][name]["seconds"], 7.2)
            self.assertLess(report["splits"][name]["seconds"], 9.2)
        self.assertTrue(all(row["text"] == "привет мир" for rows in first.values() for row in rows))
        self.assertEqual(self.output_rows("train_mixed")[1:], first["train_sova"])
        self.assertEqual(self.output_rows("val_mixed_original")[1:], first["val_sova_original"])
        self.rows.reverse()
        self.write_source()
        prepare(self.arguments("--overwrite"))
        self.assertEqual(first, {name: self.output_rows(name) for name in names})
        for name, content in self.golos_bytes.items():
            self.assertEqual((self.output / "data" / (name + ".jsonl")).read_bytes(), content)
        self.assertEqual(self.tokenizer.read_bytes(), b"existing-tokenizer")
        self.assertEqual((self.output / "data/train_text.txt").read_text(), "original training corpus\n")

    def test_filters_duration_and_empty_text_before_reserving_holdouts(self):
        self.rows[0]["duration"] = 0.1
        self.rows[1]["duration"] = 21.0
        self.rows[2]["text"] = " \t "
        self.write_source()
        report = prepare(self.arguments())
        self.assertEqual(report["sova_source"]["excluded"]["unique_excluded"], 3)
        self.assertEqual(sum(report["splits"][name]["recordings"] for name in
                             ("train_sova", "val_sova_original", "test_sova_original")), 27)

    def test_yo_normalization_is_identical_across_all_sova_splits(self):
        for row in self.rows:
            row["text"] = "  ёлка ещё   твёрдый! "
        self.rows[0]["duration"] = 0.1
        self.write_source()
        report = prepare(self.arguments())
        for name in ("train_sova", "val_sova_original", "test_sova_original"):
            self.assertTrue(all(row["text"] == "елка еще твердый!" for row in self.output_rows(name)))
        self.assertEqual(report["sova_source"]["yo_to_e_normalization"], {
            "source_changed_recordings": 30, "source_replaced_characters": 90,
            "eligible_changed_recordings": 29, "eligible_replaced_characters": 87,
        })
        for name, content in self.golos_bytes.items():
            self.assertEqual((self.output / "data" / (name + ".jsonl")).read_bytes(), content)

    def test_rejects_normalized_duplicate_uuid(self):
        duplicate = dict(self.rows[0])
        identifier = str(uuid.UUID(int=1)).replace("-", "").upper()
        duplicate["audio_filepath"] = f"/root/dataset/sova/another/{identifier}.wav"
        self.rows.append(duplicate)
        self.write_source()
        with self.assertRaisesRegex(ValueError, "duplicate Sova"):
            prepare(self.arguments())
        self.assertFalse((self.output / "data/train_sova.jsonl").exists())

    def test_rejects_overlap_with_golos_test_even_when_identity_format_differs(self):
        path = self.output / "data/test_farfield_original.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["id"] = str(uuid.UUID(int=1)).replace("-", "").upper()
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "overlap Golos"):
            prepare(self.arguments())
        self.assertFalse((self.output / "data/train_mixed.jsonl").exists())
        report = json.loads((self.output / "reports/sova_audit.json").read_text())
        self.assertEqual(report["golos_overlap_recording_counts"]["test_farfield_original"], 1)

    def test_source_path_mapping_rejects_traversal_and_unexpected_prefix(self):
        identifier = str(uuid.UUID(int=1))
        identity, mapped = mapped_audio_path(f"/root/dataset/sova/sub/{identifier}.wav", self.audio)
        self.assertEqual(identity, identifier)
        self.assertEqual(mapped, self.audio / "sub" / (identifier + ".wav"))
        for value in (f"/elsewhere/sova/{identifier}.wav", f"/root/dataset/sova/../{identifier}.wav"):
            with self.assertRaises(ValueError):
                mapped_audio_path(value, self.audio)

    def test_missing_audio_fails_before_writing_any_new_manifest(self):
        _, path = mapped_audio_path(self.rows[0]["audio_filepath"], self.audio)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing or inaccessible"):
            prepare(self.arguments())
        self.assertFalse((self.output / "data/train_mixed.jsonl").exists())
        self.assertFalse((self.output / "data/train_sova.jsonl").exists())

    def test_refuses_overwrite(self):
        prepare(self.arguments())
        with self.assertRaises(FileExistsError):
            prepare(self.arguments())


if __name__ == "__main__":
    unittest.main()
