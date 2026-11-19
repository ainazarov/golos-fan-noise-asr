"""Metric-only tests: no NeMo imports, model loading, or GPU work."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import WordErrorAccumulator, normalize_text, word_edit_counts


class WordErrorTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace_but_preserves_punctuation(self):
        self.assertEqual(normalize_text("  ПРИВЕТ,\n   Мир!  "), "привет, мир!")
        self.assertEqual(word_edit_counts("hello", "hello,")["substitutions"], 1)

    def test_identical_normalized_words_have_zero_errors(self):
        counts = word_edit_counts("  ПРИВЕТ   мир ", "привет\nМИР")
        self.assertEqual(counts["reference_words"], 2)
        self.assertEqual(counts["hypothesis_words"], 2)
        self.assertEqual(sum(counts[k] for k in ("substitutions", "deletions", "insertions")), 0)

    def test_substitution_and_insertion(self):
        counts = word_edit_counts("alpha beta gamma delta", "alpha wrong gamma extra delta")
        self.assertEqual(counts["substitutions"], 1)
        self.assertEqual(counts["deletions"], 0)
        self.assertEqual(counts["insertions"], 1)

    def test_deletion(self):
        counts = word_edit_counts("alpha beta gamma", "alpha gamma")
        self.assertEqual(counts["substitutions"], 0)
        self.assertEqual(counts["deletions"], 1)
        self.assertEqual(counts["insertions"], 0)

    def test_empty_hypothesis_counts_every_reference_word_as_deletion(self):
        counts = word_edit_counts("one two three", "")
        self.assertEqual(counts["deletions"], 3)
        self.assertEqual(counts["insertions"], 0)

    def test_empty_reference_counts_every_prediction_as_insertion(self):
        counts = word_edit_counts(" \n", "one two")
        self.assertEqual(counts["reference_words"], 0)
        self.assertEqual(counts["insertions"], 2)

    def test_corpus_wer_uses_total_words_not_mean_of_record_wers(self):
        accumulator = WordErrorAccumulator()
        accumulator.add("one two three four", "one two three four")
        accumulator.add("five", "wrong")
        summary = accumulator.summary()
        self.assertEqual(summary["word_errors"], 1)
        self.assertEqual(summary["reference_words"], 5)
        self.assertAlmostEqual(summary["wer"], 0.2)

    def test_empty_reference_insertions_contribute_to_corpus_wer(self):
        accumulator = WordErrorAccumulator()
        accumulator.add("one two three four five", "one two three four five")
        accumulator.add("", "unexpected words")
        accumulator.add("", "")
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["wer"], 0.4)
        self.assertEqual(summary["empty_reference_records"], 2)
        self.assertEqual(summary["empty_reference_insertions"], 2)
        self.assertEqual(summary["empty_reference_nonempty_hypotheses"], 1)

    def test_all_empty_references_have_undefined_wer(self):
        accumulator = WordErrorAccumulator()
        accumulator.add("", "unexpected")
        summary = accumulator.summary()
        self.assertIsNone(summary["wer"])
        self.assertIsNone(summary["wer_percent"])
        self.assertEqual(summary["insertions"], 1)

    def test_wer_may_exceed_one(self):
        accumulator = WordErrorAccumulator()
        accumulator.add("one", "wrong extra words")
        self.assertEqual(accumulator.summary()["wer"], 3.0)


if __name__ == "__main__":
    unittest.main()
