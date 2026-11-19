import unittest
import numpy as np
from audio_utils import mix_at_snr


class MixingTests(unittest.TestCase):
    def test_snr_length_and_no_clipping(self):
        rng = np.random.default_rng(112)
        speech = rng.normal(0, 0.2, 32000)
        noise = rng.normal(0, 0.1, 8000)
        mixed, meta = mix_at_snr(speech, noise, 10, 97)
        gain = meta["common_gain"]
        actual_noise = mixed.astype(float) - gain * speech
        actual = 10 * np.log10(np.mean((gain * speech) ** 2) / np.mean(actual_noise ** 2))
        self.assertAlmostEqual(actual, 10, places=4)
        self.assertEqual(len(mixed), len(speech))
        self.assertLessEqual(float(abs(mixed).max()), 0.990001)

    def test_deterministic_and_silence(self):
        speech = np.ones(100, dtype=float) * 0.1
        noise = np.arange(1, 12, dtype=float) * 0.01
        np.testing.assert_array_equal(mix_at_snr(speech, noise, 5, 3)[0], mix_at_snr(speech, noise, 5, 3)[0])
        self.assertTrue(mix_at_snr(np.zeros(100), noise, 10)[1]["skipped_silent_original"])
        with self.assertRaises(ValueError):
            mix_at_snr(speech, np.zeros(10), 10)


if __name__ == "__main__":
    unittest.main()
