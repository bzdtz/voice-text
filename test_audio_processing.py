import unittest

import numpy as np

from audio_processing import apply_auto_gain


class AutoGainTests(unittest.TestCase):
    def test_quiet_speech_is_amplified(self):
        audio = np.full((1600, 1), 0.02, dtype=np.float32)
        gained = apply_auto_gain(audio)
        self.assertGreater(float(np.sqrt(np.mean(gained ** 2))), 0.1)

    def test_peak_limiter_prevents_clipping(self):
        audio = np.array([[0.5], [-0.5]], dtype=np.float32)
        gained = apply_auto_gain(audio)
        self.assertLessEqual(float(np.max(np.abs(gained))), 0.98001)

    def test_near_silence_is_not_amplified(self):
        audio = np.full((100, 1), 0.0005, dtype=np.float32)
        np.testing.assert_allclose(apply_auto_gain(audio), audio)


if __name__ == "__main__":
    unittest.main()
