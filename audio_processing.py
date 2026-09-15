"""Small, dependency-free audio preprocessing helpers."""

from __future__ import annotations

import numpy as np


def apply_auto_gain(
    audio: np.ndarray,
    target_rms: float = 0.12,
    max_gain: float = 8.0,
    noise_floor: float = 0.002,
) -> np.ndarray:
    """Raise quiet speech to a useful level while protecting against clipping.

    Near-silence is deliberately not boosted: amplifying it would turn
    microphone noise into a louder signal and make recognition worse.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return samples.copy()

    rms = float(np.sqrt(np.mean(np.square(samples))))
    if not np.isfinite(rms) or rms < noise_floor:
        return samples.copy()

    gain = min(max_gain, max(1.0, target_rms / rms))
    amplified = samples * gain
    peak = float(np.max(np.abs(amplified)))
    if peak > 0.98:
        amplified *= 0.98 / peak
    return amplified.astype(np.float32, copy=False)
