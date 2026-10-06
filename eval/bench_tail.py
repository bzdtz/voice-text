"""How much silence does finish() need before input_finished()?

``StreamingSession.finish(tail_seconds=X)`` appends X seconds of silence before
declaring the input complete, so the decoder has room to finalize trailing
tokens. That value is a magic number in most codebases; this measures what it
should be by scoring final text at several lengths with the same similarity
metric as compare.py.

Finding on the current 6-utterance set (2026-10-06): 0.30s and 0.60s produce
byte-identical output at 0.9905 mean similarity, while 0.00s and 0.15s
truncate the last token or two. 0.3s is the minimum that works and is therefore
the default in local_asr.py.

    python eval\\bench_tail.py
"""

from __future__ import annotations

import os
import sys
import time
import wave
from difflib import SequenceMatcher

import numpy as np

# Importable both as `python eval\bench_tail.py` and as a module.
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(_EVAL_DIR), _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from compare import AUDIO_DIR, normalize_for_sim, read_manifest  # noqa: E402
from local_asr import StreamingZipformerASR  # noqa: E402

TAILS = (0.0, 0.15, 0.3, 0.6, 1.2)


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as wav:
        return (np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                .astype(np.float32) / 32768.0)


def main() -> None:
    manifest = read_manifest()
    asr = StreamingZipformerASR()
    print(f"model: {asr.model_dir.name}\n")

    texts: dict[tuple[str, float], str] = {}
    sims: dict[float, list[float]] = {}
    times: dict[float, list[float]] = {}
    for name, truth, _dur in manifest:
        audio = load_wav(f"{AUDIO_DIR}\\{name}.wav")
        for tail in TAILS:
            session = asr.create_session()
            # Feed the whole utterance at once: the tail is the only variable,
            # so the streaming cadence would just add noise.
            session.accept(np.ascontiguousarray(audio))
            start = time.perf_counter()
            text = session.finish(tail_seconds=tail)
            elapsed = time.perf_counter() - start

            texts[name, tail] = text
            sims.setdefault(tail, []).append(
                SequenceMatcher(None,
                                normalize_for_sim(text),
                                normalize_for_sim(truth)).ratio()
            )
            times.setdefault(tail, []).append(elapsed)

    print(f"{'tail':>7s} | {'mean sim':>9s} | {'mean time':>10s} | worst case")
    print("-" * 68)
    for tail in TAILS:
        scored = sims[tail]
        worst = max(range(len(scored)), key=lambda i: -scored[i])
        print(f"{tail:6.2f}s | {np.mean(scored):9.4f} | "
              f"{np.mean(times[tail]) * 1000:8.1f} ms | "
              f"{manifest[worst][0][:16]} {scored[worst]:.3f}")

    print("\nOutputs where tail length changes the result:")
    for name, truth, _dur in manifest:
        by_tail = {tail: texts[name, tail] for tail in TAILS}
        if len(set(by_tail.values())) > 1:
            print(f"\n[{name}]\n  truth: {truth}")
            for tail in TAILS:
                print(f"  tail {tail:4.2f}s -> {by_tail[tail]}")


if __name__ == "__main__":
    main()
