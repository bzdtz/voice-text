"""Endpoint detection: is it armed, and would turning it on do anything?

Push-to-talk has exactly one terminator -- the user letting go of the hotkey.
Endpoint detection adds a second one based on trailing silence, which is what
cuts speech into segments for a server VAD. Two questions are worth answering
before shipping an input method:

1. Is it currently armed? ``local_asr.py`` pins
   ``enable_endpoint_detection=False``; this confirms the app's real behaviour
   by comparing the kwarg-omitted case against the explicit ones.
2. Would arming it help or hurt? The utterance below is three real eval clips
   joined by 3.0s of silence, i.e. a user pausing mid-thought. Ground truth is
   the concatenation of the three prompts.

Findings on this build (sherpa-onnx 1.13.4, x-asr, 2026-10-07):

- The endpoint *does* fire when armed: after ~1.6s of trailing silence, on the
  first pause only.
- ``rule1_min_trailing_silence`` has no measurable effect -- 0.5 / 2.4 / 10.0
  all fire at the same point. The rule that actually fires is rule2.
- Yet the final text is byte-identical with the flag on and off, because
  ``StreamingSession`` never reads ``is_endpoint()``. The flag is computed and
  discarded, so arming it would add compute without changing one character.

``rule1``/``rule2`` sweeps are in probe_rules2.py during development; the
reproducible part is the three-way comparison below.

    python eval\\bench_endpoint.py
"""

from __future__ import annotations

import os
import sys
import time
import wave
from difflib import SequenceMatcher

import numpy as np

# Importable both as `python eval\bench_endpoint.py` and as a module.
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(_EVAL_DIR), _EVAL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from compare import AUDIO_DIR, normalize_for_sim, read_manifest  # noqa: E402
from local_asr import StreamingSession, resolve_local_model_config  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_S = 0.08                 # STREAM_POLL_MS in main.py
CHUNK = int(CHUNK_S * SAMPLE_RATE)
PAUSE_S = 3.0                  # well above any rule threshold in this build
PARTS = ("s05_chinese", "s04_switch", "s06_english")


def load(name: str) -> np.ndarray:
    with wave.open(f"{AUDIO_DIR}\\{name}.wav") as wav:
        return (np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
                .astype(np.float32) / 32768.0)


def make_utterance():
    """Three clips joined by PAUSE_S of silence, plus where each clip sits."""
    silence = np.zeros(int(PAUSE_S * SAMPLE_RATE), dtype=np.float32)
    pieces, spans, offset = [], [], 0
    for i, name in enumerate(PARTS):
        if i:
            pieces.append(silence)
            offset += silence.size
        seg = load(name)
        pieces.append(seg)
        spans.append((name, offset / SAMPLE_RATE, (offset + seg.size) / SAMPLE_RATE))
        offset += seg.size
    return np.concatenate(pieces), spans


def build_session(enable_endpoint, pass_flag: bool) -> StreamingSession:
    """Same kwargs as local_asr.py, toggling one thing.

    pass_flag=False omits the kwarg entirely -- that is literally what
    local_asr.py did before it pinned the value, so this row reports the
    app's historical default rather than an assumption about it.
    """
    import sherpa_onnx

    config, model_dir = resolve_local_model_config()
    kwargs = dict(
        tokens=str(model_dir / "tokens.txt"),
        encoder=str(model_dir / config.encoder),
        decoder=str(model_dir / config.decoder),
        joiner=str(model_dir / config.joiner),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        provider="cpu",
        decoding_method="greedy_search",
    )
    if pass_flag:
        kwargs["enable_endpoint_detection"] = enable_endpoint
    return StreamingSession(sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs))


def run(label: str, enable_endpoint: bool, pass_flag: bool) -> dict:
    audio, spans = make_utterance()
    manifest = dict((name, truth) for name, truth, _d in read_manifest())
    truth = " ".join(manifest[p] for p in PARTS)
    session = build_session(enable_endpoint, pass_flag)

    endpoint_at = None
    declared = False
    at_segment_end = {}
    for start in range(0, len(audio), CHUNK):
        session.accept(audio[start:start + CHUNK])
        stream = session._stream
        if session._recognizer.is_endpoint(stream):
            declared = True
            if endpoint_at is None:
                endpoint_at = (start + CHUNK) / SAMPLE_RATE
        partial = session._recognizer.get_result(stream)
        # Partial at the last moment of each clip: the number that would
        # expose truncation if an endpoint ever cut the stream.
        for name, _s0, s1 in spans:
            if s1 - CHUNK_S <= start / SAMPLE_RATE < s1:
                at_segment_end[name] = partial

    t0 = time.perf_counter()
    final = session.finish(tail_seconds=0.3)
    finalize_ms = (time.perf_counter() - t0) * 1000

    return dict(label=label, final=final,
                sim=SequenceMatcher(None, normalize_for_sim(final),
                                    normalize_for_sim(truth)).ratio(),
                endpoint_at=endpoint_at, declared=declared,
                finalize_ms=finalize_ms, spans=spans,
                at_segment_end=at_segment_end, dur=len(audio) / SAMPLE_RATE)


def main() -> None:
    audio, spans = make_utterance()
    manifest = dict((name, truth) for name, truth, _d in read_manifest())

    print(f"pause inserted between clips: {PAUSE_S}s")
    for name, s0, s1 in spans:
        print(f"  {name:16s} {s0:5.2f}s - {s1:5.2f}s")
    print(f"  total audio: {len(audio) / SAMPLE_RATE:.2f}s")

    cases = (
        ("kwarg omitted  (app default before pinning)", False, False),
        ("enable_endpoint_detection=False", False, True),
        ("enable_endpoint_detection=True", True, True),
    )
    runs = [run(label, flag, pf) for label, flag, pf in cases]

    print()
    print(f"{'mode':44s} {'sim':>6s} {'endpt@':>7s} {'final':>6s}  declared")
    print("-" * 76)
    for r in runs:
        where = f"{r['endpoint_at']:.2f}s" if r["endpoint_at"] else "-"
        print(f"{r['label']:44s} {r['sim']:6.3f} {where:>7s} "
              f"{r['finalize_ms']:5.0f}ms  {r['declared']}")
        print(f"   got: {r['final']}\n")

    print("omitted-kwarg == explicit False :", runs[0]["final"] == runs[1]["final"])
    print("explicit True   == explicit False:",
          runs[2]["final"] == runs[1]["final"],
          "<-- endpoint fired but changed no character")

    print("\nground truth:")
    print("  " + " ".join(manifest[p] for p in PARTS))

    print("\npartial visible to the UI at the last moment of each clip "
          "(endpoint ON). The partial is cumulative and lags the audio by one "
          "decode step, so a value under 1.0 on the first clip is decode lag, "
          "not truncation:")
    idx = 0
    for name, _s0, _s1 in spans:
        got_norm = normalize_for_sim(runs[2]["at_segment_end"].get(name) or "")
        want = "".join(normalize_for_sim(manifest[p]) for p in PARTS[:idx + 1])
        sim = SequenceMatcher(None, got_norm, want).ratio()
        print(f"  sim={sim:5.3f}  vs everything spoken so far   ({name})")
        idx += 1


if __name__ == "__main__":
    main()
