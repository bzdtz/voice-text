"""Smoke tests for the local streaming ASR wrapper.

These run against whatever model files are actually present, so they double as
a check that model resolution picked a loadable candidate.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from local_asr import (
    CANDIDATE_MODELS,
    LOCAL_STREAMING_MODEL_DIRS,
    OfflinePunctuator,
    StreamingZipformerASR,
    resolve_local_model_config,
    resolve_punctuation_model,
)

X_ASR_DIR_NAME = (
    "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-"
    "zh-en-punct-int8-2026-06-05"
)


class LocalAsrTests(unittest.TestCase):
    def test_resolve_model_dir(self) -> None:
        config, model_dir = resolve_local_model_config()
        self.assertTrue(model_dir.exists())
        self.assertTrue((model_dir / "tokens.txt").exists())
        self.assertTrue(any(path.exists() for path in LOCAL_STREAMING_MODEL_DIRS))
        # The resolved config must describe the directory we actually loaded.
        self.assertEqual(config.dir_name, model_dir.name)

    def test_preferred_model_declares_its_capabilities(self) -> None:
        preferred = next(c for c in CANDIDATE_MODELS if c.dir_name == X_ASR_DIR_NAME)
        self.assertTrue(
            preferred.builtin_punctuation,
            "The preferred model must declare built-in punctuation, otherwise "
            "the CT-Transformer stage would be stacked on top of it.",
        )
        self.assertFalse(
            bool(preferred.bpe_vocab),
            "X-ASR ships no bpe.vocab; the config must say so so that hotwords "
            "are reported as unavailable instead of silently degrading.",
        )

    def test_load_and_decode_silence(self) -> None:
        asr = StreamingZipformerASR()
        # True whether the model emits punctuation itself or an offline stage
        # adds it.
        self.assertTrue(asr.uses_punctuation)
        session = asr.create_session()
        partial = session.accept(np.zeros(3200, dtype=np.float32))
        self.assertIsInstance(partial, str)
        final = session.finish(tail_seconds=0.2)
        self.assertIsInstance(final, str)

    def test_punctuation_adds_marks(self) -> None:
        # The fallback model still needs this, so the stage must keep working
        # even when the selected model does not use it.
        punct_path = resolve_punctuation_model()
        self.assertIsNotNone(punct_path)
        punctuator = OfflinePunctuator(punct_path)
        result = punctuator.add_punctuation("今天天气很好我们去公园玩吧")
        self.assertTrue(
            any(mark in result for mark in ("，", "。", "？", "！", ",", ".", "?", "!"))
        )

    def test_endpoint_detection_is_pinned_off(self) -> None:
        # Push-to-talk has exactly one terminator: the user letting go of the
        # hotkey. Endpoint detection adds a second one based on trailing
        # silence, which would cut a mid-thought pause. The flag is pinned
        # rather than inherited from sherpa-onnx's default -- a default nobody
        # chose here is a default upstream can change.
        #
        # Behaviours do not distinguish the flag in this build (rule1 is
        # inert), so assert the kwarg. This fails the moment someone drops it.
        import sherpa_onnx

        with mock.patch.object(
            sherpa_onnx.OnlineRecognizer,
            "from_transducer",
            side_effect=FileNotFoundError("weights not loaded by this test"),
        ) as patched:
            with self.assertRaises(FileNotFoundError):
                StreamingZipformerASR()

        self.assertIs(
            patched.call_args.kwargs.get("enable_endpoint_detection"),
            False,
            "endpoint detection must be pinned off, not left to the default",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
