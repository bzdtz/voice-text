"""Smoke tests for local streaming Zipformer wrapper."""

from __future__ import annotations

import unittest

import numpy as np

from local_asr import (
    LOCAL_STREAMING_MODEL_DIRS,
    OfflinePunctuator,
    StreamingZipformerASR,
    resolve_local_model_dir,
    resolve_punctuation_model,
)


class LocalAsrTests(unittest.TestCase):
    def test_resolve_model_dir(self) -> None:
        model_dir = resolve_local_model_dir()
        self.assertTrue(model_dir.exists())
        self.assertTrue((model_dir / "tokens.txt").exists())
        self.assertTrue(any(path.exists() for path in LOCAL_STREAMING_MODEL_DIRS))

    def test_load_and_decode_silence(self) -> None:
        asr = StreamingZipformerASR()
        self.assertTrue(asr.uses_punctuation)
        session = asr.create_session()
        partial = session.accept(np.zeros(3200, dtype=np.float32))
        self.assertIsInstance(partial, str)
        final = session.finish(tail_seconds=0.2)
        self.assertIsInstance(final, str)

    def test_punctuation_adds_marks(self) -> None:
        punct_path = resolve_punctuation_model()
        self.assertIsNotNone(punct_path)
        punctuator = OfflinePunctuator(punct_path)
        result = punctuator.add_punctuation("今天天气很好我们去公园玩吧")
        self.assertTrue(any(mark in result for mark in ("，", "。", "？", "！", ",", ".", "?", "!")))


if __name__ == "__main__":
    unittest.main()
