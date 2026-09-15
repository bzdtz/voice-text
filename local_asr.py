"""Local streaming ASR via sherpa-onnx Zipformer (Chinese + English)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LOCAL_STREAMING_MODEL_DIRS = (
    ROOT
    / "sherpa-onnx-probe"
    / "models"
    / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    ROOT / "AI_Models" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
)
PUNCTUATION_MODEL_DIRS = (
    ROOT
    / "sherpa-onnx-probe"
    / "models"
    / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
    ROOT
    / "AI_Models"
    / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
)


def resolve_local_model_dir() -> Path:
    for path in LOCAL_STREAMING_MODEL_DIRS:
        if (path / "tokens.txt").exists() and (
            path / "encoder-epoch-99-avg-1.int8.onnx"
        ).exists():
            return path
    raise FileNotFoundError(
        "未找到双语流式模型目录 sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    )


def resolve_punctuation_model() -> Path | None:
    for path in PUNCTUATION_MODEL_DIRS:
        candidate = path / "model.int8.onnx"
        if candidate.exists():
            return candidate
        candidate = path / "model.onnx"
        if candidate.exists():
            return candidate
    return None


def hotwords_path() -> Path:
    import os

    app_data = Path(os.environ.get("APPDATA", Path.home() / ".config"))
    return app_data / "VoiceTextInput" / "hotwords.txt"


def read_hotwords_text() -> str:
    path = hotwords_path()
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return ""


def write_hotwords_text(text: str) -> None:
    path = hotwords_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n").strip() + ("\n" if text.strip() else ""), encoding="utf-8")


def _mono_float32(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio.reshape(-1))


class OfflinePunctuator:
    """Optional CT-Transformer punctuation for final ASR text."""

    def __init__(self, model_path: Path | None = None, num_threads: int = 1) -> None:
        import sherpa_onnx

        path = Path(model_path) if model_path else resolve_punctuation_model()
        if path is None:
            raise FileNotFoundError("未找到标点模型 model.int8.onnx")
        config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
                ct_transformer=str(path),
                num_threads=num_threads,
                provider="cpu",
                debug=False,
            )
        )
        self._punct = sherpa_onnx.OfflinePunctuation(config)
        self.model_path = path

    def add_punctuation(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned
        return self._punct.add_punctuation(cleaned).strip()


class StreamingSession:
    """One push-to-talk utterance over a shared OnlineRecognizer."""

    def __init__(self, recognizer, punctuator: OfflinePunctuator | None = None) -> None:
        self._recognizer = recognizer
        self._punctuator = punctuator
        self._stream = recognizer.create_stream()
        self._lock = threading.Lock()

    def accept(self, samples: np.ndarray) -> str:
        audio = _mono_float32(samples)
        if audio.size == 0:
            with self._lock:
                return self._recognizer.get_result(self._stream)
        with self._lock:
            self._stream.accept_waveform(16000, audio)
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            return self._recognizer.get_result(self._stream)

    def finish(self, tail_seconds: float = 0.6) -> str:
        with self._lock:
            if tail_seconds > 0:
                tail = np.zeros(int(16000 * tail_seconds), dtype=np.float32)
                self._stream.accept_waveform(16000, tail)
            self._stream.input_finished()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            text = self._recognizer.get_result(self._stream).strip()
        if text and self._punctuator is not None:
            try:
                text = self._punctuator.add_punctuation(text)
            except Exception:
                # Keep raw ASR text if punctuation fails.
                pass
        return text


class StreamingZipformerASR:
    """Load once; create a fresh StreamingSession per recording."""

    def __init__(
        self,
        model_dir: Path | None = None,
        hotwords_file: Path | None = None,
        num_threads: int = 2,
        enable_punctuation: bool = True,
    ) -> None:
        import sherpa_onnx

        self.model_dir = Path(model_dir) if model_dir else resolve_local_model_dir()
        self.hotwords_file = Path(hotwords_file) if hotwords_file else hotwords_path()
        tokens = self.model_dir / "tokens.txt"
        encoder = self.model_dir / "encoder-epoch-99-avg-1.int8.onnx"
        decoder = self.model_dir / "decoder-epoch-99-avg-1.onnx"
        joiner = self.model_dir / "joiner-epoch-99-avg-1.int8.onnx"
        bpe_vocab = self.model_dir / "bpe.vocab"
        for required in (tokens, encoder, decoder, joiner):
            if not required.exists():
                raise FileNotFoundError(f"模型文件缺失：{required.name}")

        use_hotwords = (
            self.hotwords_file.exists()
            and self.hotwords_file.stat().st_size > 0
            and bpe_vocab.exists()
        )
        kwargs: dict = {
            "tokens": str(tokens),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "joiner": str(joiner),
            "num_threads": num_threads,
            "sample_rate": 16000,
            "feature_dim": 80,
            "provider": "cpu",
            "model_type": "zipformer",
        }
        if use_hotwords:
            kwargs.update(
                decoding_method="modified_beam_search",
                modeling_unit="cjkchar+bpe",
                bpe_vocab=str(bpe_vocab),
                hotwords_file=str(self.hotwords_file),
                hotwords_score=1.5,
            )
        else:
            kwargs["decoding_method"] = "greedy_search"

        self.uses_hotwords = use_hotwords
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)

        self.punctuator: OfflinePunctuator | None = None
        self.uses_punctuation = False
        if enable_punctuation and resolve_punctuation_model() is not None:
            self.punctuator = OfflinePunctuator(num_threads=1)
            self.uses_punctuation = True

    def create_session(self) -> StreamingSession:
        return StreamingSession(self.recognizer, self.punctuator)
