"""Local streaming ASR over sherpa-onnx (Chinese + English).

Model selection is driven by the model's declared capabilities rather than a
hardcoded path, so swapping in a better model is a data change instead of a
code change. Candidates are tried in priority order and the first one whose
weights are actually on disk wins.

Punctuation and hotword boosting are attached only when the selected model
supports them:

- ``builtin_punctuation`` - the model emits punctuation and casing itself.
  Running a second punctuation pass on top would double-punctuate, so the
  CT-Transformer stage is skipped.
- ``bpe_vocab`` - required by the hotword-boosted ``modified_beam_search``
  decoder. Most community models do not ship one, so hotwords fall back to
  plain ``greedy_search`` instead of failing.

Model provenance: the preferred model is X-ASR, Apache-2.0, from
https://github.com/Gilgamesh-J/X-ASR .
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent

# Where models may live. The first root is the one this repo documents.
MODEL_SEARCH_ROOTS = (
    ROOT / "sherpa-onnx-probe" / "models",
    ROOT / "AI_Models",
)

PUNCTUATION_MODEL_DIRS = (
    ROOT
    / "sherpa-onnx-probe"
    / "models"
    / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
    ROOT / "AI_Models" / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
)


@dataclass(frozen=True)
class ModelConfig:
    """One candidate streaming ASR model and what it can do."""

    name: str
    dir_name: str
    encoder: str
    decoder: str
    joiner: str
    builtin_punctuation: bool = False
    bpe_vocab: str = "bpe.vocab"

    def files_ok(self, model_dir: Path) -> bool:
        """True when every weight file this model actually loads is present."""
        required = (
            "tokens.txt",
            self.encoder,
            self.decoder,
            self.joiner,
        )
        if self.bpe_vocab:
            required = required + (self.bpe_vocab,)
        return all((model_dir / name).exists() for name in required)


# Priority order = the ranking in docs/ASR模型选型评测.md.
# Mean char similarity on the 6-utterance zh-en set: 0.99 vs 0.72.
CANDIDATE_MODELS = (
    ModelConfig(
        name="x-asr-punct-int8-480ms",
        dir_name=(
            "sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-"
            "zh-en-punct-int8-2026-06-05"
        ),
        encoder="encoder.int8.onnx",
        decoder="decoder.onnx",
        joiner="joiner.int8.onnx",
        builtin_punctuation=True,
        # X-ASR ships bpe.model but no bpe.vocab, so the hotword decoder
        # cannot be used with it. Reported as disabled rather than failing.
        bpe_vocab="",
    ),
    ModelConfig(
        name="streaming-zipformer-bilingual-2023-02-20",
        dir_name="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        encoder="encoder-epoch-99-avg-1.int8.onnx",
        decoder="decoder-epoch-99-avg-1.onnx",
        joiner="joiner-epoch-99-avg-1.int8.onnx",
    ),
)

# Kept for callers and tests that only need the search paths.
LOCAL_STREAMING_MODEL_DIRS = tuple(
    root / config.dir_name for root in MODEL_SEARCH_ROOTS for config in CANDIDATE_MODELS
)


def resolve_local_model_config() -> tuple[ModelConfig, Path]:
    """Pick the highest-priority candidate whose weights are on disk."""
    missing: list[str] = []
    for root in MODEL_SEARCH_ROOTS:
        for config in CANDIDATE_MODELS:
            model_dir = root / config.dir_name
            if config.files_ok(model_dir):
                return config, model_dir
            missing.append(model_dir.as_posix())
    raise FileNotFoundError(
        "未找到可用的本地流式 ASR 模型。按优先级把下面任一个目录解压到 "
        "sherpa-onnx-probe\\models\\ 下即可（目录名保持不变）：\n"
        + "\n".join(f"  - {c.dir_name}" for c in CANDIDATE_MODELS)
        + "\n已尝试的位置：\n"
        + "\n".join(f"  - {p}" for p in missing)
    )


def resolve_local_model_dir() -> Path:
    _, model_dir = resolve_local_model_config()
    return model_dir


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


def write_hotwords_text(text: str) -> str:
    path = hotwords_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n").strip() + ("\n" if text.strip() else ""), encoding="utf-8")


def _mono_float32(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio.reshape(-1))


class OfflinePunctuator:
    """CT-Transformer punctuation stage for models without built-in punctuation."""

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
    """Load the best available model once; create a fresh session per recording."""

    def __init__(
        self,
        model_dir: Path | None = None,
        hotwords_file: Path | None = None,
        num_threads: int = 2,
        enable_punctuation: bool = True,
    ) -> None:
        import sherpa_onnx

        if model_dir is None:
            self.config, self.model_dir = resolve_local_model_config()
        else:
            self.model_dir = Path(model_dir)
            self.config = next(
                (c for c in CANDIDATE_MODELS if self.model_dir.name == c.dir_name),
                ModelConfig(
                    name=self.model_dir.name,
                    dir_name=self.model_dir.name,
                    encoder="encoder.int8.onnx",
                    decoder="decoder.onnx",
                    joiner="joiner.int8.onnx",
                ),
            )

        tokens = self.model_dir / "tokens.txt"
        encoder = self.model_dir / self.config.encoder
        decoder = self.model_dir / self.config.decoder
        joiner = self.model_dir / self.config.joiner
        for required in (tokens, encoder, decoder, joiner):
            if not required.exists():
                raise FileNotFoundError(f"模型文件缺失：{required.name}")

        bpe_vocab = (
            self.model_dir / self.config.bpe_vocab if self.config.bpe_vocab else None
        )
        hotwords_file = Path(hotwords_file) if hotwords_file else hotwords_path()
        self.hotwords_file = hotwords_file

        # Every condition here is required; missing any one silently degrades to
        # greedy_search instead of failing, which would look like a working
        # hotword feature that just never fires.
        use_hotwords = bool(
            bpe_vocab is not None
            and bpe_vocab.exists()
            and hotwords_file.exists()
            and hotwords_file.stat().st_size > 0
            and hotwords_file.read_text(encoding="utf-8").strip()
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
        }
        # Do NOT set model_type="zipformer". Some community encoders (X-ASR
        # among them) omit the attention_dims graph metadata that path
        # requires, and forcing the architecture aborts inside onnxruntime
        # with no Python-visible exception. Omitting it makes sherpa-onnx
        # infer the architecture from the graph, which is also what the
        # benchmark in docs/ASR模型选型评测.md used.
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

        # Only attach the CT-Transformer stage when the model does not already
        # produce punctuation; stacking them double-punctuates the output.
        self.punctuator: Optional[OfflinePunctuator] = None
        if enable_punctuation and not self.config.builtin_punctuation:
            if resolve_punctuation_model() is not None:
                try:
                    self.punctuator = OfflinePunctuator(num_threads=1)
                except Exception:
                    self.punctuator = None

        # True when the final output is expected to carry punctuation, whether
        # the model emits it itself or the offline stage adds it.
        self.uses_punctuation = bool(self.config.builtin_punctuation or self.punctuator)

    def create_session(self) -> StreamingSession:
        return StreamingSession(self.recognizer, self.punctuator)
