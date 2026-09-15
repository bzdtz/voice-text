"""SiliconFlow transcription client used by the Tk desktop application."""

from __future__ import annotations

import io
import threading
import wave
from typing import Callable, Optional

import numpy as np
import requests

SILICONFLOW_TRANSCRIPTIONS_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
SILICONFLOW_MODEL = "FunAudioLLM/SenseVoiceSmall"


class CloudASRError(RuntimeError):
    """An error that is safe to show to the user."""


class RequestCancelled(RuntimeError):
    """The user cancelled the pending transcription."""


def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float32 sounddevice buffer as an in-memory WAV file."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype("<i2")

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return output.getvalue()


def _response_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        for key in ("message", "detail"):
            if payload.get(key):
                return str(payload[key])
    return f"HTTP {response.status_code}"


class SiliconFlowASR:
    """One-shot, non-streaming SiliconFlow ASR client.

    ``cancel_event`` makes the request result cancellable.  Requests has no
    thread-safe equivalent of browser AbortController, so the caller also uses
    a monotonically increasing request id and discards a cancelled response.
    """

    def __init__(
        self,
        api_key: str,
        post: Callable = requests.post,
        timeout: tuple[float, float] = (10, 90),
    ) -> None:
        self.api_key = api_key.strip()
        self._post = post
        self._timeout = timeout

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        if not self.api_key:
            raise CloudASRError("请先在设置中填写 SiliconFlow API Key。")
        if cancel_event and cancel_event.is_set():
            raise RequestCancelled()

        wav_data = audio_to_wav(audio, sample_rate)
        files = {"file": ("recording.wav", wav_data, "audio/wav")}
        response = self._post(
            SILICONFLOW_TRANSCRIPTIONS_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={"model": SILICONFLOW_MODEL},
            files=files,
            timeout=self._timeout,
        )

        if cancel_event and cancel_event.is_set():
            raise RequestCancelled()
        if not response.ok:
            raise CloudASRError(_response_message(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise CloudASRError("识别服务返回的不是有效 JSON。") from exc
        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise CloudASRError("识别服务未返回有效的 text 字段。")
        return text.strip()
