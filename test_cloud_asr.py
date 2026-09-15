import json
import threading
import unittest

import numpy as np

from cloud_asr import (
    CloudASRError,
    RequestCancelled,
    SILICONFLOW_MODEL,
    SILICONFLOW_TRANSCRIPTIONS_URL,
    SiliconFlowASR,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class SiliconFlowASRTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.zeros((1600, 1), dtype=np.float32)

    def test_posts_expected_url_headers_model_and_audio_file(self):
        captured = {}

        def post(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeResponse(payload={"text": "你好"})

        text = SiliconFlowASR("test-key", post=post).transcribe(self.audio, 16000)

        self.assertEqual(text, "你好")
        self.assertEqual(captured["args"][0], SILICONFLOW_TRANSCRIPTIONS_URL)
        self.assertEqual(captured["kwargs"]["headers"], {"Authorization": "Bearer test-key"})
        self.assertEqual(captured["kwargs"]["data"], {"model": SILICONFLOW_MODEL})
        file_name, wav_data, mime_type = captured["kwargs"]["files"]["file"]
        self.assertEqual(file_name, "recording.wav")
        self.assertTrue(wav_data.startswith(b"RIFF"))
        self.assertEqual(mime_type, "audio/wav")

    def test_parses_text_field(self):
        client = SiliconFlowASR("key", post=lambda *a, **k: FakeResponse(payload={"text": "  云端结果  "}))
        self.assertEqual(client.transcribe(self.audio, 16000), "云端结果")

    def test_missing_text_is_a_user_facing_error(self):
        client = SiliconFlowASR("key", post=lambda *a, **k: FakeResponse(payload={"id": "request"}))
        with self.assertRaisesRegex(CloudASRError, "text"):
            client.transcribe(self.audio, 16000)

    def test_service_error_message_is_preserved(self):
        client = SiliconFlowASR(
            "key",
            post=lambda *a, **k: FakeResponse(401, {"error": {"message": "API Key 无效"}}),
        )
        with self.assertRaisesRegex(CloudASRError, "API Key 无效"):
            client.transcribe(self.audio, 16000)

    def test_cancelled_request_does_not_return_a_result(self):
        cancel_event = threading.Event()

        def post(*args, **kwargs):
            cancel_event.set()
            return FakeResponse(payload={"text": "不应显示"})

        client = SiliconFlowASR("key", post=post)
        with self.assertRaises(RequestCancelled):
            client.transcribe(self.audio, 16000, cancel_event)


if __name__ == "__main__":
    unittest.main()
