# -*- coding: utf-8 -*-
"""Generate a fixed set of zh-en mixed test WAVs (16 kHz mono) via edge-tts.

Each sentence's prompt text IS the ground truth used later by compare.py.
The same audio files are fed to every model, so the comparison is apples-to-apples.
"""
import asyncio
import io
import os
import sys

import edge_tts
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import soundfile as sf

VOICE = "zh-CN-XiaoxiaoNeural"  # Chinese voice that also speaks English words

# (filename, prompt_text)
SENTENCES = [
    ("s01_mixed_official",
     "昨天是 monday today day is zero eight two the day after tomorrow 是星期三"),
    ("s02_digits_letters",
     "我住在 building three 号楼的 twelve o two 房间，wifi 密码是 a b c 一二三四五六"),
    ("s03_tech",
     "这个 api 的 latency 是 fifty milliseconds，支持 http two 和 websocket 协议"),
    ("s04_switch",
     "we need to 讨论一下 tomorrow 的 meeting agenda 和 budget"),
    ("s05_chinese",
     "今天我们讨论了新产品的发布计划，下周一进行内测"),
    ("s06_english",
     "The meeting was postponed to next tuesday afternoon at three thirty"),
]

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")


async def synth_one(name, text, wav_path):
    """edge-tts -> mp3 bytes -> 16k mono float wav."""
    comm = edge_tts.Communicate(text, VOICE)
    mp3_bytes = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            mp3_bytes += chunk["data"]
    if not mp3_bytes:
        raise RuntimeError(f"no audio from edge-tts for {name}")

    data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32", always_2d=True)
    data = data[:, 0]  # mono
    if sr != 16000:
        # rational resample: sr -> 16000
        from math import gcd
        d = gcd(sr, 16000)
        data = signal.resample_poly(data, 16000 // d, sr // d)
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    wavfile.write(wav_path, 16000, pcm)
    return len(data) / 16000.0  # duration in seconds


async def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    manifest = []
    for name, text in SENTENCES:
        wav_path = os.path.join(AUDIO_DIR, name + ".wav")
        dur = await synth_one(name, text, wav_path)
        manifest.append((name, text, dur))
        print(f"[{name}] {dur:5.2f}s  -> {os.path.basename(wav_path)}")
    with open(os.path.join(AUDIO_DIR, "manifest.txt"), "w", encoding="utf-8") as f:
        for name, text, dur in manifest:
            f.write(f"{name}\t{dur:.3f}\t{text}\n")
    print(f"done. {len(manifest)} wavs in {AUDIO_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
