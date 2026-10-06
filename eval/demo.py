# -*- coding: utf-8 -*-
"""Interactive hands-on demo: speak or play audio, watch live gray subtitles.

Usage:
  python demo.py devices                      # list input devices
  python demo.py mic-test 3                   # check mic level (speak during it)
  python demo.py mic-test 3 --device 16       # try a specific device
  python demo.py mic 15 --device 16           # live subtitle from mic
  python demo.py mic 15                       # mic, default device
  python demo.py file audio/s01_mixed_official.wav
  python demo.py file any.wav --fast
  python demo.py mic 10 --model baseline      # try another candidate model

Models: baseline_zipformer_2023-02-20 / streaming_paraformer_bilingual / x-asr_punct_int8_480ms
"""
import argparse
import os
import sys
import time

import numpy as np

from compare import MODELS, build_recognizer

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
CHUNK = int(0.1 * SR)


def pick_model(name):
    cfg = next((c for c in MODELS if c["name"] == name), None)
    if cfg is None:
        sys.exit(f"unknown model {name!r}; choices: {[c['name'] for c in MODELS]}")
    return cfg


def level_bar(rms):
    n = int(min(1.0, rms * 20) * 20)
    return "[" + "█" * n + " " * (20 - n) + "]"


def show_partial(text, rms=None):
    line = text.strip().replace("\n", " ")
    meter = level_bar(rms) if rms is not None else ""
    out = f"\r{meter} {line[:52]}"
    sys.stdout.write(out + " " * max(0, 60 - len(out)) + "\r")
    sys.stdout.flush()


def finish(rec, stream, show_final=True):
    tail = np.zeros(int(0.3 * SR), dtype=np.float32)
    stream.accept_waveform(SR, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    res = rec.get_result_all(stream)
    text = res.text if hasattr(res, "text") else str(res)
    if show_final:
        sys.stdout.write("\n" + "=" * 66 + "\nFINAL: " + text.strip() + "\n")
    return text


def open_input(device):
    import sounddevice as sd
    return sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                          blocksize=CHUNK, device=device)


def run_mic(cfg, max_duration, silence_secs, device):
    rec = build_recognizer(cfg)
    stream = rec.create_stream()
    print(f"\n 用 {cfg['name']} —— 开始说话吧（最多 {max_duration}s，停 {silence_secs}s 自动结束）")
    print(f" 输入设备: {device if device is not None else '默认'}   实时字幕:\n")
    silence = 0.0
    t0 = time.perf_counter()
    with open_input(device) as inp:
        while time.perf_counter() - t0 < max_duration:
            block, _ = inp.read(CHUNK)
            block = np.clip(block[:, 0], -1.0, 1.0).astype(np.float32)
            stream.accept_waveform(SR, block)
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            rms = float(np.sqrt(np.mean(block ** 2)))
            if rms < 0.01:
                silence += CHUNK / SR
            else:
                silence = 0.0
            show_partial(rec.get_result(stream), rms)
            if silence >= silence_secs:
                break
    finish(rec, stream)


def run_file(cfg, wav_path, fast):
    from scipy.io import wavfile

    rec = build_recognizer(cfg)
    stream = rec.create_stream()
    sr, pcm = wavfile.read(wav_path)
    samples = (pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16
               else pcm.astype(np.float32))
    if samples.ndim > 1:
        samples = samples[:, 0]
    dur = len(samples) / float(sr)
    print(f"\n 用 {cfg['name']} 转写 {os.path.basename(wav_path)}（{dur:.1f}s）\n")
    print(" 实时字幕:\n")
    t0 = time.perf_counter()
    for i in range(0, len(samples), CHUNK):
        stream.accept_waveform(SR, samples[i:i + CHUNK])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        show_partial(rec.get_result(stream))
        if not fast:
            target = (i + CHUNK) / SR
            time.sleep(max(0.0, target - (time.perf_counter() - t0)))
    finish(rec, stream)


def mic_test(seconds, device):
    print(f"\n 录制 {seconds}s 检验麦克风（请出声说话）…\n")
    levels = []
    with open_input(device) as inp:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            block, _ = inp.read(CHUNK)
            rms = float(np.sqrt(np.mean(block[:, 0] ** 2)))
            levels.append(rms)
            sys.stdout.write(f"\r level={rms:.4f}  peak={max(levels):.4f}   ")
            sys.stdout.flush()
    peak = max(levels)
    print(f"\n\n 最大电平: {peak:.4f}")
    if peak > 0.02:
        print(" ✓ 麦克风有声，可以用了")
    else:
        print(" ✗ 全是静音 —— 换设备试试: python demo.py mic-test 3 --device <编号>\n"
              "   列出设备: python demo.py devices")


def list_devices():
    import sounddevice as sd
    print("输入设备列表：")
    default = sd.default.device[0] if isinstance(sd.default.device, (tuple, list)) else None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <== 默认输入" if i == default else ""
            print(f"[{i}] {d['name']}  {int(d['default_samplerate'])}Hz, "
                  f"{d['max_input_channels']}ch{mark}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["mic", "file", "mic-test", "devices"])
    ap.add_argument("target", nargs="?", help="mic/mic-test: seconds; file: wav path")
    ap.add_argument("--model", default="x-asr_punct_int8_480ms")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--silence-secs", type=float, default=1.2)
    args = ap.parse_args()

    if args.mode == "devices":
        list_devices()
        return
    cfg = pick_model(args.model)
    if args.mode == "mic-test":
        mic_test(float(args.target or 3), args.device)
    elif args.mode == "mic":
        run_mic(cfg, float(args.target or 15), args.silence_secs, args.device)
    else:
        run_file(cfg, args.target, args.fast)
    print()


if __name__ == "__main__":
    main()
