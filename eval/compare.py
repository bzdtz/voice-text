# -*- coding: utf-8 -*-
"""Run all candidate streaming ASR models on the same zh-en mixed audio and compare.

Metrics per (model, wav):
  - RTF (whole-utterance decode wall time / audio duration), num_threads=1
  - latency to first non-empty partial (streaming, 0.1 s chunks)
  - output text + English casing analysis
  - char similarity vs the TTS ground truth (difflib ratio)
Also reports the on-disk size of exactly the files each model loads.
"""
import difflib
import os
import re
import sys
import time

import numpy as np
import scipy.io.wavfile as wavfile
import sherpa_onnx

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)
AUDIO_DIR = os.path.join(BASE, "audio")

# Models are large (163-531 MB each) and are intentionally NOT committed to this repo.
# They are downloaded separately -- see README.md. Search these locations in order so
# the harness works whether the models live next to the desktop app or standalone.
# Override everything with VOICE_TEXT_EVAL_MODELS=/abs/path/to/models.
MODEL_SEARCH_PATHS = [
    os.environ.get("VOICE_TEXT_EVAL_MODELS", ""),
    os.path.join(BASE, "models"),
    os.path.join(PROJECT_ROOT, "sherpa-onnx-probe", "models"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "Lip_vioce", "models"),
]


def resolve_models_dir():
    for candidate in MODEL_SEARCH_PATHS:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "未找到 ASR 模型目录。请把模型解压到 sherpa-onnx-probe\\models\\，"
        "或设置环境变量 VOICE_TEXT_EVAL_MODELS 指向模型目录。"
    )


MODELS_DIR = resolve_models_dir()

# --------------------------------------------------------------------- helpers

def load_wav(path):
    """Return (float32 samples in [-1,1], sample_rate, duration_seconds)."""
    sr, pcm = wavfile.read(path)
    if pcm.dtype == np.int16:
        samples = pcm.astype(np.float32) / 32768.0
    elif pcm.dtype == np.float32:
        samples = pcm.astype(np.float32)
    else:
        samples = pcm.astype(np.float32)
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples.astype(np.float32), sr, len(samples) / float(sr)


def build_recognizer(cfg):
    tokens = os.path.join(MODELS_DIR, cfg["dir"], cfg["tokens"])
    encoder = os.path.join(MODELS_DIR, cfg["dir"], cfg["encoder"])
    decoder = os.path.join(MODELS_DIR, cfg["dir"], cfg["decoder"])
    common = dict(
        tokens=tokens, encoder=encoder, decoder=decoder,
        num_threads=1,
        decoding_method="greedy_search",
        # endpoint detection can finalize mid-sentence at natural pauses,
        # truncating output unfairly -> keep it off for clean whole-audio metrics
        enable_endpoint_detection=False,
    )
    if cfg["kind"] == "transducer":
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            **common, joiner=os.path.join(MODELS_DIR, cfg["dir"], cfg["joiner"]))
    return sherpa_onnx.OnlineRecognizer.from_paraformer(**common)


def run_whole(rec, samples):
    """Feed all audio + tail, decode to end, return (final_text, rtf)."""
    stream = rec.create_stream()
    stream.accept_waveform(16000, samples)
    tail = np.zeros(int(0.3 * 16000), dtype=np.float32)
    stream.accept_waveform(16000, tail)
    stream.input_finished()
    t0 = time.perf_counter()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    t1 = time.perf_counter()
    result = rec.get_result_all(stream)
    text = result.text if hasattr(result, "text") else str(result)
    return text, (t1 - t0)


def run_streaming(rec, samples):
    """Feed 0.1 s chunks; measure wall time AND audio position of first partial."""
    chunk = int(0.1 * 16000)
    stream = rec.create_stream()
    t_start = time.perf_counter()
    first_partial_at = None      # wall-clock seconds after start
    first_partial_audio = None   # seconds of audio fed when first partial appears
    first_partial_text = ""
    fed = 0.0
    for i in range(0, len(samples), chunk):
        stream.accept_waveform(16000, samples[i:i + chunk])
        fed += len(samples[i:i + chunk]) / 16000.0
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        if first_partial_at is None:
            partial = rec.get_result(stream)
            if partial.strip():
                first_partial_at = time.perf_counter() - t_start
                first_partial_audio = fed
                first_partial_text = partial
    tail = np.zeros(int(0.3 * 16000), dtype=np.float32)
    stream.accept_waveform(16000, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    result = rec.get_result_all(stream)
    final = result.text if hasattr(result, "text") else str(result)
    return first_partial_at, first_partial_audio, first_partial_text, final


def normalize_for_sim(s):
    """lowercase, strip punctuation/spaces for char-level comparison."""
    s = s.lower()
    s = re.sub(r"[^\w一-鿿]+", "", s)
    return s


def analyze(text):
    """Casing analysis: count ALL-CAPS english word runs."""
    upper_words = re.findall(r"[A-Z]{2,}", text)      # e.g. MONDAY
    upper_chars = sum(len(w) for w in upper_words)
    return upper_words[:8], upper_chars


# --------------------------------------------------------------------- models

MODELS = [
    dict(
        name="baseline_zipformer_2023-02-20",
        kind="transducer",
        dir="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        tokens="tokens.txt",
        encoder="encoder-epoch-99-avg-1.int8.onnx",
        decoder="decoder-epoch-99-avg-1.int8.onnx",
        joiner="joiner-epoch-99-avg-1.int8.onnx",
    ),
    dict(
        name="streaming_paraformer_bilingual",
        kind="paraformer",
        dir="paraformer-bilingual-int8",
        tokens="tokens.txt",
        encoder="encoder.int8.onnx",
        decoder="decoder.int8.onnx",
    ),
    dict(
        name="x-asr_punct_int8_480ms",
        kind="transducer",
        dir="sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05",
        tokens="tokens.txt",
        encoder="encoder.int8.onnx",
        decoder="decoder.onnx",
        joiner="joiner.int8.onnx",
    ),
]


def deployment_size(cfg):
    total = 0
    for key in ("tokens", "encoder", "decoder", "joiner"):
        if key in cfg:
            total += os.path.getsize(os.path.join(MODELS_DIR, cfg["dir"], cfg[key]))
    return total


def model_is_available(cfg):
    """True when every file the model actually loads exists on disk."""
    for key in ("tokens", "encoder", "decoder", "joiner"):
        if key in cfg and not os.path.exists(
                os.path.join(MODELS_DIR, cfg["dir"], cfg[key])):
            return False
    return True


def available_models():
    """Candidates whose weights are present. Missing models are skipped, not fatal,
    so the harness stays runnable before all ~900 MB have been downloaded."""
    return [c for c in MODELS if model_is_available(c)]


def read_manifest():
    rows = []
    with open(os.path.join(AUDIO_DIR, "manifest.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], parts[2], float(parts[1])))  # name, text, dur
    return rows


# --------------------------------------------------------------------- main

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    manifest = read_manifest()
    wavs = [os.path.join(AUDIO_DIR, n + ".wav") for n, _, _ in manifest]

    runnable = available_models()
    skipped = [c for c in MODELS if c not in runnable]
    print(f"models dir: {MODELS_DIR}")
    for c in skipped:
        print(f"  [skip] {c['name']}：模型文件缺失（{c['dir']}）")
    if not runnable:
        sys.exit("没有任何候选模型可用，退出。")
    if skipped:
        print(f"  -> 本次只评测 {len(runnable)}/{len(MODELS)} 个模型，"
              "对比结论会弱于全量对比，请按 docs/ASR模型选型评测.md 补齐模型后重跑")

    # pre-load audio once
    audio = []
    for w in wavs:
        s, sr, dur = load_wav(w)
        audio.append((s, dur))

    summary = []  # per (model, sentence): (rtf, first_latency, sim, upper_chars, text)
    all_transcripts = []

    for cfg in runnable:
        print(f"\n==== loading {cfg['name']} ====")
        t0 = time.perf_counter()
        rec = build_recognizer(cfg)
        print(f"  loaded in {time.perf_counter()-t0:.1f}s, deploy size "
              f"{deployment_size(cfg)/1e6:.1f} MB")
        # warmup
        wu = rec.create_stream()
        wu.accept_waveform(16000, np.zeros(int(0.2 * 16000), dtype=np.float32))
        wu.input_finished()
        while rec.is_ready(wu):
            rec.decode_stream(wu)

        for (s, dur), (name, truth, _) in zip(audio, manifest):
            text, rtf = run_whole(rec, s)
            first_at, first_audio, first_text, final_text = run_streaming(rec, s)
            sim = difflib.SequenceMatcher(
                None, normalize_for_sim(truth), normalize_for_sim(text)).ratio()
            upper_words, upper_chars = analyze(text)
            summary.append(dict(model=cfg["name"], s=name, truth=truth, text=text,
                                rtf=rtf, first_at=first_at, first_audio=first_audio,
                                sim=sim, upper_chars=upper_chars, upper_words=upper_words))
            print(f"  [{name}] rtf={rtf:.3f} first_audio={first_audio:.2f}s "
                  f"sim={sim:.2f} UPPER={upper_chars}")

    # ---- console summary tables
    models = [c["name"] for c in runnable]
    rows = summary
    print("\n\n========== SUMMARY ==========")
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        avg_rtf = np.mean([r["rtf"] for r in mrows])
        avg_first = np.mean([r["first_audio"] for r in mrows])
        avg_sim = np.mean([r["sim"] for r in mrows])
        total_upper = sum(r["upper_chars"] for r in mrows)
        size = deployment_size(next(c for c in MODELS if c["name"] == m)) / 1e6
        print(f"\n{m}\n"
              f"  RTF(1线程)均值={avg_rtf:.3f}  首字出现音频位置均值={avg_first:.2f}s\n"
              f"  与ground-truth相似度均值={avg_sim:.2f}  全大写英文字符总数={total_upper}\n"
              f"  部署体积(加载文件)={size:.1f} MB")

    # ---- write report.md
    write_report(models, rows, manifest)

    # quick verdict heuristics
    print("\n==== verdict hints (auto, for manual review) ====")
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        print(f"{m}: avg_sim={np.mean([r['sim'] for r in mrows]):.2f}, "
              f"UPPER_total={sum(r['upper_chars'] for r in mrows)}")


def write_report(models, rows, manifest):
    lines = ["# ASR 三候选实测对比\n"]
    lines.append("同一批 6 段中英混合音频（edge-tts 合成，16k mono），"
                 "sherpa-onnx 1.13.4，greedy_search，num_threads=1。\n")
    lines.append("| 模型 | 类型 | 部署体积 | 平均 RTF(1线程) | 首字出现音频位置 | 平均相似度 | 全大写英文字符 |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in models:
        mrows = [r for r in rows if r["model"] == m]
        size = deployment_size(next(c for c in MODELS if c["name"] == m)) / 1e6
        lines.append(
            f"| {m} | {next(c for c in MODELS if c['name']==m)['kind']} | "
            f"{size:.1f}MB | {np.mean([r['rtf'] for r in mrows]):.3f} | "
            f"{np.mean([r['first_audio'] for r in mrows]):.2f}s | "
            f"{np.mean([r['sim'] for r in mrows]):.2f} | "
            f"{sum(r['upper_chars'] for r in mrows)} |")
    lines.append("")
    lines.append("## 逐句输出\n")
    for m in models:
        lines.append(f"### {m}\n")
        for r in [x for x in rows if x["model"] == m]:
            lines.append(f"- **{r['s']}** (rtf {r['rtf']:.3f}, first_audio {r['first_audio']:.2f}s, "
                         f"sim {r['sim']:.2f})\n  - 期望: {r['truth']}\n  - 实际: **{r['text']}**")
        lines.append("")
    with open(os.path.join(BASE, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nreport written -> {os.path.join(BASE, 'report.md')}")


if __name__ == "__main__":
    main()
