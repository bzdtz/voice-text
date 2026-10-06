# -*- coding: utf-8 -*-
"""实时中英字幕 GUI —— sherpa-onnx 三候选模型随意切换、手动开始/结束。

运行:  python gui.py
依赖:  纯 tkinter（Python 自带），音频走 sounddevice，模型复用 compare.MODELS
"""
import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from compare import MODELS, MODELS_DIR, available_models

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
CHUNK = int(0.1 * SR)

MODEL_LABELS = {
    "x-asr_punct_int8_480ms": "X-ASR punct int8 480ms（推荐）",
    "streaming_paraformer_bilingual": "流式 Paraformer 双语 int8",
    "baseline_zipformer_2023-02-20": "流式 Zipformer 双语（基线）",
}


def build_rec(cfg, threads=4):
    """Build an OnlineRecognizer from a compare.MODELS dict (num_threads tunable)."""
    import sherpa_onnx
    common = dict(
        tokens=os.path.join(MODELS_DIR, cfg["dir"], cfg["tokens"]),
        encoder=os.path.join(MODELS_DIR, cfg["dir"], cfg["encoder"]),
        decoder=os.path.join(MODELS_DIR, cfg["dir"], cfg["decoder"]),
        num_threads=threads,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )
    if cfg["kind"] == "transducer":
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            **common, joiner=os.path.join(MODELS_DIR, cfg["dir"], cfg["joiner"]))
    return sherpa_onnx.OnlineRecognizer.from_paraformer(**common)


def list_input_devices():
    """Return ([(index, name), ...], default_input_index_or_None)."""
    import sounddevice as sd
    default = None
    try:
        default = sd.query_devices(kind="input")["index"]
    except Exception:
        pass
    out = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
           if d["max_input_channels"] > 0]
    return out, default


class App:
    BG = "#1e1e1e"
    FG = "#e8e8e8"
    SUB = "#7ddcff"      # 字幕青色
    METER_ON = "#4cd964"
    METER_OFF = "#555"

    def __init__(self, root):
        self.root = root
        root.title("实时中英字幕 · sherpa-onnx")
        root.configure(bg=self.BG)
        root.geometry("980x640")
        root.minsize(720, 480)

        self.q = queue.Queue()
        self.stop_evt = threading.Event()
        self.thread = None
        self.running = False
        self._fed = 0.0
        self._rtf = 0.0

        self.devices, self.default_dev = list_input_devices()
        self._build_widgets()
        root.after(60, self._poll)

    # ------------------------------------------------------------- widgets
    def _build_widgets(self):
        pad = dict(padx=8, pady=6)

        bar = tk.Frame(self.root, bg=self.BG)
        bar.pack(fill="x", **pad)

        tk.Label(bar, text="模型", bg=self.BG, fg=self.FG, font=("Microsoft YaHei UI", 10))\
            .grid(row=0, column=0, sticky="e")
        self.model_var = tk.StringVar(value="x-asr_punct_int8_480ms")
        self.model_cb = ttk.Combobox(
            bar, textvariable=self.model_var, state="readonly", width=34,
            values=[m["name"] for m in MODELS])
        self.model_cb.grid(row=0, column=1, padx=(4, 16))

        tk.Label(bar, text="麦克风", bg=self.BG, fg=self.FG, font=("Microsoft YaHei UI", 10))\
            .grid(row=0, column=2, sticky="e")
        dev_names = [f"{i}: {n}" for i, n in self.devices]
        self.dev_var = tk.StringVar(
            value=dev_names[self.default_dev] if self.default_dev is not None else (dev_names[0] if dev_names else ""))
        self.dev_cb = ttk.Combobox(bar, textvariable=self.dev_var, state="readonly",
                                   width=34, values=dev_names)
        self.dev_cb.grid(row=0, column=3, padx=(4, 8))
        ttk.Button(bar, text="刷新", command=self._refresh_devices, width=5)\
            .grid(row=0, column=4)

        self.start_btn = ttk.Button(bar, text="开始识别", command=self._start, width=10)
        self.start_btn.grid(row=0, column=5, padx=(14, 4))
        self.stop_btn = ttk.Button(bar, text="结束识别", command=self._stop, width=10, state="disabled")
        self.stop_btn.grid(row=0, column=6, padx=(0, 8))

        # 电平 + 状态行
        info = tk.Frame(self.root, bg=self.BG)
        info.pack(fill="x", **pad)
        self.meter_var = tk.StringVar(value="[                    ] 电平")
        self.meter_lbl = tk.Label(info, textvariable=self.meter_var, bg=self.BG,
                                  fg=self.METER_OFF, font=("Consolas", 11), anchor="w")
        self.meter_lbl.pack(side="left")
        self.status_var = tk.StringVar(value="就绪 —— 选好模型和麦克风，点「开始识别」")
        tk.Label(info, textvariable=self.status_var, bg=self.BG, fg="#9aa", anchor="e")\
            .pack(side="right", fill="x", expand=True)

        # 实时字幕大字
        self.subtitle_var = tk.StringVar(value="（字幕区）")
        self.sub_lbl = tk.Label(self.root, textvariable=self.subtitle_var, bg=self.BG,
                                fg=self.SUB, font=("Microsoft YaHei UI", 22, "bold"),
                                wraplength=940, justify="left", anchor="nw")
        self.sub_lbl.pack(fill="both", expand=True, padx=10, pady=(4, 6))
        self.root.bind("<Configure>", self._on_resize)

        # 结果日志
        logbar = tk.Frame(self.root, bg=self.BG)
        logbar.pack(fill="x", padx=8)
        tk.Label(logbar, text="识别结果（每次会话追加）", bg=self.BG, fg=self.FG,
                 font=("Microsoft YaHei UI", 10)).pack(side="left")
        ttk.Button(logbar, text="复制全部", command=self._copy_log, width=8).pack(side="right")
        ttk.Button(logbar, text="清空", command=self._clear_log, width=5).pack(side="right", padx=4)

        self.log = tk.Text(self.root, height=8, bg="#141414", fg="#d0d0d0",
                           font=("Microsoft YaHei UI", 11), wrap="word",
                           insertbackground="#fff", relief="flat", padx=8, pady=6)
        self.log.pack(fill="x", padx=8, pady=(4, 10))
        self.log.configure(state="disabled")

    # ------------------------------------------------------------- helpers
    def _on_resize(self, evt):
        if evt.widget is self.root:
            self.sub_lbl.configure(wraplength=max(400, self.root.winfo_width() - 24))

    def _refresh_devices(self):
        if self.running:
            return
        self.devices, self.default_dev = list_input_devices()
        dev_names = [f"{i}: {n}" for i, n in self.devices]
        self.dev_cb.configure(values=dev_names)
        if self.default_dev is not None and self.default_dev < len(dev_names):
            self.dev_var.set(dev_names[self.default_dev])
        elif dev_names:
            self.dev_var.set(dev_names[0])

    def _dev_index(self):
        val = self.dev_var.get()
        return int(val.split(":")[0]) if val else None

    def _set_running(self, on):
        self.running = on
        self.start_btn.configure(state="disabled" if on else "normal")
        self.stop_btn.configure(state="normal" if on else "disabled")
        self.model_cb.configure(state="disabled" if on else "readonly")
        self.dev_cb.configure(state="disabled" if on else "readonly")

    # ------------------------------------------------------------- control
    def _start(self):
        if self.running:
            return
        if not self.devices:
            self.status_var.set("没有可用的输入设备！")
            return
        self._set_running(True)
        self._fed = 0.0
        self._rtf = 0.0
        self.stop_evt = threading.Event()
        name = self.model_var.get()
        label = MODEL_LABELS.get(name, name)
        self.status_var.set(f"正在加载模型: {label} …")
        self.subtitle_var.set("（加载模型，稍候…）")
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _stop(self):
        if not self.running:
            return
        self.stop_evt.set()
        self.status_var.set("已请求结束，正在收尾…")

    def _worker(self):
        import sounddevice as sd
        name = self.model_var.get()
        cfg = next(c for c in MODELS if c["name"] == name)
        dev = self._dev_index()
        try:
            t0 = time.perf_counter()
            rec = build_rec(cfg, threads=4)
            stream = rec.create_stream()
            self.q.put(("status", f"模型已加载（{time.perf_counter()-t0:.1f}s），设备 {dev}，开始识别"))
        except Exception as e:
            self.q.put(("error", f"模型加载失败: {e}"))
            self.q.put(("done", None))
            return
        t_start = time.perf_counter()
        try:
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                blocksize=CHUNK, device=dev) as inp:
                while not self.stop_evt.is_set():
                    block, _ = inp.read(CHUNK)
                    block = np.clip(block[:, 0], -1.0, 1.0).astype(np.float32)
                    stream.accept_waveform(SR, block)
                    while rec.is_ready(stream):
                        rec.decode_stream(stream)
                    self._fed += CHUNK / SR
                    rms = float(np.sqrt(np.mean(block ** 2)))
                    partial = rec.get_result(stream).strip().replace("\n", " ")
                    self.q.put(("partial", (partial, rms)))
                    if self._fed >= 1.0 and self._fed % 1.0 < 0.11:
                        self._rtf = self._fed / max(1e-6, time.perf_counter() - t_start)
        except Exception as e:
            self.q.put(("error", f"音频输入失败: {e}"))
        # 收尾：静音尾 + input_finished + 解到最终
        try:
            tail = np.zeros(int(0.3 * SR), dtype=np.float32)
            stream.accept_waveform(SR, tail)
            stream.input_finished()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            res = rec.get_result_all(stream)
            final = res.text if hasattr(res, "text") else str(res)
            self.q.put(("final", final.strip()))
        except Exception as e:
            self.q.put(("error", f"收尾失败: {e}"))
        self.q.put(("done", None))

    # ------------------------------------------------------------- pump
    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "partial":
                    text, rms = payload
                    self.subtitle_var.set(text if text else "…")
                    self._draw_meter(rms)
                elif kind == "final":
                    self._append_log(f"[{time.strftime('%H:%M:%S')}] {payload}\n")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "error":
                    self.status_var.set(payload)
                    self._append_log(f"[错误] {payload}\n")
                elif kind == "done":
                    self._set_running(False)
                    self.status_var.set(
                        f"已结束（实时率≈{self._rtf:.2f}x，小于 1 即能跟上语速）")
                    self.subtitle_var.set("（会话结束，可切换模型重新开始）")
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _draw_meter(self, rms):
        n = int(min(1.0, rms * 25) * 20)
        bar = "█" * n + " " * (20 - n)
        self.meter_var.set(f"[{bar}] 电平")
        self.meter_lbl.configure(fg=self.METER_ON if n else self.METER_OFF)

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log.get("1.0", "end").strip())
        self.status_var.set("已复制到剪贴板")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
