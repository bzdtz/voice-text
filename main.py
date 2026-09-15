"""Dual-mode push-to-talk desktop voice input.

Ctrl+Space starts recording and releasing either key submits one complete
recording.  Recognition can be switched between local streaming Zipformer
(Chinese/English via sherpa-onnx) and SiliconFlow's cloud ASR endpoint.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, scrolledtext, ttk

import keyboard
import numpy as np
import pyperclip
import pystray
import sounddevice as sd
from PIL import Image, ImageDraw

from audio_processing import apply_auto_gain
from cloud_asr import CloudASRError, RequestCancelled, SiliconFlowASR
from local_asr import (
    StreamingZipformerASR,
    read_hotwords_text,
    write_hotwords_text,
)

SAMPLE_RATE = 16000
MIN_RECORDING_SECONDS = 0.25
# Ctrl+Space 在中文输入法下容易瞬间断一下；释放需连续确认后才定稿。
HOTKEY_RELEASE_STABLE_POLLS = 4
FLOATING_WIDTH = 210
FLOATING_HEIGHT = 76
RESULT_MIN_WIDTH = 230
RESULT_MAX_WIDTH = 720
RESULT_MIN_HEIGHT = 56
RESULT_PAD_X = 24
RESULT_PAD_Y = 20
# Must match edit_text padx/pady in _create_edit_popup.
RESULT_TEXT_PAD_X = 10
RESULT_TEXT_PAD_Y = 6
RESULT_TEXT_FONT = ("Microsoft YaHei", 11)
# 未满该字数必须完整展示、不可出现滚动条；达到后过高才改为可滚动。
RESULT_SCROLL_CHAR_THRESHOLD = 100
RESULT_SCROLL_MAX_LINES = 8
STREAM_POLL_MS = 80


def application_settings_path() -> Path:
    """Keep credentials outside of the project directory and source control."""
    app_data = Path(os.environ.get("APPDATA", Path.home() / ".config"))
    return app_data / "VoiceTextInput" / "settings.json"


class SettingsStore:
    def load(self) -> dict:
        try:
            settings = json.loads(application_settings_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            settings = {}
        mode = settings.get("mode")
        return {
            # Cloud mode is the default.  The local model is loaded only after
            # the user explicitly selects and saves the local mode.
            "mode": mode if mode in {"local", "cloud"} else "cloud",
            "api_key": settings.get("api_key", "") if isinstance(settings.get("api_key", ""), str) else "",
        }

    def save(self, settings: dict) -> None:
        target = application_settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "mode": settings["mode"],
                    "api_key": settings["api_key"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, target)


class VoiceInputApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("语音输入")
        self.root.geometry("560x470")
        self.root.minsize(480, 380)
        self.root.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

        self.settings_store = SettingsStore()
        self.settings = self.settings_store.load()
        self.mode = self.settings["mode"]
        self.api_key = self.settings["api_key"]
        self.is_running = True
        self.is_recording = False
        self.is_processing = False
        self.stream: sd.InputStream | None = None
        self.audio_chunks: list[np.ndarray] = []
        self.recording_started_at = 0.0
        self.local_model: StreamingZipformerASR | None = None
        self.model_loading = False
        self.cancel_event: threading.Event | None = None
        self.request_id = 0
        self.edit_popup: tk.Toplevel | None = None
        self.edit_timer = None
        self.floating_window: tk.Toplevel | None = None
        self.floating_canvas: tk.Canvas | None = None
        self.wave_bars: list[int] = []
        self.wave_status_id: int | None = None
        self.current_volume = 0.0
        self.visual_height = 0.0
        self.wave_phase = 0.0
        self.wave_animation_running = False
        self.tray_icon: pystray.Icon | None = None
        self._local_session = None
        self._streamed_samples = 0
        self._last_partial = ""
        self._stream_poll_after = None
        self._recording_partial = False
        self._hide_floating_after = None
        self._recording_token = 0
        self._edit_popup_scrollable = False

        self._build_ui()
        self._create_edit_popup()
        self._create_floating_window()
        self._create_tray_icon()
        self._start_hotkey_listener()
        self._update_mode_status(load_local=self.mode == "local")

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, padx=18, pady=14)
        header.pack(fill=tk.X)
        tk.Label(header, text="语音输入", font=("Microsoft YaHei", 18, "bold")).pack(side=tk.LEFT)
        tk.Button(header, text="设置", command=self.open_settings).pack(side=tk.RIGHT)
        tk.Button(header, text="最小化到托盘", command=self.minimize_to_tray).pack(side=tk.RIGHT, padx=(0, 8))

        self.mode_label = tk.Label(self.root, font=("Microsoft YaHei", 10), fg="#496582")
        self.mode_label.pack()
        self.status_label = tk.Label(self.root, text="正在初始化…", font=("Microsoft YaHei", 12, "bold"), fg="#007AFF")
        self.status_label.pack(pady=(8, 16))

        action_frame = tk.Frame(self.root, padx=18, pady=12, bd=1, relief=tk.GROOVE)
        action_frame.pack(fill=tk.X, padx=18)
        self.recording_label = tk.Label(action_frame, text="按住 Ctrl + Space 开始说话", font=("Microsoft YaHei", 12))
        self.recording_label.pack(pady=(2, 6))
        tk.Label(action_frame, text="松开后上传/识别整段音频；短于 250ms 的录音将丢弃", fg="#777777").pack()
        self.cancel_button = tk.Button(action_frame, text="取消当前识别", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(pady=(10, 0))

        result_header = tk.Frame(self.root, padx=18, pady=4)
        result_header.pack(fill=tk.X, pady=(18, 0))
        tk.Label(result_header, text="识别结果", font=("Microsoft YaHei", 11, "bold")).pack(side=tk.LEFT)
        tk.Button(result_header, text="清空", command=self.clear_result).pack(side=tk.RIGHT)
        self.result_area = scrolledtext.ScrolledText(self.root, height=10, font=("Microsoft YaHei", 11), wrap=tk.WORD)
        self.result_area.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))
        self.result_area.configure(state=tk.DISABLED)

    def _create_tray_icon(self) -> None:
        image = Image.new("RGB", (64, 64), "#3978E8")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((24, 10, 40, 40), radius=8, fill="white")
        draw.arc((16, 22, 48, 52), 0, 180, fill="white", width=4)
        draw.line((32, 48, 32, 57), fill="white", width=4)
        draw.line((22, 57, 42, 57), fill="white", width=4)
        menu = pystray.Menu(
            pystray.MenuItem("显示主窗口", self._show_main_window),
            pystray.MenuItem("退出", self.quit_app),
        )
        self.tray_icon = pystray.Icon("VoiceTextInput", image, "语音输入", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def minimize_to_tray(self) -> None:
        self._hide_edit_popup()
        self._hide_floating_window()
        self.root.withdraw()

    def _show_main_window(self, _icon=None, _item=None) -> None:
        self.root.after(0, self._restore_main_window)

    def _restore_main_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes("-topmost", True)
        self.root.after(100, lambda: self.root.attributes("-topmost", False))

    def _create_edit_popup(self) -> None:
        self.edit_popup = tk.Toplevel(self.root)
        self.edit_popup.withdraw()
        self.edit_popup.overrideredirect(True)
        self.edit_popup.attributes("-topmost", True)
        self.edit_popup.attributes("-alpha", 0.97)
        transparent = "#000002"
        self.edit_popup.configure(bg=transparent)
        try:
            self.edit_popup.attributes("-transparentcolor", transparent)
        except tk.TclError:
            pass
        self.edit_canvas = tk.Canvas(
            self.edit_popup, width=RESULT_MIN_WIDTH, height=RESULT_MIN_HEIGHT, bg=transparent, highlightthickness=0
        )
        self.edit_canvas.pack()
        self.edit_text = tk.Text(
            self.edit_popup,
            wrap=tk.WORD,
            font=RESULT_TEXT_FONT,
            bg="#252525",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT,
            highlightthickness=0,
            bd=0,
            padx=RESULT_TEXT_PAD_X,
            pady=RESULT_TEXT_PAD_Y,
        )
        self.edit_text.bind("<MouseWheel>", self._on_edit_mousewheel)
        self.edit_popup.bind("<Enter>", lambda _event: self._cancel_edit_timer())
        self.edit_popup.bind("<Leave>", lambda _event: self._start_edit_timer())

    def _create_floating_window(self) -> None:
        """Create the shared recording/processing bubble for both ASR modes."""
        self.floating_window = tk.Toplevel(self.root)
        self.floating_window.overrideredirect(True)
        self.floating_window.withdraw()
        self.floating_window.attributes("-topmost", True)
        self.floating_window.attributes("-alpha", 0.96)

        transparent = "#000001"
        self.floating_window.configure(bg=transparent)
        try:
            self.floating_window.attributes("-transparentcolor", transparent)
        except tk.TclError:
            pass

        width, height = FLOATING_WIDTH, FLOATING_HEIGHT
        self.floating_canvas = tk.Canvas(
            self.floating_window, width=width, height=height, bg=transparent, highlightthickness=0
        )
        self.floating_canvas.pack()
        self._draw_rounded_rect(self.floating_canvas, 2, 2, width - 2, height - 2, 18, fill="#1E1E1E", outline="#3A3A3A")

        center_x, wave_y = width // 2, 28
        for index in range(17):
            x = center_x - 48 + index * 6
            self.wave_bars.append(
                self.floating_canvas.create_line(x, wave_y, x, wave_y, width=4, fill="#FFFFFF", capstyle=tk.ROUND)
            )
        self.wave_status_id = self.floating_canvas.create_text(
            center_x, 56, text="准备就绪", font=("Microsoft YaHei UI", 9), fill="#AAAAAA"
        )
        self._set_wave_bars_visible(False)

    @staticmethod
    def _draw_rounded_rect(canvas: tk.Canvas, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _set_wave_bars_visible(self, visible: bool) -> None:
        if not self.floating_canvas:
            return
        state = tk.NORMAL if visible else tk.HIDDEN
        for bar_id in self.wave_bars:
            self.floating_canvas.itemconfig(bar_id, state=state)
        if self.wave_status_id is not None:
            # Listening: status under waves. Idle/done: status centered in the bubble.
            y = 56 if visible else FLOATING_HEIGHT // 2
            self.floating_canvas.coords(self.wave_status_id, FLOATING_WIDTH // 2, y)

    def _floating_screen_pos(self) -> tuple[int, int]:
        """Same bottom-center anchor used by the wave bubble."""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - FLOATING_WIDTH) // 2
        y = screen_height - FLOATING_HEIGHT - 130
        return x, y

    def _show_floating_window(self, text: str) -> None:
        if not self.floating_window or not self.floating_canvas or self.wave_status_id is None:
            return
        self._cancel_hide_floating()
        listening = text.startswith("正在聆听")
        self._set_wave_bars_visible(listening)
        self.floating_canvas.itemconfig(self.wave_status_id, text=text)
        x, y = self._floating_screen_pos()
        self.floating_window.geometry(f"+{x}+{y}")
        self.floating_window.deiconify()
        self.floating_window.lift()
        if listening:
            if not self.wave_animation_running:
                self.wave_animation_running = True
                self.visual_height = 0.0
                self._animate_waves()
        else:
            self.wave_animation_running = False
            self.current_volume = 0.0
            self.visual_height = 0.0

    def _cancel_hide_floating(self) -> None:
        if self._hide_floating_after is not None:
            try:
                self.root.after_cancel(self._hide_floating_after)
            except Exception:
                pass
            self._hide_floating_after = None

    def _hide_floating_window(self) -> None:
        self._cancel_hide_floating()
        if self.floating_window:
            self.floating_window.withdraw()
        self.wave_animation_running = False
        self.current_volume = 0.0
        self.visual_height = 0.0
        self._set_wave_bars_visible(False)

    def _schedule_hide_floating(self, delay_ms: int = 1000) -> None:
        self._cancel_hide_floating()
        self._hide_floating_after = self.root.after(delay_ms, self._hide_floating_if_idle)

    def _animate_waves(self) -> None:
        if not self.wave_animation_running or not self.floating_window or not self.floating_window.winfo_viewable():
            self.wave_animation_running = False
            return
        if not self.is_recording:
            self.wave_animation_running = False
            self._set_wave_bars_visible(False)
            return
        self.wave_phase += 0.22
        target_height = math.sqrt(max(self.current_volume, 0.0)) * 80
        self.visual_height = self.visual_height * 0.7 + target_height * 0.3
        wave_y = 28
        midpoint = len(self.wave_bars) // 2
        for index, bar_id in enumerate(self.wave_bars):
            distance = abs(index - midpoint)
            focus = max(0.0, 1 - distance / (len(self.wave_bars) / 2))
            height = 3 + self.visual_height * focus * (0.8 + 0.3 * math.sin(self.wave_phase + index * 0.6))
            height = min(max(height, 2), 28)
            x = self.floating_canvas.coords(bar_id)[0]
            self.floating_canvas.coords(bar_id, x, wave_y - height, x, wave_y + height)
        self.root.after(30, self._animate_waves)

    def open_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("设置")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        frame = tk.Frame(dialog, padx=18, pady=18)
        frame.pack(fill=tk.BOTH, expand=True)

        mode_labels = {"local": "本地流式（中英 Zipformer）", "cloud": "云端 SiliconFlow"}
        mode_by_label = {label: mode for mode, label in mode_labels.items()}
        mode_var = tk.StringVar(value=mode_labels.get(self.mode, mode_labels["cloud"]))
        api_key_var = tk.StringVar(value=self.api_key)
        tk.Label(frame, text="识别模式").grid(row=0, column=0, sticky="nw", pady=(0, 8))
        mode_box = ttk.Combobox(
            frame,
            state="readonly",
            width=28,
            textvariable=mode_var,
            values=tuple(mode_by_label),
        )
        mode_box.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        tk.Label(
            frame,
            text="local：本地 sherpa-onnx 流式中英模型\ncloud：SiliconFlow 云端 ASR",
            justify=tk.LEFT,
            fg="#666666",
        ).grid(row=1, column=1, sticky="w", pady=(0, 12))
        tk.Label(frame, text="SiliconFlow API Key").grid(row=2, column=0, sticky="w")
        api_key_entry = tk.Entry(frame, width=32, show="*", textvariable=api_key_var)
        api_key_entry.grid(row=2, column=1, sticky="ew")
        tk.Label(frame, text="密钥仅保存到本机应用设置，不写入本项目。", fg="#666666").grid(
            row=3, column=1, sticky="w", pady=(6, 10)
        )
        tk.Label(frame, text="本地热词").grid(row=4, column=0, sticky="nw")
        hotwords_box = scrolledtext.ScrolledText(frame, width=32, height=6, font=("Microsoft YaHei", 10))
        hotwords_box.grid(row=4, column=1, sticky="ew")
        hotwords_box.insert("1.0", read_hotwords_text())
        tk.Label(
            frame,
            text="一行一个；仅本地模式生效。可写中英，例如：\nCursor\n语音输入:2.5",
            justify=tk.LEFT,
            fg="#666666",
        ).grid(row=5, column=1, sticky="w", pady=(6, 14))
        frame.columnconfigure(1, weight=1)

        def save() -> None:
            selected_mode = mode_by_label[mode_var.get()]
            key = api_key_var.get().strip()
            if selected_mode == "cloud" and not key:
                messagebox.showwarning("需要 API Key", "云端模式需要填写 SiliconFlow API Key。", parent=dialog)
                return
            new_hotwords = hotwords_box.get("1.0", "end-1c")
            old_hotwords = read_hotwords_text()
            self.cancel_processing()
            self.mode = selected_mode
            self.api_key = key
            self.settings = {
                "mode": self.mode,
                "api_key": self.api_key,
            }
            try:
                self.settings_store.save(self.settings)
                write_hotwords_text(new_hotwords)
            except OSError as exc:
                messagebox.showerror("保存失败", f"无法保存设置：{exc}", parent=dialog)
                return
            dialog.destroy()
            reload_local = self.mode == "local" and (
                self.local_model is None or new_hotwords.strip() != old_hotwords.strip()
            )
            if reload_local:
                self.local_model = None
            self._update_mode_status(load_local=reload_local)

        buttons = tk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e")
        tk.Button(buttons, text="取消", command=dialog.destroy).pack(side=tk.RIGHT)
        tk.Button(buttons, text="保存", command=save).pack(side=tk.RIGHT, padx=(0, 8))

    def _update_mode_status(self, load_local: bool) -> None:
        if self.mode == "cloud":
            self.mode_label.config(text="当前模式：云端 SiliconFlow（FunAudioLLM/SenseVoiceSmall）")
            self.set_status("待命中（云端模式，Ctrl + Space）", "#007AFF")
        else:
            extras = []
            if self.local_model and self.local_model.uses_hotwords:
                extras.append("热词")
            if self.local_model and self.local_model.uses_punctuation:
                extras.append("标点")
            extra_note = f"（已启用{'/'.join(extras)}）" if extras else ""
            self.mode_label.config(text=f"当前模式：本地流式 Zipformer（中英）{extra_note}")
            if self.local_model is not None:
                self.set_status("待命中（本地流式，Ctrl + Space）", "#007AFF")
            elif load_local and not self.model_loading:
                self._load_local_model()

    def _load_local_model(self) -> None:
        self.model_loading = True
        self.set_status("正在加载本地流式模型…", "#FF9500")
        threading.Thread(target=self._load_local_model_worker, daemon=True).start()

    def _load_local_model_worker(self) -> None:
        try:
            model = StreamingZipformerASR()
            self.local_model = model
            extras = []
            if model.uses_hotwords:
                extras.append("热词")
            if model.uses_punctuation:
                extras.append("标点")
            extra_note = f"（已启用{'/'.join(extras)}）" if extras else ""
            status_note = f"，{'/'.join(extras)}已加载" if extras else ""

            def mark_ready() -> None:
                self.mode_label.config(text=f"当前模式：本地流式 Zipformer（中英）{extra_note}")
                self.set_status(f"待命中（本地流式{status_note}，Ctrl + Space）", "#007AFF")

            self.root.after(0, mark_ready)
        except Exception as exc:
            message = f"本地模型加载失败：{exc}"
            self.root.after(0, lambda msg=message: self.set_status(msg, "#FF3B30"))
        finally:
            self.model_loading = False

    def set_status(self, text: str, color: str = "#007AFF") -> None:
        self.status_label.config(text=text, fg=color)

    def _cancel_stream_poll(self) -> None:
        if self._stream_poll_after is not None:
            try:
                self.root.after_cancel(self._stream_poll_after)
            except Exception:
                pass
            self._stream_poll_after = None

    def _schedule_stream_poll(self) -> None:
        self._cancel_stream_poll()
        if self.is_recording and self.mode == "local" and self._local_session is not None:
            self._stream_poll_after = self.root.after(STREAM_POLL_MS, self._poll_streaming_partial)

    def _poll_streaming_partial(self) -> None:
        self._stream_poll_after = None
        if not self.is_recording or self._local_session is None or not self.audio_chunks:
            self._schedule_stream_poll()
            return
        audio = np.concatenate(self.audio_chunks, axis=0)
        mono = audio[:, 0] if audio.ndim > 1 else audio.reshape(-1)
        if mono.size > self._streamed_samples:
            new_samples = np.ascontiguousarray(mono[self._streamed_samples :])
            self._streamed_samples = int(mono.size)
            try:
                text = self._local_session.accept(new_samples)
            except Exception as exc:
                self.set_status(f"流式识别异常：{exc}", "#FF3B30")
                return
            if text != self._last_partial:
                self._last_partial = text
                if text.strip():
                    self._recording_partial = True
                    self._cancel_edit_timer()
                    self._show_edit_popup(text, start_timer=False, live=True)
        self._schedule_stream_poll()

    def start_recording(self) -> None:
        if self.is_recording:
            return
        if self.mode == "local" and self.local_model is None:
            self.set_status("本地模型尚未加载完成。", "#FF9500")
            return
        if self.mode == "cloud" and not self.api_key:
            self.set_status("请在设置中填写 SiliconFlow API Key。", "#FF3B30")
            return
        # 先切到「正在聆听」，避免上一段还停在「正在识别」时闪一下。
        self._cancel_hide_floating()
        self._show_floating_window("正在聆听…")
        self.cancel_processing(hide_floating=False)
        self.is_recording = True
        self._recording_token += 1
        recording_token = self._recording_token
        self.audio_chunks = []
        self.recording_started_at = time.monotonic()
        self._streamed_samples = 0
        self._last_partial = ""
        self._recording_partial = False
        self._local_session = self.local_model.create_session() if self.mode == "local" else None
        self.recording_label.config(text="正在录音…松开 Ctrl + Space 后识别")
        self.set_status("正在录音…", "#FF3B30")
        self._hide_edit_popup()
        self._show_floating_window("正在聆听…")
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            if self.mode == "local" and recording_token == self._recording_token:
                self._schedule_stream_poll()
        except Exception as exc:
            if recording_token == self._recording_token:
                self.is_recording = False
                self._local_session = None
                self._cancel_stream_poll()
                self._hide_floating_window()
                self.set_status(f"无法使用录音设备：{exc}", "#FF3B30")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        token = self._recording_token
        duration = time.monotonic() - self.recording_started_at
        stream = self.stream
        chunks = self.audio_chunks
        session = self._local_session
        streamed = self._streamed_samples

        self.is_recording = False
        self._cancel_stream_poll()
        self.recording_label.config(text="按住 Ctrl + Space 开始说话")

        # 若松开处理期间用户又按下，新录音会抬高 token；不要动新会话的设备。
        if token != self._recording_token:
            return

        try:
            if stream:
                stream.stop()
                stream.close()
        finally:
            if token == self._recording_token:
                self.stream = None

        if token != self._recording_token:
            return

        if duration < MIN_RECORDING_SECONDS or not chunks:
            self._local_session = None
            self._hide_edit_popup()
            self._hide_floating_window()
            self.set_status("录音不足 250ms，已丢弃。", "#FF9500")
            return

        if self.mode == "local":
            self._local_session = None
            self._start_local_finalize(session, np.concatenate(chunks, axis=0), streamed, token)
            return

        audio = apply_auto_gain(np.concatenate(chunks, axis=0))
        self._start_recognition(audio, token)

    def _audio_callback(self, indata, _frames, _time_info, _status) -> None:
        if self.is_recording:
            self.audio_chunks.append(indata.copy())
            if len(indata):
                self.current_volume = float(np.linalg.norm(indata) / np.sqrt(len(indata)))

    def _start_local_finalize(self, session, audio: np.ndarray, streamed_samples: int, token: int) -> None:
        if token != self._recording_token:
            return
        self.is_processing = True
        self.cancel_event = threading.Event()
        self.request_id += 1
        request_id = self.request_id
        self.cancel_button.config(state=tk.NORMAL)
        self.set_status("正在定稿本地识别…", "#FF9500")
        # 定稿中不占用声波浮窗，避免下一次按下时先闪出「正在识别」。
        self._hide_floating_window()
        threading.Thread(
            target=self._local_finalize_worker,
            args=(session, audio, streamed_samples, request_id, self.cancel_event),
            daemon=True,
        ).start()

    def _local_finalize_worker(self, session, audio, streamed_samples, request_id, cancel_event) -> None:
        try:
            if session is None:
                raise CloudASRError("本地识别会话无效。")
            mono = audio[:, 0] if getattr(audio, "ndim", 1) > 1 else np.asarray(audio).reshape(-1)
            if mono.size > streamed_samples:
                session.accept(np.ascontiguousarray(mono[streamed_samples:]))
            if cancel_event.is_set() or request_id != self.request_id:
                return
            text = session.finish()
            if cancel_event.is_set() or request_id != self.request_id:
                return
            clean_text = text.strip()
            if not clean_text:
                raise CloudASRError("没有识别到文字。")
            self.root.after(0, lambda: self._recognition_succeeded(clean_text, request_id))
        except RequestCancelled:
            pass
        except Exception as exc:
            if not cancel_event.is_set() and request_id == self.request_id:
                message = str(exc)
                self.root.after(0, lambda msg=message, rid=request_id: self._recognition_failed(msg, rid))
        finally:
            self.root.after(0, lambda: self._finish_request(request_id))

    def _start_recognition(self, audio: np.ndarray, token: int) -> None:
        if token != self._recording_token:
            return
        self.is_processing = True
        self.cancel_event = threading.Event()
        self.request_id += 1
        request_id = self.request_id
        self.cancel_button.config(state=tk.NORMAL)
        self.set_status("正在上传并识别整段录音…", "#FF9500")
        # 同上：识别中隐藏浮窗，按下时应直接出现「正在聆听」。
        self._hide_floating_window()
        api_key = self.api_key
        threading.Thread(
            target=self._recognition_worker,
            args=(audio, api_key, request_id, self.cancel_event),
            daemon=True,
        ).start()

    def _recognition_worker(self, audio, api_key, request_id, cancel_event) -> None:
        try:
            text = SiliconFlowASR(api_key).transcribe(audio, SAMPLE_RATE, cancel_event)
            if cancel_event.is_set() or request_id != self.request_id:
                return
            clean_text = re.sub(r"<\|.*?\|>", "", text).strip()
            if not clean_text:
                raise CloudASRError("没有识别到文字。")
            self.root.after(0, lambda: self._recognition_succeeded(clean_text, request_id))
        except RequestCancelled:
            pass
        except Exception as exc:
            if not cancel_event.is_set() and request_id == self.request_id:
                message = str(exc)
                self.root.after(0, lambda msg=message, rid=request_id: self._recognition_failed(msg, rid))
        finally:
            self.root.after(0, lambda: self._finish_request(request_id))

    def _recognition_succeeded(self, text: str, request_id: int) -> None:
        if request_id != self.request_id:
            return
        self._recording_partial = False
        self.result_area.config(state=tk.NORMAL)
        self.result_area.insert(tk.END, text + "\n\n")
        self.result_area.see(tk.END)
        self.result_area.config(state=tk.DISABLED)
        pyperclip.copy(text)
        # 先弹出声波状态，结果框才能贴着它定位（定稿阶段浮窗是隐藏的）。
        self.set_status("识别完成，文字已复制到剪贴板。", "#34C759")
        self._show_floating_window("识别完成")
        self._show_edit_popup(text, start_timer=True)
        self._schedule_hide_floating(1000)

    def _recognition_failed(self, message: str, request_id: int) -> None:
        if request_id == self.request_id:
            self._recording_partial = False
            self.set_status(f"识别失败：{message}", "#FF3B30")
            self._hide_floating_window()

    def _hide_floating_if_idle(self) -> None:
        if not self.is_recording and not self.is_processing:
            self._hide_floating_window()

    def _finish_request(self, request_id: int) -> None:
        if request_id == self.request_id:
            self.is_processing = False
            self.cancel_event = None
            self.cancel_button.config(state=tk.DISABLED)

    def cancel_processing(self, hide_floating: bool = True) -> None:
        self._cancel_stream_poll()
        self._local_session = None
        self._recording_partial = False
        if self.cancel_event and not self.cancel_event.is_set():
            self.cancel_event.set()
            self.request_id += 1
            self.is_processing = False
            self.cancel_button.config(state=tk.DISABLED)
            if hide_floating:
                self.set_status("已取消当前识别。", "#007AFF")
                self._hide_floating_window()

    def _measure_edit_popup_width(self, text: str) -> int:
        """Widen toward both screen edges; keep padx slack so the last glyph is not forced to wrap."""
        measure_font = tkfont.Font(font=RESULT_TEXT_FONT)
        screen_w = self.root.winfo_screenwidth()
        max_width = min(RESULT_MAX_WIDTH, max(RESULT_MIN_WIDTH, screen_w - 24))
        lines = text.splitlines() or [""]
        longest_px = max((measure_font.measure(line) for line in lines), default=0)
        needed_content = longest_px + RESULT_TEXT_PAD_X * 2 + 24
        content_width = min(max(needed_content, RESULT_MIN_WIDTH - RESULT_PAD_X), max_width - RESULT_PAD_X)
        return content_width + RESULT_PAD_X

    def _count_edit_display_lines(self) -> int:
        """Count real wrapped lines after Tk has laid out the Text (WORD wrap)."""
        lines = 0
        index = "1.0"
        seen: set[str] = set()
        while lines < 500:
            if self.edit_text.compare(index, ">=", "end"):
                break
            if self.edit_text.dlineinfo(index) is None:
                break
            lines += 1
            seen.add(index)
            next_index = self.edit_text.index(f"{index} +1 display lines")
            if next_index in seen or self.edit_text.compare(next_index, "==", index):
                break
            if self.edit_text.compare(next_index, ">=", "end"):
                break
            index = next_index
        return max(1, lines)

    def _edit_popup_screen_pos(self, popup_width: int, popup_height: int) -> tuple[int, int]:
        """Result bubble centered under the wave bubble; width grows to both sides."""
        fx, fy = self._floating_screen_pos()
        x = fx + (FLOATING_WIDTH - popup_width) // 2
        y = fy + FLOATING_HEIGHT + 10
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max(8, min(x, screen_w - popup_width - 8))
        y = max(8, min(y, screen_h - popup_height - 8))
        return x, y

    def _layout_edit_popup(self, popup_width: int, text_height: int, popup_height: int) -> None:
        self.edit_canvas.config(width=popup_width, height=popup_height)
        self.edit_canvas.delete("all")
        self._draw_rounded_rect(
            self.edit_canvas, 2, 2, popup_width - 2, popup_height - 2, 13,
            fill="#252525", outline="#444B55",
        )
        self.edit_canvas.create_window(
            popup_width // 2,
            popup_height // 2,
            window=self.edit_text,
            width=popup_width - RESULT_PAD_X,
            height=text_height,
        )

    def _estimate_edit_popup_height(self, text: str, popup_width: int) -> tuple[int, int, bool]:
        """Fast height estimate for live updates (no off-screen relayout)."""
        measure_font = tkfont.Font(font=RESULT_TEXT_FONT)
        # Prefer slightly tall lines so WORD-wrap underestimates do not crush glyphs.
        line_height = measure_font.metrics("linespace") + 6
        wrap_width = max(40, popup_width - RESULT_PAD_X - RESULT_TEXT_PAD_X * 2)
        char_count = len(text.strip())
        lines = text.splitlines() or [""]
        visual_lines = 0
        for line in lines:
            if not line:
                visual_lines += 1
                continue
            # Approximate WORD wrap: break on spaces when possible, else by char.
            current = 0
            wrapped = 1
            i = 0
            while i < len(line):
                # Take next run: whitespace-separated token or single CJK/other char.
                if line[i].isspace():
                    token = line[i]
                    i += 1
                else:
                    j = i
                    while j < len(line) and not line[j].isspace():
                        # Keep CJK-ish chars as single-char tokens for safer wrap counts.
                        if ord(line[j]) > 0x2E7F:
                            if j > i:
                                break
                            j += 1
                            break
                        j += 1
                    token = line[i:j]
                    i = j
                token_w = measure_font.measure(token)
                if current + token_w > wrap_width and current > 0:
                    wrapped += 1
                    current = token_w
                else:
                    current += token_w
            visual_lines += wrapped
        visual_lines = max(1, visual_lines)
        allow_scroll = char_count >= RESULT_SCROLL_CHAR_THRESHOLD and visual_lines > RESULT_SCROLL_MAX_LINES
        show_lines = RESULT_SCROLL_MAX_LINES if allow_scroll else visual_lines
        text_height = show_lines * line_height + RESULT_TEXT_PAD_Y * 2 + 10
        popup_height = max(RESULT_MIN_HEIGHT, text_height + RESULT_PAD_Y)
        return popup_height, text_height, allow_scroll

    def _fit_edit_popup_to_text(self, text: str, popup_width: int) -> tuple[int, int, bool]:
        """Precise layout with Tk WORD wrap + reqheight. Used for final results only."""
        char_count = len(text.strip())
        probe_h = 2400
        self.edit_text.configure(height=80)
        self._layout_edit_popup(popup_width, probe_h, probe_h + RESULT_PAD_Y)
        self.edit_popup.geometry("+-8000+-8000")
        self.edit_popup.deiconify()
        self.edit_popup.update_idletasks()

        display_lines = self._count_edit_display_lines()
        allow_scroll = char_count >= RESULT_SCROLL_CHAR_THRESHOLD and display_lines > RESULT_SCROLL_MAX_LINES
        show_lines = RESULT_SCROLL_MAX_LINES if allow_scroll else display_lines

        self.edit_text.configure(height=show_lines)
        self.edit_popup.update_idletasks()
        text_height = max(self.edit_text.winfo_reqheight() + 6, RESULT_MIN_HEIGHT - RESULT_PAD_Y)
        popup_height = max(RESULT_MIN_HEIGHT, text_height + RESULT_PAD_Y)
        self._layout_edit_popup(popup_width, text_height, popup_height)
        self.edit_popup.update_idletasks()

        if not allow_scroll and self.edit_text.yview() != (0.0, 1.0):
            extra = self._count_edit_display_lines()
            self.edit_text.configure(height=max(show_lines, extra))
            self.edit_popup.update_idletasks()
            text_height = self.edit_text.winfo_reqheight() + 6
            popup_height = max(RESULT_MIN_HEIGHT, text_height + RESULT_PAD_Y)
            self._layout_edit_popup(popup_width, text_height, popup_height)
            self.edit_popup.update_idletasks()

        return popup_height, text_height, allow_scroll

    def _on_edit_mousewheel(self, event: tk.Event) -> str | None:
        if not getattr(self, "_edit_popup_scrollable", False):
            return None
        if self.edit_text.yview() != (0.0, 1.0):
            self.edit_text.yview_scroll(int(-event.delta / 120), "units")
            return "break"
        return None

    def _show_edit_popup(self, text: str, start_timer: bool = True, live: bool = False) -> None:
        if not self.edit_popup:
            return
        popup_width = self._measure_edit_popup_width(text)
        already_open = bool(self.edit_popup.winfo_viewable())

        self.edit_text.delete("1.0", tk.END)
        self.edit_text.insert("1.0", text)

        if live and already_open:
            # Streaming: only update text and grow in place — never teleport off-screen.
            popup_height, text_height, allow_scroll = self._estimate_edit_popup_height(text, popup_width)
            cur_w = int(self.edit_canvas.winfo_width() or 0)
            cur_h = int(self.edit_canvas.winfo_height() or 0)
            popup_width = max(popup_width, cur_w)
            popup_height = max(popup_height, cur_h)
            text_height = max(text_height, popup_height - RESULT_PAD_Y)
            self._edit_popup_scrollable = allow_scroll
            line_height = tkfont.Font(font=RESULT_TEXT_FONT).metrics("linespace") + 6
            self.edit_text.configure(height=max(1, text_height // line_height))
            if popup_width != cur_w or popup_height != cur_h:
                self._layout_edit_popup(popup_width, text_height, popup_height)
                x, y = self._edit_popup_screen_pos(popup_width, popup_height)
                self.edit_popup.geometry(f"+{x}+{y}")
            self.edit_text.tag_add("sel", "1.0", "end")
            self.edit_text.see("end")
            self._cancel_edit_timer()
            return

        if live:
            popup_height, text_height, allow_scroll = self._estimate_edit_popup_height(text, popup_width)
            line_height = tkfont.Font(font=RESULT_TEXT_FONT).metrics("linespace") + 6
            self.edit_text.configure(height=max(1, text_height // line_height))
            self._layout_edit_popup(popup_width, text_height, popup_height)
        else:
            popup_height, text_height, allow_scroll = self._fit_edit_popup_to_text(text, popup_width)

        self._edit_popup_scrollable = allow_scroll
        x, y = self._edit_popup_screen_pos(popup_width, popup_height)
        self.edit_popup.geometry(f"+{x}+{y}")
        self.edit_popup.deiconify()
        self.edit_popup.lift()
        self.edit_text.tag_add("sel", "1.0", "end")
        self.edit_text.see("1.0" if not live else "end")
        self.edit_text.focus_set()
        if start_timer:
            self._start_edit_timer()
        else:
            self._cancel_edit_timer()

    def _start_edit_timer(self) -> None:
        if self.is_recording or self._recording_partial:
            return
        self._cancel_edit_timer()
        self.edit_timer = self.root.after(2000, self._copy_and_hide_edit_popup)

    def _cancel_edit_timer(self) -> None:
        if self.edit_timer:
            self.root.after_cancel(self.edit_timer)
            self.edit_timer = None

    def _copy_and_hide_edit_popup(self) -> None:
        if self.edit_popup and self.edit_popup.winfo_viewable():
            pyperclip.copy(self.edit_text.get("1.0", "end-1c").strip())
        self._hide_edit_popup()

    def _hide_edit_popup(self) -> None:
        self._cancel_edit_timer()
        if self.edit_popup:
            self.edit_popup.withdraw()

    def clear_result(self) -> None:
        self.result_area.config(state=tk.NORMAL)
        self.result_area.delete("1.0", tk.END)
        self.result_area.config(state=tk.DISABLED)

    def _start_hotkey_listener(self) -> None:
        threading.Thread(target=self._listen_hotkey, daemon=True).start()

    def _hotkey_pressed(self) -> None:
        """Main-thread entry for Ctrl+Space press: force listening UI before setup."""
        self._cancel_hide_floating()
        self._show_floating_window("正在聆听…")
        self.start_recording()

    def _listen_hotkey(self) -> None:
        active = False
        release_polls = 0
        while self.is_running:
            try:
                pressed = keyboard.is_pressed("ctrl") and keyboard.is_pressed("space")
                if pressed and not active:
                    active = True
                    release_polls = 0
                    # 先切文案再进入录音，减少主线程排队时露出旧状态。
                    self.root.after(0, self._hotkey_pressed)
                elif active and pressed:
                    release_polls = 0
                elif active and not pressed:
                    release_polls += 1
                    # 连续若干次轮询都松开，才算真正松手，过滤 Ctrl+Space 抖动。
                    if release_polls >= HOTKEY_RELEASE_STABLE_POLLS:
                        active = False
                        release_polls = 0
                        self.root.after(0, self.stop_recording)
                time.sleep(0.02)
            except Exception:
                # Keep the UI usable on systems where a global keyboard hook is unavailable.
                self.root.after(0, lambda: self.set_status("全局快捷键监听不可用。", "#FF3B30"))
                return

    def quit_app(self, _icon=None, _item=None) -> None:
        self.is_running = False
        self.cancel_processing()
        try:
            if self.stream:
                self.stream.close()
        finally:
            if self.tray_icon:
                self.tray_icon.stop()
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    VoiceInputApp(root)
    root.mainloop()
