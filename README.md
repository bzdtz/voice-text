# 语音输入（本地 / 云端双模式）

按住 `Ctrl + Space` 录音，松开后识别整段音频；成功结果会自动复制到剪贴板，并显示在主窗口和可编辑浮窗中。

## 模式

- `local`：继续使用 `AI_Models/models/*/SenseVoiceSmall` 中现有的本地 SenseVoice 模型。
- `cloud`：上传一次完整录音到 SiliconFlow：`POST https://api.siliconflow.cn/v1/audio/transcriptions`，模型为 `FunAudioLLM/SenseVoiceSmall`。

在应用的“设置”中选择模式。云端模式的 API Key 保存于 Windows 的 `%APPDATA%\VoiceTextInput\settings.json`，不会写进项目目录。

当前项目是 Python/Tk 桌面程序，因此保留 `sounddevice` 的单声道录音实现；云端模式将该录音编码为内存中的 `audio/wav` 后进行 multipart 上传。它不依赖浏览器的 `getUserMedia` / `MediaRecorder`，也无需 ffmpeg 或本地 ASR 进程。

## 安装与启动

```powershell
pip install -r requirements.txt
python main.py
```

也可运行 `run.bat`。它会优先使用已配置的 `utools` Python 环境，环境不存在时回退到系统 `python`。
若启动失败，可查看项目目录下自动生成的 `startup.log`。

## 测试

```powershell
python -m unittest -v test_cloud_asr.py
```

测试覆盖云端请求构造、`text` 响应解析、取消结果、缺失响应文本和服务端错误信息。
