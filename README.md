# 语音输入（本地 / 云端双模式）

按住 `Ctrl + Space` 录音，松开后识别整段音频；成功结果会自动复制到剪贴板，并显示在主窗口和可编辑浮窗中。

- `local` 模式：本地推理，全程离线。识别模型为 sherpa-onnx 的 **Streaming Zipformer**（中英双语），可选加载 **CT-Transformer** 标点恢复模型。
- `cloud` 模式：上传一次完整录音到 SiliconFlow，接口 `POST https://api.siliconflow.cn/v1/audio/transcriptions`，模型 `FunAudioLLM/SenseVoiceSmall`。

这是 Python / Tk 桌面程序，录音基于 `sounddevice` 的单声道实现，不依赖浏览器的 `getUserMedia` / `MediaRecorder`，云端模式也无需 ffmpeg 或本地 ASR 进程。

## 安装与启动

```powershell
pip install -r requirements.txt
python main.py
```

也可以双击 `run.bat`。它会自动挑选可用的 Python 解释器，**无需手工指定路径**，优先级如下：

1. 项目目录下的 `.venv`
2. 当前已激活的 conda / virtualenv 环境
3. 本机任意 conda 环境中**已装好项目依赖**的那个
4. `PATH` 中已装好项目依赖的 `python`
5. 以上都没有时，回退到系统 `python` 并提示安装依赖

> 依赖不全的环境会被自动跳过，所以 Conda 里有很多环境时通常不用管，脚本会直接命中正确的那个。

启动失败时可查看项目目录下自动生成的 `startup.log`。

## 本地模式：下载模型

仓库本体不含模型文件（体积过大），**只有要用 `local` 模式才需要下载**，用 `cloud` 模式可直接跳过本节。

在 [Releases](https://github.com/bzdtz/voice-text/releases) 页下载压缩包，解压到项目根的 `sherpa-onnx-probe\models\` 下即可（解压后的目录名必须保持不变）：

| 压缩包 | 用途 | 是否必需 |
| --- | --- | --- |
| `models-streaming-bilingual-zh-en.zip` | 中英双语流式识别主模型 | 是 |
| `models-punctuation-ct-transformer.zip` | 自动补全句号、逗号等标点 | 否，缺失时退化为无标点输出 |
| `models-streaming-zh-14M.zip` | 轻量单语中文模型，仅供 `sherpa-onnx-probe/probe-streaming.js` 测试 | 否 |

PowerShell 一键下载并解压：

```powershell
cd sherpa-onnx-probe\models
Invoke-WebRequest -Uri "https://github.com/bzdtz/voice-text/releases/download/v1.0/models-streaming-bilingual-zh-en.zip" -OutFile "asr.zip"
Expand-Archive -Path "asr.zip" -DestinationPath "." -Force
Remove-Item "asr.zip"
```

解压完成后目录应形如：

```
sherpa-onnx-probe\models\
└── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20\
    ├── tokens.txt
    ├── encoder-epoch-99-avg-1.int8.onnx
    ├── decoder-epoch-99-avg-1.onnx
    ├── joiner-epoch-99-avg-1.int8.onnx
    └── bpe.vocab
```

模型原始版权归 sherpa-onnx 项目所有。若需要 float32 精度或其他变体，可前往其官方仓库获取。

## 云端模式

在应用的“设置”中填写 SiliconFlow API Key。Key 保存于 Windows 的 `%APPDATA%\VoiceTextInput\settings.json`，不写入项目目录，因此不会随仓库泄露。

## 测试

```powershell
python -m unittest -v test_cloud_asr.py
```

测试覆盖云端请求构造、`text` 响应解析、取消结果、缺失响应文本和服务端错误信息。
