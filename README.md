# 语音输入（本地 / 云端双模式）

按住 `Ctrl + Space` 录音，松开后识别整段音频；结果自动复制到剪贴板，并显示在主窗口和可编辑浮窗中。

- `local` 模式：本地推理，全程离线。模型为 sherpa-onnx 的 Streaming Zipformer（中英双语），可选挂 CT-Transformer 做标点恢复。
- `cloud` 模式：整段上传到 SiliconFlow，接口 `POST https://api.siliconflow.cn/v1/audio/transcriptions`，模型 `FunAudioLLM/SenseVoiceSmall`。

Python / Tk 桌面程序，录音基于 `sounddevice` 单声道实现，不依赖浏览器 `getUserMedia` / `MediaRecorder`，云端模式也无需 ffmpeg 或本地 ASR 进程。

---

## 为什么有本地和云端两种模式

云端 SenseVoiceSmall 是**整段上传后一次性返回**，没有流式 partial——用户按住说话的过程里看不到任何字。本地模型可以做到边说边出字，而且全程不出本机。

所以两者不是冗余，是分工：**本地是主路径，云端是本地模型不可用时的兜底。**

## 模型选型

本地模型不是"哪个新用哪个"，是实测出来的。见 [docs/ASR模型选型评测.md](docs/ASR模型选型评测.md)。

结论摘要（6 段中英混合音频，`num_threads=1`，greedy_search）：

| 模型 | 加载体积 | RTF | 首字延迟 | 相似度 | 全大写英文字符 |
|---|---|---|---|---|---|
| `baseline_zipformer_2023-02-20` | 198MB | **0.374** | 0.65s | 0.72 | **119** |
| `streaming_paraformer_bilingual` | 237MB | 0.576 | 0.70s | 0.95 | 0 |
| `x-asr_punct_int8_480ms` | 169MB | 0.436 | 1.12s | **0.99** | 4 |

评测选出 `x-asr_punct_int8_480ms`，但**这个结论还没落到代码里**——`local_asr.py` 目前仍指向表里表现最差的基线模型。原因和待办写在评测文档最后一节。

---

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

模型原始版权归 sherpa-onnx 项目所有。

## 云端模式

在应用的"设置"中填写 SiliconFlow API Key。**Key 保存于 Windows 的 `%APPDATA%\VoiceTextInput\settings.json`，不写入项目目录**，因此不会随仓库泄露。

## 模型评测

`eval/` 下是一套可复现的选型评测：

```powershell
pip install -r eval\requirements.txt
python eval\compare.py     # 跑完把结果写到 eval\report.md
python eval\gui.py         # 三个模型实时切换对比
```

- `eval/make_audio.py` 生成评测集（edge-tts 合成，prompt 文本即 ground truth）
- `eval/audio/` 已包含生成好的 6 段音频，不需要重新合成
- 模型体积较大，默认从 `sherpa-onnx-probe\models\` 找，可用环境变量 `VOICE_TEXT_EVAL_MODELS` 指向别处；缺哪个模型就跳过哪个，不会中断

评测方法论、控制变量、已知局限写在 [docs/ASR模型选型评测.md](docs/ASR模型选型评测.md)。

## 测试

```powershell
python -m unittest -v test_cloud_asr.py
```

测试覆盖云端请求构造、`text` 响应解析、取消结果、缺失响应文本和服务端错误信息。

## Roadmap

- [ ] **落地选型结论**：把本地模型切换到评测选出的 `x-asr_punct_int8_480ms`（`local_asr.py` 当前仍指向基线 Zipformer），并移除不再需要的 CT-Transformer 标点恢复链路
- [ ] **端到端延迟测量**：评测测的是模型 RTF，还需要测「松开热键 → 结果上屏」的用户体感延迟
- [ ] **扩充评测集**：当前 6 段均为 TTS 合成音频，计划加入真人录音、背景噪声和长时音频，并补 WER/CER
- [ ] **云端模式成本表**：给出单次调用的实际成本，与本地模式对比
- [ ] **浏览器级交互测试**：现有单测覆盖云端请求链路，本地模型部分待补

评测的方法论、控制变量和已知局限完整记录在 [docs/ASR模型选型评测.md](docs/ASR模型选型评测.md)。

