# 语音输入（本地 / 云端双模式）

按住 `Ctrl + Space` 录音，松开后识别整段音频；结果自动复制到剪贴板，并显示在主窗口和可编辑浮窗中。

- `local` 模式：本地推理，全程离线。默认模型为 [X-ASR](https://github.com/Gilgamesh-J/X-ASR) 流式 Zipformer 中英双语（内置标点恢复），未下载时自动回退到 sherpa-onnx 官方基线模型并挂 CT-Transformer 补标点。
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

评测选出的 `x-asr_punct_int8_480ms` 已落地到代码：`local_asr.py` 按 `CANDIDATE_MODELS` 优先级加载，x-asr 排第一，基线 Zipformer 作兜底，所以没下载 x-asr 也不会坏。

模型能力是**声明式**的，不是硬编码路径——每个模型声明自己的文件布局和两项能力，加载器据此决定挂什么：

| 能力 | x-asr（默认） | 基线 Zipformer（兜底） |
|---|---|---|
| `builtin_punctuation` | `True`，跳过 CT-Transformer | `False`，挂 `OfflinePunctuator` |
| `bpe_vocab` | 缺失 → 热词如实标记为不可用 | 存在 → 热词可用 |

热词和标点链路都保留了：它们是兜底路径的有效代码，而不是死代码。落地细节和验证数据见 [docs/ASR模型选型评测.md](docs/ASR模型选型评测.md) 的「落地状态」一节。

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

解压到项目根的 `sherpa-onnx-probe\models\` 下即可，**解压后的目录名必须保持不变**（加载器按目录名匹配模型配置）。

| 模型目录 | 用途 | 来源 |
| --- | --- | --- |
| `sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05` | **默认主模型**，内置标点 | [Gilgamesh-J/X-ASR](https://github.com/Gilgamesh-J/X-ASR)（Apache-2.0） |
| `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` | 兜底主模型，无内置标点 | 本仓库 [Releases v1.0](https://github.com/bzdtz/voice-text/releases/tag/v1.0) 的 `models-streaming-bilingual-zh-en.zip` |
| `sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8` | 仅为兜底模型补标点 | 同上，`models-punctuation-ct-transformer.zip` |

只下第一个模型即可，应用会以完整功能运行。只下第二个时也能跑，但准确度回到基线水平（评测见上文），且需要标点模型才有关联标点。

下载兜底模型：

```powershell
cd sherpa-onnx-probe\models
Invoke-WebRequest -Uri "https://github.com/bzdtz/voice-text/releases/download/v1.0/models-streaming-bilingual-zh-en.zip" -OutFile "asr.zip"
Expand-Archive -Path "asr.zip" -DestinationPath "." -Force
Remove-Item "asr.zip"
```

默认模型目录应形如：

```
sherpa-onnx-probe\models\
└── sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05\
    ├── tokens.txt
    ├── encoder.int8.onnx
    ├── decoder.onnx
    ├── joiner.int8.onnx
    └── bpe.model
```

注意：该模型**没有 `bpe.vocab`**，所以热词加权对它不可用，应用会如实显示"热词未启用"。这是模型本身的限制，不是配置错误。

## 模型版权

| 模型 | 许可 | 出处 |
| --- | --- | --- |
| X-ASR（默认） | Apache-2.0 | [Gilgamesh-J/X-ASR](https://github.com/Gilgamesh-J/X-ASR) |
| Streaming Zipformer 双语基线 | 见模型仓库 LICENSE | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) |
| CT-Transformer 标点 | 见模型仓库 LICENSE | 同上 |

本仓库不含模型权重，均由用户自行下载。使用 Apache-2.0 模型需保留其版权声明与许可文本。

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
python -m unittest -v test_cloud_asr.py test_local_asr.py test_audio_processing.py
```

- `test_cloud_asr.py`：云端请求构造、`text` 响应解析、取消结果、缺失响应文本和服务端错误信息
- `test_local_asr.py`：模型解析优先级、能力声明（标点 / 热词）、静音解码、标点阶段在回退路径上仍然可用
- `test_audio_processing.py`：音频重采样与削波处理

`test_local_asr.py` 依赖真实模型文件，缺失时会失败——这是有意的，避免"没下载模型也能测过"的假绿灯。

## Roadmap

- [x] **落地选型结论**：本地默认模型已切换为评测选出的 `x-asr_punct_int8_480ms`，基线 Zipformer 保留为兜底（见上文「模型选型」）
- [ ] **端到端延迟测量**：已测得首字出现位置（音频位置 1.12s），但还缺「松开热键 → 结果上屏」的完整链路测量，包括 UI 线程回切
- [ ] **修 x-asr 的中文标点后空格**：实测输出形如 `房间， wifi`，中文逗号后多一个空格，需要一个小后处理
- [ ] **扩充评测集**：当前 6 段均为 TTS 合成音频，计划加入真人录音、背景噪声和长时音频，并补 WER/CER
- [ ] **云端模式成本表**：给出单次调用的实际成本，与本地模式对比
- [ ] **评估端点检测对输入法的适配**：流式端点检测会在自然停顿处截断，输入法语境下用户中途停顿思考就会导致文本提前提交，需要确认开关策略

评测的方法论、控制变量和已知局限完整记录在 [docs/ASR模型选型评测.md](docs/ASR模型选型评测.md)。

