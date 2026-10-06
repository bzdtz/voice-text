# ASR 模型选型评测

本项目的选型依据。结论不是"我查了文档觉得哪个好"，而是同一批音频、同一套指标下实测出来的。

复现：`python eval/compare.py` → 结果写入 `eval/report.md`。交互式对比：`python eval/gui.py`。

---

## 结论

| 模型 | 类型 | 加载体积 | 平均 RTF | 首字出现 | 平均相似度 | 全大写英文字符 |
|---|---|---|---|---|---|---|
| `baseline_zipformer_2023-02-20` | transducer | 198MB | **0.374** | 0.65s | 0.72 | **119** |
| `streaming_paraformer_bilingual` | paraformer | 237MB | 0.576 | 0.70s | 0.95 | 0 |
| `x-asr_punct_int8_480ms` | transducer | 169MB | 0.436 | 1.12s | **0.99** | 4 |

**选型：`x-asr_punct_int8_480ms`**

三个理由，按重要性排序：

1. **基线模型有系统性的英文大小写缺陷。** 119 个全大写字符不是噪声，是分布性的：`MONDAY`、`WIFI`、`HTTP TWO`、`WE NEED TO` 整段被输出成全大写。对一个"识别出来要粘到文档/消息框里"的产品来说这是**可用性问题**，不是精度问题——用户必须手工改回。
2. **体积反而更小。** x-asr 加载文件 169MB，基线 198MB，整目录 163MB vs 531MB。流式 ASR 常被默认认为"模型越大越准"，这里恰恰相反。
3. **自带标点。** x-asr 的 punct 变体内置标点，基线方案需要额外挂一个 CT-Transformer 标点恢复模型（`local_asr.py` 的 `OfflinePunctuator`），多一个模型、多一段失败降级逻辑。

代价：首字延迟 0.65s → 1.12s。对"按住说话、松开识别"的交互，这个代价可接受——**用户感知到的是松开后多久出字，不是说话过程中**。

## 方法论

### 测试集

6 段中英混合音频，`edge-tts` 合成，16kHz 单声道，总时长 35.06s。刻意覆盖了语音输入法最容易翻车的五类内容：

| 音频 | 考察点 |
|---|---|
| `s01_mixed_official` | 中英混排 + 数字字母混读（`zero eight two`） |
| `s02_digits_letters` | 字母逐个念（`a b c`）+ 数字读法（`twelve o two`） |
| `s03_tech` | 技术缩写（`api`、`http two`、`websocket`） |
| `s04_switch` | 句首英文切中文（`we need to 讨论一下`） |
| `s05_chinese` | 纯中文，作为对照 |
| `s06_english` | 纯英文，作为对照 |

**prompt 文本本身就是 ground truth**，所以不需要人工标注，也不存在标注一致性争议。

### 控制变量

- `num_threads=1`，排除多线程调度的机器差异
- `decoding_method="greedy_search"`，三个模型统一解码策略
- 每个模型先跑一次 warmup，再开始计时
- **`enable_endpoint_detection=False`** —— 这是最容易被忽略的一条。流式端点检测会在自然停顿处直接 finalize 输出，同一模型换一句可能就截在中间。**关掉才能测到模型本身的能力，而不是端点策略的副作用**

### 指标

- **RTF**：整段解码墙钟耗时 / 音频时长
- **首字延迟**：流式按 100ms 一块喂入，同时记录**墙上时钟耗时**和**已喂入的音频秒数**。后者才反映"用户说完多久看到字"的真实体感，前者会被机器性能污染
- **相似度**：`difflib.SequenceMatcher` 字符级 ratio，比对前统一小写、去掉标点和空格
- **全大写英文字符数**：`re.findall(r"[A-Z]{2,}")` 累计字符数。这是一个**专门为了抓大小写缺陷而设的指标**，字符相似度对大小写完全不敏感
- **部署体积**：只统计模型实际加载的文件（tokens / encoder / decoder / joiner），不含目录里的 `test_wavs` 等未加载文件

---

## 已知局限（面试会被问，这里先写清楚）

1. **测试集只有 6 段，且全是 TTS 合成音。** 不含真人录音、不含背景噪声、不含远场、不含长时对话。结论只适用于"同条件下的模型排序"，不能当绝对精度结论。
2. **用字符相似度而不是 WER/CER。** 对中英混排更宽松（不区分词边界），适合快速定位"哪个模型明显有问题"。局限是对漏字和多字不敏感，做严肃选型应该补 WER。
3. **单机器、单次测量。** 没有多次重复取分布，RTF 有波动。三者的量级差距足够大，结论不受单次波动影响。
4. **edge-tts 的合成音色固定**（`zh-CN-XiaoxiaoNeural`），只测了单一说话人。
5. **`x-asr_punct_int8_480ms` 是社区模型**，不是官方主线模型。它的准确率优势是否稳定、是否还在维护，需要持续关注。这是选型里的实际风险，不是已解决的问题。

## 落地状态

✅ **已落地。** `local_asr.py` 现在按 `CANDIDATE_MODELS` 的优先级选择模型，x-asr 排第一，基线 Zipformer 作为兜底——所以即使没下载 x-asr，应用也不会坏。

选型不是硬编码路径，而是**能力驱动**的：每个模型声明自己的文件布局和两项能力，加载器按声明决定挂什么、不挂什么。

| 能力 | x-asr（选中） | 基线 Zipformer（兜底） |
|---|---|---|
| `builtin_punctuation` | `True` → 不挂 CT-Transformer | `False` → 挂 `OfflinePunctuator` |
| `bpe_vocab` 文件 | 缺失 → 热词标记为不可用 | 存在 → 热词可用 |

### 为什么不直接删掉标点链路

计划里写过"删除 CT-Transformer 标点链路"，实际保留了。因为兜底模型需要它：如果 x-asr 没下载，应用回退到基线 Zipformer，那时没有标点阶段输出就没有标点。`OfflinePunctuator` / `resolve_punctuation_model` 仍是兜底路径的有效代码，`test_punctuation_adds_marks` 专门测它还在工作。

同理热词也没有删。x-asr 不带 `bpe.vocab`，而热词加权依赖它；保留代码但由配置判定不可用，UI 上如实显示"热词未启用"，而不是静默退化成 `greedy_search`。

### 验证

用 `eval/audio/` 的 6 段音频走**应用真实调用路径**（`StreamingZipformerASR` → `create_session()` → 100ms 分块 `accept()` → `finish()`）复测，不是重跑评测脚本：

| 指标 | 基线 | x-asr（应用实测） |
|---|---|---|
| 平均字符相似度 | 0.72 | **0.991** |
| 全大写英文字符总数 | 119 | **4** |
| 首字出现（音频位置） | 0.65s | 1.12s |

首字延迟与上文表格的 1.12s 完全一致，说明这条验证路径与评测脚本测的是同一件事。

顺带暴露了 x-asr 的两个真实瑕疵（不是评测问题）：标点符号后多一个空格（`房间， wifi`），以及中英文夹杂时偶发替换（`tomorrow 的 meeting` → `tomorrow the meeting`，该句相似度 0.952，其余 5 句 ≥0.991）。

### 落地过程中踩到的一个坑

`model_type="zipformer"` 会让 x-asr **直接崩溃**（onnxruntime 内 abort，退出码 255，没有 Python 异常可捕）。x-asr 的 encoder 缺少该路径所需的 `attention_dims` 图元数据，基线模型有，所以旧代码没事。

不传 `model_type`，让 sherpa-onnx 从图上推断架构，就能正常加载——这也正好和评测脚本一致（评测没传这个参数）。已在 `local_asr.py` 留注释说明原因，这是换模型时最容易踩的坑。
