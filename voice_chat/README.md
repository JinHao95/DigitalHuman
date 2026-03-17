# 实时语音对话系统

基于 VAD + Whisper + OpenAI LLM + TTS 的低延迟语音交互系统，支持自动断句和播放中打断，运行于 macOS Apple Silicon。

## 技术架构

```text
麦克风 → sounddevice (16kHz)
       → webrtcvad (VAD 自动断句)
       → faster-whisper (ASR 语音识别，cpu + int8)
       → OpenAI API (LLM 对话生成，流式输出)
       → edge-tts / ChatTTS (TTS 语音合成)
       → sounddevice (播放，支持说话打断)
```

## 项目结构

```text
voice_chat/
├── main.py           # 主循环（入口）
├── vad_recorder.py   # VAD 自动断句录音器
├── asr.py            # faster-whisper 语音识别
├── llm.py            # OpenAI LLM 封装（流式）
├── tts.py            # ChatTTS + edge-tts 备用
├── audio_player.py   # 音频播放
├── check_mic.py      # 麦克风音量校准工具
├── requirements.txt  # 依赖列表
└── README.md
```

## 环境要求

- macOS 12+，**Apple Silicon (M1/M2/M3)**
- **必须使用原生 ARM64 Python**（x86_64 / Rosetta 版本功能异常，见下方说明）
- 麦克风权限：系统设置 → 隐私与安全性 → 麦克风 → 允许终端访问
- 网络连接（LLM API 调用；edge-tts 也需要联网）

## 快速开始

### 1. 安装原生 ARM64 Python

> **重要**：必须使用原生 ARM64 Python，用 Rosetta x86_64 Python 会导致 CoreAudio 行为异常。
> 通过 `/usr/local` 下的 Homebrew 安装的 Python 均为 x86_64，需要单独获取 ARM64 版本。

#### 方式 A：用 ARM64 Homebrew 安装（推荐，需 sudo）

```bash
# 安装 ARM64 Homebrew（安装到 /opt/homebrew）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 Python
/opt/homebrew/bin/brew install python@3.11

# 验证架构
file /opt/homebrew/opt/python@3.11/bin/python3.11   # 应显示 arm64
```

#### 方式 B：下载预编译独立包（无需 sudo）

```bash
curl -L "https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-aarch64-apple-darwin-install_only.tar.gz" \
  -o /tmp/python311-arm64.tar.gz
cd /tmp && tar xzf python311-arm64.tar.gz
cp -r python ~/.python311-arm64

# 验证
~/.python311-arm64/bin/python3.11 -c "import platform; print(platform.machine())"  # arm64
```

### 2. 创建 venv

```bash
cd voice_chat

# 方式 A（Homebrew）
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv

# 方式 B（独立包）
~/.python311-arm64/bin/python3.11 -m venv venv

source venv/bin/activate
python -c "import platform; print(platform.machine())"  # 确认输出 arm64
```

### 3. 安装依赖

```bash
# 基础依赖
pip install "numpy<2" scipy sounddevice soundfile webrtcvad-wheels
pip install faster-whisper openai edge-tts

# PyTorch（ARM64 MPS 加速版）
pip install torch==2.2.2 torchaudio==2.2.2
```

### 4. 设置环境变量

```bash
# LLM API（必填）
export OPENAI_API_KEY=sk-your-key-here

# 可选：兼容 OpenAI 格式的第三方接口（如 Qwen、DeepSeek）
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-turbo
```

### 5. 运行

```bash
python main.py
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | （必填） | OpenAI 或兼容接口的 API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 接入点 |
| `LLM_MODEL` | `gpt-4o-mini` | 使用的模型名称 |
| `WHISPER_MODEL` | `base` | Whisper 模型大小（tiny/base/small/medium/large-v3） |
| `WHISPER_LANGUAGE` | `zh` | ASR 语言（留空自动检测） |
| `WHISPER_DEVICE` | `cpu` | 推理设备（cpu / cuda） |
| `WHISPER_COMPUTE` | `int8` | 量化精度（int8 / float16） |
| `MIC_DEVICE` | `-1` | 麦克风设备 ID（-1 = 系统默认） |
| `CHATTTS_SPEAKER_SEED` | `2222` | ChatTTS 音色种子（固定 = 固定音色） |
| `CHATTTS_SPEED` | `5` | ChatTTS 语速 1~10 |
| `SYSTEM_PROMPT` | （内置） | LLM 系统提示词 |
| `LLM_MAX_HISTORY` | `10` | 保留最近 N 轮对话历史 |

## VAD 参数说明

`vad_recorder.py` 中的关键参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `SAMPLE_RATE` | 16000 | 采样率（webrtcvad 要求 8/16/32kHz） |
| `FRAME_DURATION` | 30ms | 帧长（10/20/30ms） |
| `VAD_AGGRESSIVENESS` | 3 | 灵敏度 0~3，越大过滤噪声越多 |
| `SILENCE_THRESHOLD` | 1.0s | 静音多久判断为句子结束 |
| `MIN_SPEECH_DURATION` | 0.5s | 最短有效语音，防误触发 |
| `MAX_RECORD_DURATION` | 30s | 最长单次录音，防卡死 |

## 打断（Barge-in）参数说明

`main.py` 中控制打断灵敏度的参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `_INTERRUPT_POLL_MS` | 200 | 麦克风检测间隔（ms） |
| `_INTERRUPT_NOISE_MS` | 600 | 播放前采底噪时长（ms） |
| `_INTERRUPT_SNRATIO` | 2.0 | RMS 超过底噪 N 倍才触发打断 |
| `_INTERRUPT_MIN_ABS` | 0.040 | 绝对最低打断阈值（防底噪误触） |

## TTS 说明

系统优先使用 **ChatTTS**（本地推理，自然中文女声，约 3GB 模型）。
若 ChatTTS 未安装，自动降级到 **edge-tts**（微软在线 TTS，需要联网）。

```bash
# 安装 ChatTTS（可选）
pip install chattts
```

## 麦克风校准工具

`check_mic.py` 可实时显示麦克风音量，用于排查打断阈值设置是否合理：

```bash
python check_mic.py
# 安静时看底噪 RMS，说话时看峰值 RMS
# 打断阈值（_INTERRUPT_MIN_ABS）应设在底噪和说话 RMS 之间
```

## 切换麦克风设备

```bash
# 列出所有音频设备
python -c "import sounddevice; print(sounddevice.query_devices())"

# 指定设备 ID（如 MacBook 内置麦克风为 2）
export MIC_DEVICE=2
python main.py
```

## 常见问题

### Q: 打断检测不生效 / 误触发

运行 `python check_mic.py` 测量你的麦克风底噪和说话 RMS，然后调整 `main.py` 中的 `_INTERRUPT_MIN_ABS` 和 `_INTERRUPT_SNRATIO`。

### Q: ASR 识别结果为空或乱码

尝试更大的模型：`export WHISPER_MODEL=small`；或确认 `WHISPER_LANGUAGE=zh`。

### Q: 想用其他 LLM（DeepSeek、Qwen 等）

设置兼容的 `OPENAI_BASE_URL`：

```bash
# 阿里云百炼（Qwen）
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-turbo

# DeepSeek
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
```

### Q: `torch` 与 `ctranslate2` 冲突导致程序崩溃

已在 `main.py` 顶部自动设置 `KMP_DUPLICATE_LIB_OK=TRUE`，无需手动处理。

### Q: sounddevice 找不到麦克风

系统设置 → 隐私与安全性 → 麦克风 → 允许终端（Terminal / iTerm）访问。
