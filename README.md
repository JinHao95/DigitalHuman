# DigitalHuman

数字人项目集合，目标是构建一套可用于真实直播、对话场景的低延迟数字人交互系统。
当前包含两个子项目：语音对话版（voice_chat）和文字输入版（text_chat），共享同一套 LLM / TTS / 播放内核。

---

## 整体架构

```text
                    ┌─────────────────────────────────────────┐
                    │            输入层（可替换）               │
                    │  麦克风 + VAD    /    文字 / 弹幕 API     │
                    └────────────────┬────────────────────────┘
                                     │ 文字
                    ┌────────────────▼────────────────────────┐
                    │         LLM（OpenAI 兼容接口）            │
                    │  流式输出，支持多轮对话历史，100字内口语回复  │
                    └────────────────┬────────────────────────┘
                                     │ 文字
                    ┌────────────────▼────────────────────────┐
                    │              TTS 语音合成                 │
                    │   ChatTTS（本地）→ 降级 edge-tts（在线）   │
                    └────────────────┬────────────────────────┘
                                     │ 音频
                    ┌────────────────▼────────────────────────┐
                    │          播放器（sounddevice）            │
                    │       支持非阻塞播放 + 随时打断             │
                    └─────────────────────────────────────────┘
```

---

## 技术方案

### 语音识别（ASR）

- 引擎：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)，OpenAI Whisper 的 CTranslate2 加速版
- 设备：CPU，量化：int8，模型默认 `base`（可切换 small / medium / large-v3）
- 语言：中文（`zh`），支持自动检测
- 延迟：base 模型在 M 系芯片上约 0.5~1s

### 语音活动检测（VAD）

- 引擎：[webrtcvad](https://github.com/wiseman/py-webrtcvad)，Google WebRTC 内置 VAD
- 采样率：16000 Hz，帧长：30ms
- 触发机制：滑动窗口 300ms，窗口内 ≥60% 帧判断为语音则触发录音
- 前置缓冲：触发时自动往前取 ~500ms 音频，确保句子开头不被截断
- 静音结束判断：连续静音 1 秒则视为句子结束

### 大语言模型（LLM）

- 接口：OpenAI Chat API 兼容格式，支持任意兼容服务（OpenAI / 阿里云百炼 Qwen / DeepSeek 等）
- 输出方式：流式（stream=True），逐 token 打印，降低首字延迟
- 对话历史：保留最近 10 轮，超出后自动滚动
- 提示词：性感活泼女性风格，口语化简短回复，并指导 LLM 在合适位置插入 ChatTTS 韵律标签

### 语音合成（TTS）

- 主选：[ChatTTS](https://github.com/2noise/ChatTTS)，本地推理，24kHz，约 3GB 模型
- 备选：[edge-tts](https://github.com/rany2/edge-tts)，微软在线 TTS，`zh-CN-XiaoxiaoNeural`，需联网
- 自动降级：ChatTTS 未安装时自动切换到 edge-tts，无需手动配置
- 音色：通过 `CHATTTS_SPEAKER_SEED` 固定音色（默认 4000，女声），可扫描种子批量试听
- 韵律控制：支持在文本中插入标签 `[uv_break]`（短停顿）、`[lbreak]`（长停顿）、`[laugh]`（笑声）、`[oral_0~9]`（口语化程度）；LLM 系统提示已配置为自动插入这些标签
- 推理速度：M 系芯片 CPU 推理，RTF ≈ 0.8~1.2（合成 1 秒音频约需 1 秒），GPT 核心不支持 MPS 加速
- `tts.py` 可独立运行，交互式输入文本测试合成效果并保存 wav 文件

### 音频播放与打断

- 播放：sounddevice，支持阻塞和非阻塞两种模式
- 打断机制（voice_chat）：全局共享麦克风流持续采样，检测 RMS 超过底噪 2 倍则立即 stop()
- 打断机制（text_chat）：CLI 输入在子线程中运行，主循环轮询消息队列，有新消息则立即 stop() 打断当前播放

### 运行环境

- 系统：macOS 12+，Apple Silicon（M1/M2/M3）
- Python：**必须使用原生 ARM64 Python 3.11**（Rosetta x86_64 版本会导致 CoreAudio 行为异常、麦克风黄灯不亮）
- 推荐安装方式：`/opt/homebrew/bin/brew install python@3.11`（ARM64 Homebrew）

---

## 子项目

### [voice_chat](./voice_chat/)

实时语音对话系统。用户对着麦克风说话，VAD 自动断句后送入 Whisper 识别，再经 LLM 生成回复，TTS 合成后播放。播放期间用户说话可立即打断。

**适用场景**：本地语音助手、语音交互原型

详见 [voice_chat/README.md](./voice_chat/README.md)。

### [text_chat](./text_chat/)

文字输入版数字人。输入层已抽象为独立模块，支持命令行交互和弹幕队列两种模式，LLM / TTS / 播放模块与 voice_chat 完全共享。

**适用场景**：直播弹幕/评论自动回复、文字驱动的数字人演示

详见 [text_chat/README.md](./text_chat/README.md)。

---

## 快速开始

```bash
# 1. 进入任意子项目
cd voice_chat   # 或 text_chat

# 2. 激活 ARM64 venv
source venv/bin/activate

# 3. 设置 API Key
export OPENAI_API_KEY=sk-your-key
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  # 可选
export LLM_MODEL=qwen-turbo  # 可选

# 4. 运行
python main.py
```

---

## 模块依赖关系

```text
voice_chat/
├── vad_recorder.py   ← 仅 voice_chat 使用
├── asr.py            ← 仅 voice_chat 使用
├── llm.py            ← voice_chat 和 text_chat 共用
├── tts.py            ← voice_chat 和 text_chat 共用
├── audio_player.py   ← voice_chat 和 text_chat 共用
└── main.py

text_chat/
├── input_source.py   ← 仅 text_chat 使用
└── main.py           ← 通过 sys.path 引用 voice_chat 的共享模块
```
