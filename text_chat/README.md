# 文字对话系统

基于 OpenAI LLM + TTS 的文字输入版数字人，适用于直播弹幕/评论回复场景。

## 技术架构

```text
文字输入（命令行 / 弹幕 API）
       → OpenAI API（LLM 对话生成，流式输出）
       → edge-tts / ChatTTS（TTS 语音合成）
       → sounddevice（播放）
```

## 项目结构

```text
text_chat/
├── main.py           # 主循环（入口）
├── input_source.py   # 输入层抽象（CLI / 弹幕队列）
├── requirements.txt  # 依赖列表
└── README.md
```

复用 `voice_chat/` 中的模块：`llm.py`、`tts.py`、`audio_player.py`

## 快速开始

### 1. 使用 voice_chat 的 venv

```bash
cd text_chat
source ../voice_chat/venv/bin/activate
```

### 2. 设置环境变量

```bash
export OPENAI_API_KEY=sk-your-key-here

# 可选：第三方兼容接口
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export LLM_MODEL=qwen-turbo
```

### 3. 运行

```bash
python main.py
```

输入文字后回车，AI 用语音回复；输入 `exit` 退出。

## 环境变量

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | （必填） | API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 接入点 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名称 |
| `INPUT_MODE` | `cli` | 输入模式：`cli`（命令行）/ `danmu`（弹幕队列） |
| `BATCH_WINDOW_SEC` | `0` | 弹幕合并窗口秒数（0 = 不合并，逐条回复） |

其余参数（TTS、Whisper 等）与 voice_chat 相同，详见 [voice_chat/README.md](../voice_chat/README.md)。

## 弹幕模式接入

将 `INPUT_MODE=danmu` 后，通过 `DanmuInputSource.push()` 推入弹幕：

```python
from text_chat.input_source import DanmuInputSource

src = DanmuInputSource()
# 在 main.py 中传入 src，或修改 INPUT_MODE 后替换 src 实例

# 弹幕回调中调用：
src.push("主播好！")
src.push("今天讲什么？")
```

**弹幕回复策略：**
- `BATCH_WINDOW_SEC=0`（默认）：逐条回复，队列积压时跳过旧消息只回复最新一条
- `BATCH_WINDOW_SEC=3`：等待 3 秒收集积压弹幕，合并成一条统一回复
