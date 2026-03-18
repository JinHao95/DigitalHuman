# DigitalHuman

数字人相关项目集合。

## 子项目

### [voice_chat](./voice_chat/)

实时语音对话系统，基于 VAD + Whisper + OpenAI LLM + TTS，支持自动断句和播放中打断，运行于 macOS Apple Silicon。

详见 [voice_chat/README.md](./voice_chat/README.md)。

### [text_chat](./text_chat/)

文字输入版数字人，适用于直播弹幕/评论回复场景。输入层已抽象，支持命令行交互和弹幕队列两种模式，复用 voice_chat 的 LLM / TTS / 播放模块。

详见 [text_chat/README.md](./text_chat/README.md)。
