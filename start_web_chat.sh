#!/bin/bash
# 数字人 Web 版启动脚本
# 启动后在浏览器访问 http://<服务器IP>:5401

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-b3c67260859e45deb54989a1dca58c95}"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"

cd "$(dirname "$0")/web_chat"
exec "$HOME/DigitalHuman/voice_chat/venv/bin/python" app.py "$@"
