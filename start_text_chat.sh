#!/bin/bash
# 数字人 text_chat 启动脚本

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# LLM 配置（运行前请填写）
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export LLM_MODEL="qwen-plus"

cd "$(dirname "$0")/text_chat"
exec "$HOME/DigitalHuman/voice_chat/venv/bin/python" main.py "$@"
