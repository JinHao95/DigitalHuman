#!/bin/bash
# 数字人 Web 版启动脚本
# 启动后在浏览器访问 http://<服务器IP>:5401

export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="${OPENAI_API_KEY:-ed0df4a7-c765-4c78-8576-ce3701f4dca1}"
export OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export LLM_MODEL="doubao-1-5-pro-32k-250115"

cd "$(dirname "$0")/web_chat"
exec "$HOME/DigitalHuman/voice_chat/venv/bin/python" app.py "$@"
