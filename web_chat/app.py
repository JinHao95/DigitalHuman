"""
web_chat/app.py
Web 版数字人：浏览器输入文字 → LLM → TTS → 浏览器播放音频

启动方式：
    export OPENAI_API_KEY=sk-...
    python app.py
然后在浏览器访问 http://<服务器IP>:5000
"""

import io
import os
import sys
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 复用 voice_chat 共享模块
_VOICE_CHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_chat")
sys.path.insert(0, os.path.abspath(_VOICE_CHAT_DIR))

from flask import Flask, request, jsonify, send_file, render_template_string
from llm import LLMClient
from tts import create_tts_engine
import scipy.io.wavfile as wav

app = Flask(__name__)

# 全局初始化（启动时加载一次，避免每次请求重载）
print("[初始化] 连接 LLM...")
llm = LLMClient()
print("[初始化] 加载 TTS 模型（首次较慢）...")
tts = create_tts_engine()
print("[初始化] GPU 预热中...")
tts.synthesize("你好")  # warm-up：消除首次推理的 GPU 冷启动开销
print("[初始化] 完成，服务就绪！")

# ─────────────────────── HTML 页面 ───────────────────────
HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>数字人对话</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f0f1a; color: #e0e0e0; height: 100vh; display: flex;
         flex-direction: column; align-items: center; padding: 24px; }
  h1 { color: #a78bfa; margin-bottom: 20px; font-size: 1.4rem; }
  #chat-box { width: 100%; max-width: 720px; flex: 1; overflow-y: auto;
              background: #1a1a2e; border-radius: 12px; padding: 16px;
              display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
  .msg { padding: 10px 14px; border-radius: 10px; max-width: 80%; line-height: 1.5; }
  .user { background: #4c1d95; align-self: flex-end; }
  .ai   { background: #1e3a5f; align-self: flex-start; }
  .sys  { color: #6b7280; font-size: 0.8rem; align-self: center; }
  #input-row { display: flex; width: 100%; max-width: 720px; gap: 10px; }
  #user-input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #374151;
                background: #1f2937; color: #e0e0e0; font-size: 1rem; outline: none; }
  #user-input:focus { border-color: #7c3aed; }
  #send-btn { padding: 12px 24px; background: #7c3aed; color: white; border: none;
              border-radius: 8px; cursor: pointer; font-size: 1rem; transition: background 0.2s; }
  #send-btn:hover { background: #6d28d9; }
  #send-btn:disabled { background: #374151; cursor: not-allowed; }
  audio { width: 100%; margin-top: 6px; border-radius: 6px; }
</style>
</head>
<body>
<h1>🤖 数字人对话</h1>
<div id="chat-box"></div>
<div id="input-row">
  <input id="user-input" type="text" placeholder="输入消息，回车发送..." autofocus />
  <button id="send-btn" onclick="sendMsg()">发送</button>
</div>

<script>
const chatBox = document.getElementById('chat-box');
const input   = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

function addMsg(role, text, audioUrl) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  if (audioUrl) {
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.autoplay = true;
    audio.src = audioUrl;
    div.appendChild(audio);
  }
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addSys(text) {
  const div = document.createElement('div');
  div.className = 'msg sys';
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

async function sendMsg() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;

  addMsg('user', text);
  const status = addSys('AI 思考中...');

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    status.remove();

    if (data.error) {
      addSys('❌ 错误：' + data.error);
    } else {
      addMsg('ai', data.ai_text, data.audio_url);
      const info = `LLM ${data.llm_time}s | TTS ${data.tts_time}s | 音频 ${data.audio_dur}s`;
      addSys(info);
    }
  } catch (e) {
    status.textContent = '❌ 请求失败：' + e.message;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}
</script>
</body>
</html>
"""

# ─────────────────────── API ───────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "消息为空"}), 400

    import time

    # LLM
    t0 = time.time()
    try:
        ai_text = "".join(llm.chat_stream(text))
    except Exception as e:
        return jsonify({"error": f"LLM 错误: {e}"}), 500
    llm_time = round(time.time() - t0, 2)

    # TTS
    t1 = time.time()
    try:
        audio = tts.synthesize(ai_text)
    except Exception as e:
        return jsonify({"error": f"TTS 错误: {e}"}), 500
    tts_time = round(time.time() - t1, 2)
    audio_dur = round(len(audio) / tts.sample_rate, 1)

    # 转为 WAV bytes
    buf = io.BytesIO()
    wav.write(buf, tts.sample_rate, (audio * 32767).astype(np.int16))
    buf.seek(0)

    # 把音频存临时文件，通过 /audio 接口返回
    audio_path = os.path.join(os.path.dirname(__file__), "_last_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(buf.read())

    return jsonify({
        "ai_text":   ai_text,
        "audio_url": "/audio",
        "llm_time":  llm_time,
        "tts_time":  tts_time,
        "audio_dur": audio_dur,
    })


@app.route("/audio")
def audio():
    audio_path = os.path.join(os.path.dirname(__file__), "_last_audio.wav")
    if not os.path.exists(audio_path):
        return "Not found", 404
    return send_file(audio_path, mimetype="audio/wav")


@app.route("/seed_samples/<filename>")
def seed_sample(filename):
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "voice_chat", "seed_samples")
    return send_file(os.path.join(sample_dir, filename), mimetype="audio/wav")


@app.route("/voices")
def voices():
    seeds = [42, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]
    items = "".join(f"""
    <div style="background:#1a1a2e;border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:16px">
      <span style="color:#a78bfa;font-weight:bold;min-width:80px">Seed {s}</span>
      <audio controls src="/seed_samples/seed_{s}.wav" style="flex:1;height:32px"></audio>
      <span style="color:#6b7280;font-size:0.85rem">点播放试听</span>
    </div>""" for s in seeds)
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<title>音色试听</title>
<style>body{{background:#0f0f1a;color:#e0e0e0;font-family:-apple-system,sans-serif;
max-width:700px;margin:40px auto;padding:20px}}
h1{{color:#a78bfa;margin-bottom:8px}}
p{{color:#6b7280;margin-bottom:24px}}
.list{{display:flex;flex-direction:column;gap:10px}}</style>
</head><body>
<h1>🎙 音色试听</h1>
<p>听完后把你喜欢的 Seed 编号告诉我，我帮你设置上。</p>
<div class="list">{items}</div>
</body></html>"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5401"))
    print(f"\n✅ 在浏览器访问：http://<服务器IP>:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
