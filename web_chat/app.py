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

_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────── 工具函数 ───────────────────────

def _synth_segments(segments: list) -> list:
    """
    将文本段列表逐段合成为音频文件，返回对应的 URL 路径列表。
    文件存储为 _seg_0.wav, _seg_1.wav ...
    """
    urls = []
    idx = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        audio = tts.synthesize(seg)
        path = os.path.join(_DIR, f"_seg_{idx}.wav")
        buf = io.BytesIO()
        wav.write(buf, tts.sample_rate, (audio * 32767).astype(np.int16))
        with open(path, "wb") as f:
            f.write(buf.getvalue())
        urls.append(f"/audio/{idx}")
        idx += 1
    return urls


def _split_segments(text: str) -> list:
    """按换行符拆分文本段，过滤空行。"""
    return [s.strip() for s in text.split("\n") if s.strip()]


# ─────────────────────── HTML 页面 ───────────────────────
HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>直播陪聊</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f0f1a; color: #e0e0e0; height: 100vh; display: flex;
         flex-direction: column; align-items: center; padding: 16px 24px; }
  header { width: 100%; max-width: 720px; display: flex; align-items: center;
           justify-content: space-between; margin-bottom: 14px; }
  .title { color: #a78bfa; font-size: 1.3rem; font-weight: bold; }
  .nickname-wrap { display: flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #9ca3af; }
  #nick-display { color: #f9a8d4; font-weight: bold; cursor: pointer;
                  border-bottom: 1px dashed #f9a8d4; padding-bottom: 1px; }
  #nick-input { display: none; background: #1f2937; border: 1px solid #7c3aed;
                color: #e0e0e0; border-radius: 6px; padding: 3px 8px; font-size: 0.9rem; outline: none; }
  #chat-box { width: 100%; max-width: 720px; flex: 1; overflow-y: auto;
              background: #1a1a2e; border-radius: 12px; padding: 16px;
              display: flex; flex-direction: column; gap: 10px; margin-bottom: 14px; }
  .msg { padding: 10px 14px; border-radius: 10px; max-width: 82%; line-height: 1.6; word-break: break-word; }
  .user { background: #4c1d95; align-self: flex-end; }
  .user-label { font-size: 0.78rem; color: #c4b5fd; display: block; margin-bottom: 3px; }
  .ai   { background: #1e3a5f; align-self: flex-start; }
  .ai-label { font-size: 0.78rem; color: #7dd3fc; display: block; margin-bottom: 3px; }
  .sys  { color: #6b7280; font-size: 0.78rem; align-self: center; text-align: center; }
  #input-row { display: flex; width: 100%; max-width: 720px; gap: 10px; }
  #user-input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #374151;
                background: #1f2937; color: #e0e0e0; font-size: 1rem; outline: none; }
  #user-input:focus { border-color: #7c3aed; }
  #send-btn { padding: 12px 24px; background: #7c3aed; color: white; border: none;
              border-radius: 8px; cursor: pointer; font-size: 1rem; transition: background 0.2s; }
  #send-btn:hover { background: #6d28d9; }
  #send-btn:disabled { background: #374151; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="title">🎙 小晴的直播间</div>
  <div class="nickname-wrap">
    <span>我的昵称：</span>
    <span id="nick-display" title="点击修改昵称"></span>
    <input id="nick-input" type="text" maxlength="12" placeholder="输入昵称回车确认" />
  </div>
</header>
<div id="chat-box"></div>
<div id="input-row">
  <input id="user-input" type="text" placeholder="说点什么吧..." autofocus />
  <button id="send-btn" onclick="sendMsg()">发送</button>
</div>

<script>
// ── 昵称管理 ──
const PREFIXES = ['小可爱', '宝贝', '亲爱的', '小甜心', '小心肝', '小宝贝'];
function genNick() {
  const p = PREFIXES[Math.floor(Math.random() * PREFIXES.length)];
  const n = String(Math.floor(Math.random() * 900) + 100);
  return p + n;
}
let username = localStorage.getItem('dh_nick') || genNick();
localStorage.setItem('dh_nick', username);

const nickDisplay = document.getElementById('nick-display');
const nickInput   = document.getElementById('nick-input');
nickDisplay.textContent = username;

nickDisplay.addEventListener('click', () => {
  nickInput.value = username;
  nickDisplay.style.display = 'none';
  nickInput.style.display = 'inline-block';
  nickInput.focus();
});
nickInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') confirmNick();
});
nickInput.addEventListener('blur', confirmNick);
function confirmNick() {
  const v = nickInput.value.trim();
  if (v) { username = v; localStorage.setItem('dh_nick', username); }
  nickDisplay.textContent = username;
  nickDisplay.style.display = 'inline';
  nickInput.style.display = 'none';
}

// ── 聊天渲染 ──
const chatBox = document.getElementById('chat-box');
const input   = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

function addMsg(role, text, nick) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const label = document.createElement('span');
  if (role === 'user') {
    label.className = 'user-label';
    label.textContent = (nick || username) + '：';
  } else {
    label.className = 'ai-label';
    label.textContent = '🎙 小晴：';
  }
  const content = document.createElement('span');
  content.textContent = text;
  div.appendChild(label);
  div.appendChild(content);
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}
function addSys(text) {
  const div = document.createElement('div');
  div.className = 'msg sys';
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

// ── 音频队列逐段播放 ──
let audioQueue = [];
let isPlaying = false;
let doneCallback = null;

function playQueue(urls, onDone) {
  audioQueue = [...urls];
  isPlaying = true;
  doneCallback = onDone || null;
  playNext();
}
function playNext() {
  if (audioQueue.length === 0) {
    isPlaying = false;
    resetIdleTimer();
    if (doneCallback) { doneCallback(); doneCallback = null; }
    return;
  }
  const url = audioQueue.shift() + '?t=' + Date.now();
  const audio = new Audio(url);
  audio.addEventListener('ended', playNext);
  audio.addEventListener('error', playNext);
  audio.play().catch(playNext);
}

// ── 冷场检测 ──
let idleTimer = null;
const IDLE_SEC = 20;

function resetIdleTimer() {
  clearTimeout(idleTimer);
  if (isBusy || isPlaying) return;
  idleTimer = setTimeout(triggerIdle, IDLE_SEC * 1000);
}
async function triggerIdle() {
  if (isBusy || isPlaying) { resetIdleTimer(); return; }
  try {
    const res = await fetch('/idle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username })
    });
    const data = await res.json();
    if (data.segments) {
      data.segments.forEach(s => addMsg('ai', s));
      playQueue(data.audio_urls);
    }
  } catch(e) {}
}

// ── 发送消息 ──
let isBusy = false;

async function sendMsg() {
  const text = input.value.trim();
  if (!text || isBusy) return;
  input.value = '';
  isBusy = true;
  sendBtn.disabled = true;
  clearTimeout(idleTimer);

  addMsg('user', text);
  const status = addSys('小晴思考中...');

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, username })
    });
    const data = await res.json();
    status.remove();

    if (data.error) {
      addSys('❌ ' + data.error);
    } else {
      data.segments.forEach(s => addMsg('ai', s));
      playQueue(data.audio_urls);
    }
  } catch (e) {
    status.textContent = '❌ 请求失败：' + e.message;
  } finally {
    isBusy = false;
    sendBtn.disabled = false;
    input.focus();
    resetIdleTimer();
  }
}

// ── 进入直播间 ──
async function onEnter() {
  isBusy = true;
  clearTimeout(idleTimer);
  try {
    const res = await fetch('/enter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username })
    });
    const data = await res.json();
    if (data.segments) {
      data.segments.forEach(s => addMsg('ai', s));
      playQueue(data.audio_urls);
    }
  } catch(e) {}
  finally {
    isBusy = false;
    resetIdleTimer();
  }
}

// 页面加载完成后触发进入事件
window.addEventListener('load', () => setTimeout(onEnter, 500));
</script>
</body>
</html>
"""

# ─────────────────────── 公共处理函数 ───────────────────────

def _handle_request(prompt_text):
    """LLM → 按段 TTS → 返回 segments + audio_urls。"""
    import time
    t0 = time.time()
    try:
        ai_text = "".join(llm.chat_stream(prompt_text))
    except Exception as e:
        return {"error": f"LLM 错误: {e}"}
    llm_time = round(time.time() - t0, 2)

    segments = _split_segments(ai_text)
    if not segments:
        segments = [ai_text.strip()] if ai_text.strip() else ["嗯～"]

    t1 = time.time()
    try:
        audio_urls = _synth_segments(segments)
    except Exception as e:
        return {"error": f"TTS 错误: {e}"}
    tts_time = round(time.time() - t1, 2)

    return {
        "segments":   segments,
        "audio_urls": audio_urls,
        "llm_time":   llm_time,
        "tts_time":   tts_time,
    }


# ─────────────────────── API ───────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    text     = (data or {}).get("text", "").strip()
    username = (data or {}).get("username", "宝贝").strip() or "宝贝"
    if not text:
        return jsonify({"error": "消息为空"}), 400

    prompt = f"[{username}说]：{text}"
    result = _handle_request(prompt)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route("/enter", methods=["POST"])
def enter():
    data = request.get_json()
    username = (data or {}).get("username", "宝贝").strip() or "宝贝"
    prompt = f"[系统通知]：用户「{username}」刚刚进入了直播间，请主动欢迎他，要热情、带情绪、自然活泼。"
    result = _handle_request(prompt)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route("/idle", methods=["POST"])
def idle():
    data = request.get_json()
    username = (data or {}).get("username", "").strip()
    if username:
        prompt = f"[系统通知]：直播间已经有一段时间没人说话了，「{username}」还在线，请主动发起一个轻松有趣的话题，吸引他继续聊天。"
    else:
        prompt = "[系统通知]：直播间已经有一段时间没人说话了，请主动发起一个轻松有趣的话题，营造气氛。"
    result = _handle_request(prompt)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route("/audio/<int:idx>")
def audio(idx):
    path = os.path.join(_DIR, f"_seg_{idx}.wav")
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="audio/wav")


@app.route("/seed_samples/<filename>")
def seed_sample(filename):
    sample_dir = os.path.join(_DIR, "..", "voice_chat", "seed_samples")
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
