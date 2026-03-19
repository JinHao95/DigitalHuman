"""
web_chat/app.py
Web 版数字人：浏览器输入文字 → LLM → TTS → 浏览器播放音频

架构：多人广播模式
  每个用户持有一条 GET /events SSE 长连接（广播通道）
  用户发言 POST /stream → 入队 _reply_queue
  _reply_worker 单线程串行处理：LLM → 切段 → TTS → broadcast 给所有在线用户
  所有用户都能看到和听到主播的回复
"""

import io
import os
import sys
import json
import uuid
import time
import queue
import threading
import logging
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_VOICE_CHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_chat")
sys.path.insert(0, os.path.abspath(_VOICE_CHAT_DIR))

from flask import Flask, request, jsonify, send_file, render_template_string, Response, stream_with_context
from llm import LLMClient
from tts import create_tts_engine, TTSEngine
import scipy.io.wavfile as wav

app = Flask(__name__)

# ─────────────────────── 运行日志 ───────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_DIR = os.path.join(_DIR, "audio_cache")
_LOG_DIR   = os.path.join(_DIR, "logs")
os.makedirs(_AUDIO_DIR, exist_ok=True)
os.makedirs(_LOG_DIR,   exist_ok=True)

_run_log_path = os.path.join(_LOG_DIR, "app.log")
_handler = logging.FileHandler(_run_log_path, encoding="utf-8")
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_stream_handler)

logger.info("=" * 60)
logger.info("服务启动")
logger.info("=" * 60)

logger.info("连接 LLM...")
llm = LLMClient()
logger.info("LLM 连接完成")
logger.info("加载 TTS 模型（首次较慢）...")
tts = create_tts_engine()
logger.info("GPU 预热中...")
_t0 = time.time()
tts.synthesize("你好")
logger.info(f"TTS 预热完成，耗时 {time.time()-_t0:.1f}s，服务就绪")

# TTS 线程池：GPU 串行资源，单线程池
_tts_pool = ThreadPoolExecutor(max_workers=1)


# ─────────────────────── 广播管理 ───────────────────────

# username → list of {"q": Queue, "dead": bool}
# 每条 tab 连接对应一个 entry，dead=True 表示连接已断
_sse_clients: dict[str, list] = {}
_sse_lock = threading.Lock()


class _Client:
    """代表一条 SSE 长连接。"""
    def __init__(self, client_id: str, username: str):
        self.client_id = client_id
        self.username  = username
        self.q    = queue.Queue(maxsize=100)
        self.dead = False


# client_id → _Client，精确管理每条连接
_sse_clients: dict[str, "_Client"] = {}
_sse_lock = threading.Lock()


def _register_client(client_id: str, username: str) -> "_Client":
    c = _Client(client_id, username)
    with _sse_lock:
        _sse_clients[client_id] = c
    logger.info(f"[SSE] 注册 {username}({client_id})，在线: {online_count()}")
    return c


def _unregister_client(client_id: str):
    with _sse_lock:
        c = _sse_clients.pop(client_id, None)
    if c:
        c.dead = True
        logger.info(f"[SSE] 注销 {c.username}({client_id})，在线: {online_count()}")


def broadcast(event: dict):
    """向所有在线连接广播，写入失败的连接自动清理。"""
    msg = _sse(event)
    dead_ids = []
    with _sse_lock:
        snapshot = list(_sse_clients.items())
    for cid, c in snapshot:
        if c.dead:
            dead_ids.append(cid)
            continue
        try:
            c.q.put_nowait(msg)
        except queue.Full:
            logger.warning(f"[SSE] {c.username}({cid}) 队列满，标记清理")
            dead_ids.append(cid)
    for cid in dead_ids:
        _unregister_client(cid)


def online_count() -> int:
    with _sse_lock:
        return len(_sse_clients)


# ─────────────────────── 关系阶段 ───────────────────────

_user_state: dict[str, int] = {}

_RELATION_STAGES = [
    (0,  "陌生"),
    (3,  "有点熟"),
    (8,  "熟悉"),
    (15, "有点暧昧"),
    (25, "偏爱"),
]

def _get_relation(username: str) -> str:
    n = _user_state.get(username, 0)
    stage = _RELATION_STAGES[0][1]
    for threshold, label in _RELATION_STAGES:
        if n >= threshold:
            stage = label
    return stage

def _inc_user_turns(username: str):
    _user_state[username] = _user_state.get(username, 0) + 1


# ─────────────────────── 对话日志 ───────────────────────

def _log(role: str, username: str, text: str):
    today = datetime.date.today().isoformat()
    path  = os.path.join(_LOG_DIR, f"{today}.jsonl")
    record = {
        "ts":       datetime.datetime.now().isoformat(timespec="seconds"),
        "role":     role,
        "username": username,
        "text":     text,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _next_idle_prompt(username: str) -> str:
    return (
        f"[系统通知]：直播间有一段时间没人说话了，「{username}」还在线。"
        "请你自己想一件最近发生的小事或者你的日常，随口分享出来，"
        "说话要自然、口语、慢一点，就像在跟朋友聊天，不要太正式，不要总结，不要升华。"
    )


# ─────────────────────── 工具 ───────────────────────

def _synth_to_file(text: str, path: str) -> bool:
    t0 = time.time()
    try:
        audio = tts.synthesize(text)
        buf = io.BytesIO()
        wav.write(buf, tts.sample_rate, (audio * 32767).astype(np.int16))
        with open(path, "wb") as f:
            f.write(buf.getvalue())
        logger.debug(f"[TTS] 合成完成 {time.time()-t0:.2f}s | {os.path.basename(path)} | {text[:30]!r}")
        return True
    except Exception as e:
        logger.error(f"[TTS] 合成失败 {time.time()-t0:.2f}s | {os.path.basename(path)} | {e}")
        return False


def _split_segments(text: str) -> list:
    return [s.strip() for s in text.split("\n") if s.strip()]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─────────────────────── 消息队列 + worker ───────────────────────

# 每条 item 结构：
# { "username": str, "prompt": str, "log_role": str, "display_text": str or None }
_reply_queue: queue.Queue = queue.Queue()


def _stream_and_broadcast(item: dict):
    """
    在 _reply_worker 线程中同步执行：
    LLM 流式 → 切段 → TTS 线程池 → broadcast 给所有在线用户
    """
    import re as _re
    _TAG_ONLY = _re.compile(r'^(\[[\w_]+\]|…+|～+|\s)*$')

    username    = item["username"]
    prompt_text = item["prompt"]
    log_role    = item["log_role"]
    session_id  = uuid.uuid4().hex[:8]
    req_start   = time.time()

    logger.info(f"[REQ {session_id}] 开始 | user={username} | role={log_role} | prompt={prompt_text[:60]!r}")

    buffer     = ""
    seg_idx    = 0
    pending    = []
    all_segs   = []
    carry_over = ""

    def submit_seg(text, idx):
        logger.debug(f"[REQ {session_id}] 提交TTS seg{idx} | {text[:40]!r}")
        fname = f"{session_id}_{idx}.wav"
        fpath = os.path.join(_AUDIO_DIR, fname)
        fut   = _tts_pool.submit(_synth_to_file, text, fpath)
        pending.append((idx, text, fname, fut))
        all_segs.append(text)

    def try_emit(line):
        nonlocal seg_idx, carry_over
        if not line:
            return
        if _TAG_ONLY.match(line):
            carry_over += line
            logger.debug(f"[REQ {session_id}] 暂存纯标签 {line!r}")
        else:
            full = (carry_over + line) if carry_over else line
            carry_over = ""
            submit_seg(full, seg_idx)
            seg_idx += 1

    def drain_pending(wait: bool):
        i = 0
        while i < len(pending):
            idx, text, fname, fut = pending[i]
            if i == 0 or wait:
                ok = fut.result()
            else:
                if not fut.done():
                    break
                ok = fut.result()
            if ok:
                broadcast({"type": "seg", "idx": idx, "text": text,
                           "url": f"/audio_cache/{fname}"})
            i += 1
        del pending[:i]

    # 通知前端：主播开始回复
    broadcast({"type": "ai_speaking", "value": True})

    try:
        for chunk in llm.chat_stream(prompt_text):
            buffer += chunk
            while True:
                nl = buffer.find("\n")
                punct_pos = -1
                for p in ("。", "！", "？", "～", "…"):
                    pos = buffer.find(p)
                    if pos != -1 and (punct_pos == -1 or pos < punct_pos):
                        punct_pos = pos
                if nl != -1 and (punct_pos == -1 or nl <= punct_pos):
                    line = buffer[:nl].strip()
                    buffer = buffer[nl + 1:]
                elif punct_pos != -1:
                    p_char = buffer[punct_pos]
                    cut = punct_pos + len(p_char)
                    line = buffer[:cut].strip()
                    buffer = buffer[cut:]
                else:
                    break
                try_emit(line)
                drain_pending(wait=False)

        logger.info(f"[REQ {session_id}] LLM完成 {time.time()-req_start:.2f}s | {seg_idx}段")

        last = buffer.strip()
        if last:
            try_emit(last)
        carry_over = ""
        drain_pending(wait=True)

    except Exception as e:
        logger.error(f"[REQ {session_id}] 异常: {e}", exc_info=True)
        broadcast({"type": "error", "msg": str(e)})
        broadcast({"type": "ai_speaking", "value": False})
        return

    if all_segs:
        _log("ai", username, "\n".join(all_segs))

    total_time = time.time() - req_start
    logger.info(f"[REQ {session_id}] 完成 总耗时{total_time:.2f}s | {seg_idx}段推送完毕")
    broadcast({"type": "done", "total": seg_idx})
    broadcast({"type": "ai_speaking", "value": False})


def _reply_worker():
    """后台单线程，串行消费 _reply_queue，保证 LLM/TTS 不并发。"""
    logger.info("[Worker] _reply_worker 启动")
    while True:
        try:
            item = _reply_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            _stream_and_broadcast(item)
        except Exception as e:
            logger.error(f"[Worker] 处理异常: {e}", exc_info=True)
        finally:
            _reply_queue.task_done()


# 启动 worker 线程
_worker_thread = threading.Thread(target=_reply_worker, daemon=True)
_worker_thread.start()


# ─────────────────────── HTML ───────────────────────
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
  .header-right { display: flex; align-items: center; gap: 16px; }
  .online-badge { font-size: 0.8rem; color: #6ee7b7; background: #064e3b;
                  padding: 3px 10px; border-radius: 20px; }
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
  .user-self { background: #5b21b6; align-self: flex-end; }
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
  /* AI 正在说话的提示 */
  .ai-speaking { color: #93c5fd; font-size: 0.78rem; align-self: center;
                 animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  /* 进入遮罩 */
  #enter-overlay {
    position: fixed; inset: 0; background: rgba(10,10,20,0.92);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 999; gap: 18px;
  }
  #enter-overlay h2 { color: #a78bfa; font-size: 1.6rem; }
  #enter-overlay p  { color: #9ca3af; font-size: 0.95rem; }
  #enter-btn {
    padding: 14px 48px; background: #7c3aed; color: white; border: none;
    border-radius: 12px; font-size: 1.1rem; cursor: pointer; transition: background 0.2s;
  }
  #enter-btn:hover { background: #6d28d9; }
</style>
</head>
<body>
<div id="enter-overlay">
  <h2>🎙 小晴的直播间</h2>
  <p>点击进入，和小晴聊聊天～</p>
  <button id="enter-btn" onclick="enterRoom()">进入直播间</button>
</div>
<header>
  <div class="title">🎙 小晴的直播间</div>
  <div class="header-right">
    <span class="online-badge" id="online-badge">● 1 人在线</span>
    <div class="nickname-wrap">
      <span>我的昵称：</span>
      <span id="nick-display" title="点击修改昵称"></span>
      <input id="nick-input" type="text" maxlength="12" placeholder="输入昵称回车确认" />
    </div>
  </div>
</header>
<div id="chat-box"></div>
<div id="input-row">
  <input id="user-input" type="text" placeholder="说点什么吧..." autofocus />
  <button id="send-btn" onclick="sendMsg()">发送</button>
</div>

<script>
// ── clientId：每个 tab 唯一，刷新后重新生成，用于精确管理 SSE 连接 ──
const clientId = Math.random().toString(36).slice(2) + Date.now().toString(36);

// ── 昵称管理 ──
const NICKNAMES = [
  '小可爱','宝贝','亲爱的','小甜心','小心肝','小宝贝',
  '暖暖','晴天','软糖','棉花糖','奶茶','芝芝','糖糖',
  '甜甜','暖心','小鹿','星星','小雨','夏天','初晴',
];
function genNick() {
  const base = NICKNAMES[Math.floor(Math.random() * NICKNAMES.length)];
  const num  = String(Math.floor(Math.random() * 900) + 100);  // 100~999
  return base + num;
}
// sessionStorage：每个 tab 独立，避免同一浏览器多 tab 昵称相同
// 优先用 sessionStorage，没有则随机生成新昵称（不复用 localStorage 里的）
let username = sessionStorage.getItem('dh_nick') || genNick();
sessionStorage.setItem('dh_nick', username);
// localStorage 仅做最后一次手动修改的记忆，不在多 tab 间共享初始昵称

const nickDisplay = document.getElementById('nick-display');
const nickInput   = document.getElementById('nick-input');
nickDisplay.textContent = username;

nickDisplay.addEventListener('click', () => {
  nickInput.value = username;
  nickDisplay.style.display = 'none';
  nickInput.style.display = 'inline-block';
  nickInput.focus();
});
nickInput.addEventListener('keydown', e => { if (e.key === 'Enter') confirmNick(); });
nickInput.addEventListener('blur', confirmNick);
function confirmNick() {
  const v = nickInput.value.trim();
  if (v) { username = v; sessionStorage.setItem('dh_nick', username); }
  nickDisplay.textContent = username;
  nickDisplay.style.display = 'inline';
  nickInput.style.display = 'none';
}

// ── 聊天渲染 ──
const chatBox = document.getElementById('chat-box');
const input   = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
input.addEventListener('keydown', e => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

function addMsg(role, text, nick) {
  const div = document.createElement('div');
  // 自己发的消息用稍深的紫色区分
  div.className = 'msg ' + (role === 'user' ? (nick === username ? 'user-self' : 'user') : role);
  const label = document.createElement('span');
  label.className = role === 'user' ? 'user-label' : 'ai-label';
  label.textContent = role === 'user' ? (nick || username) + '：' : '🎙 小晴：';
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

// ── 背景音乐（Web Audio API 合成 lo-fi 鼓机）──
let bgCtx = null;
let bgGain = null;
let bgStarted = false;
const BG_VOL_NORMAL = 0.18;
const BG_VOL_DUCK   = 0.04;

function _startBgMusic() {
  if (bgStarted) return;
  bgStarted = true;
  bgCtx  = new (window.AudioContext || window.webkitAudioContext)();
  bgGain = bgCtx.createGain();
  bgGain.gain.setValueAtTime(BG_VOL_NORMAL, bgCtx.currentTime);
  bgGain.connect(bgCtx.destination);
  _scheduleBeat(bgCtx.currentTime);
}

function _scheduleBeat(startTime) {
  if (!bgCtx) return;
  const bpm    = 82;
  const beat   = 60 / bpm;
  const barLen = beat * 4;
  const BARS   = 2;
  for (let bar = 0; bar < BARS; bar++) {
    const t0 = startTime + bar * barLen;
    _kick(t0); _kick(t0 + beat * 2);
    _snare(t0 + beat); _snare(t0 + beat * 3);
    for (let i = 0; i < 8; i++) _hat(t0 + i * beat * 0.5, i % 2 === 0 ? 0.55 : 0.35);
  }
  const nextBar = startTime + BARS * barLen - 0.05;
  const delay   = (nextBar - bgCtx.currentTime) * 1000;
  setTimeout(() => _scheduleBeat(startTime + BARS * barLen), Math.max(delay, 0));
}

function _kick(t) {
  const o = bgCtx.createOscillator(), g = bgCtx.createGain();
  o.connect(g); g.connect(bgGain);
  o.frequency.setValueAtTime(160, t);
  o.frequency.exponentialRampToValueAtTime(40, t + 0.15);
  g.gain.setValueAtTime(1.2, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
  o.start(t); o.stop(t + 0.35);
}
function _snare(t) {
  const bufSize = Math.floor(bgCtx.sampleRate * 0.12);
  const buf = bgCtx.createBuffer(1, bufSize, bgCtx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) data[i] = (Math.random() * 2 - 1);
  const src = bgCtx.createBufferSource(); src.buffer = buf;
  const flt = bgCtx.createBiquadFilter(); flt.type = 'highpass'; flt.frequency.value = 1800;
  const g = bgCtx.createGain();
  src.connect(flt); flt.connect(g); g.connect(bgGain);
  g.gain.setValueAtTime(0.6, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
  src.start(t); src.stop(t + 0.12);
}
function _hat(t, vol) {
  const bufSize = Math.floor(bgCtx.sampleRate * 0.04);
  const buf = bgCtx.createBuffer(1, bufSize, bgCtx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) data[i] = (Math.random() * 2 - 1);
  const src = bgCtx.createBufferSource(); src.buffer = buf;
  const flt = bgCtx.createBiquadFilter(); flt.type = 'highpass'; flt.frequency.value = 8000;
  const g = bgCtx.createGain();
  src.connect(flt); flt.connect(g); g.connect(bgGain);
  g.gain.setValueAtTime(vol * 0.4, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
  src.start(t); src.stop(t + 0.04);
}

function bgDuck()    { if (bgGain) bgGain.gain.linearRampToValueAtTime(BG_VOL_DUCK,   bgCtx.currentTime + 0.3); }
function bgRestore() { if (bgGain) bgGain.gain.linearRampToValueAtTime(BG_VOL_NORMAL, bgCtx.currentTime + 0.8); }

// ── 音频队列 ──
let audioQueue = [];
let isPlaying  = false;
let streamDone = false;
let aiSpeakingEl = null;  // "小晴正在说话..." 提示节点

// 浏览器 Autoplay 解锁
let userInteracted = false;
let pendingPlay = null;
function markInteracted() {
  if (userInteracted) return;
  userInteracted = true;
  _startBgMusic();
  if (pendingPlay) { const fn = pendingPlay; pendingPlay = null; fn(); }
}
document.addEventListener('click',   markInteracted, { once: false });
document.addEventListener('keydown', markInteracted, { once: false });

function enqueueAudio(url) {
  audioQueue.push(url);
  if (!isPlaying) tryPlayNext();
}

function tryPlayNext() {
  if (audioQueue.length === 0) {
    isPlaying = false;
    bgRestore();
    if (streamDone) onAllDone();
    return;
  }
  const playFn = () => {
    isPlaying = true;
    bgDuck();
    const url = audioQueue.shift() + '?t=' + Date.now();
    const a = new Audio(url);
    let advanced = false;
    const advance = () => { if (!advanced) { advanced = true; setTimeout(tryPlayNext, 600); } };
    a.addEventListener('ended', advance);
    a.addEventListener('error', advance);
    const watchdog = setTimeout(advance, 10000);
    a.addEventListener('ended', () => clearTimeout(watchdog));
    a.addEventListener('error', () => clearTimeout(watchdog));
    a.play().catch(() => { isPlaying = false; bgRestore(); tryPlayNext(); });
  };
  if (!userInteracted) { pendingPlay = playFn; return; }
  playFn();
}

// ── 冷场检测 ──
let idleTimer = null;
const IDLE_SEC = 20;

function resetIdleTimer() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(triggerIdle, IDLE_SEC * 1000);
}
function onAllDone() {
  if (aiSpeakingEl) { aiSpeakingEl.remove(); aiSpeakingEl = null; }
  resetIdleTimer();
}

async function triggerIdle() {
  if (isPlaying || audioQueue.length > 0) { resetIdleTimer(); return; }
  await fetch('/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username })
  });
}

// ── 广播 SSE 接收（/events 长连接）──
function connectEvents({ onReady } = {}) {
  const es = new EventSource('/events?username=' + encodeURIComponent(username) + '&clientId=' + clientId);

  es.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }

    if (evt.type === 'online') {
      // 第一次收到 online = SSE 连接真正就绪，触发 onReady 回调（只触发一次）
      if (onReady) { onReady(); onReady = null; }
      const badge = document.getElementById('online-badge');
      if (badge) badge.textContent = '● ' + evt.count + ' 人在线';

    } else if (evt.type === 'user_msg') {
      // 其他用户发言的气泡（自己的已在 sendMsg 乐观显示，跳过）
      if (evt.username !== username) {
        addMsg('user', evt.text, evt.username);
      }

    } else if (evt.type === 'ai_speaking') {
      if (evt.value) {
        // 显示"小晴正在说话..."
        if (!aiSpeakingEl) {
          aiSpeakingEl = document.createElement('div');
          aiSpeakingEl.className = 'msg ai-speaking';
          aiSpeakingEl.textContent = '🎙 小晴正在说话...';
          chatBox.appendChild(aiSpeakingEl);
          chatBox.scrollTop = chatBox.scrollHeight;
        }
        streamDone = false;
      } else {
        // AI 说完了，若队列空直接 onAllDone
        streamDone = true;
        if (!isPlaying && audioQueue.length === 0) onAllDone();
      }

    } else if (evt.type === 'seg') {
      if (aiSpeakingEl) { aiSpeakingEl.remove(); aiSpeakingEl = null; }
      console.log('[seg]', evt.idx, evt.text, 'queue:', audioQueue.length, 'playing:', isPlaying);
      enqueueAudio(evt.url);
      const delay = evt.idx * 600;
      setTimeout(() => addMsg('ai', evt.text), delay);

    } else if (evt.type === 'done') {
      streamDone = true;
      if (!isPlaying && audioQueue.length === 0) onAllDone();

    } else if (evt.type === 'error') {
      if (aiSpeakingEl) { aiSpeakingEl.remove(); aiSpeakingEl = null; }
      addSys('❌ ' + evt.msg);
      onAllDone();

    } else if (evt.type === 'heartbeat') {
      // 忽略
    }
  };

  es.onerror = () => {
    es.close();
    setTimeout(connectEvents, 3000);  // 断线重连
  };
}

// ── 发送消息 ──
async function sendMsg() {
  const text = input.value.trim();
  if (!text) return;
  markInteracted();
  input.value = '';
  clearTimeout(idleTimer);

  // 乐观渲染自己的气泡（不等广播回来）
  addMsg('user', text, username);

  await fetch('/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, username })
  });
}

// ── 进入直播间 ──
async function enterRoom() {
  const overlay = document.getElementById('enter-overlay');
  overlay.style.display = 'none';
  markInteracted();

  // 先建立广播连接，等收到第一个 online 事件（SSE 连接真正就绪）再发欢迎请求
  // 避免欢迎语广播时 /events 还没建立而丢失
  connectEvents({ onReady: () => {
    fetch('/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enter: true, username })
    });
    resetIdleTimer();
  }});
}

// 页面关闭/刷新时主动通知后端注销该 tab 的连接
window.addEventListener('beforeunload', () => {
  navigator.sendBeacon('/leave', JSON.stringify({ clientId }));
});
</script>
</body>
</html>
"""


# ─────────────────────── API ───────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/events")
def events():
    """SSE 长连接：每个用户进入直播间后持有，接收广播消息。"""
    uname     = request.args.get("username",  "宝贝").strip() or "宝贝"
    client_id = request.args.get("clientId",  "").strip() or uuid.uuid4().hex[:12]
    c = _register_client(client_id, uname)

    # 通知所有人刷新在线人数
    broadcast({"type": "online", "count": online_count()})

    def generate():
        try:
            while True:
                try:
                    msg = c.q.get(timeout=20)
                except queue.Empty:
                    msg = _sse({"type": "heartbeat"})
                try:
                    yield msg
                except Exception:
                    break
        finally:
            _unregister_client(client_id)
            broadcast({"type": "online", "count": online_count()})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/stream", methods=["POST"])
def stream():
    """接收用户消息，入队 _reply_queue，立即返回 204。"""
    data     = request.get_json() or {}
    uname    = data.get("username", "宝贝").strip() or "宝贝"
    text     = data.get("text", "").strip()
    is_enter = data.get("enter", False)

    if is_enter:
        logger.info(f"[进入直播间] user={uname}")
        prompt = (
            f"[系统通知]：用户「{uname}」刚刚进入了直播间。"
            "请像真实直播主播一样欢迎他，说3行，每行一句，换行分隔。"
            "第一行：欢迎他进入直播间，可以用『欢迎欢迎』『哎呀来啦』『诶～来了』等开场，带他的名字；"
            "第二行：一句暖场的话，比如夸他、问他哪里来的、说今天等他好久了之类；"
            "第三行：邀请他留下来聊，说法要自然撩人，不要太正式；"
            "语气热情、口语、有点黏，不要像机器人报幕，不要用感叹号，不要书面语。"
            "每行可以在中间插一个语气标签（[uv_break]或[laugh_0]），但不能放在行末。"
        )
        _log("system", uname, "进入直播间")
        _reply_queue.put({"username": uname, "prompt": prompt, "log_role": "system", "display_text": None})

    elif text:
        logger.info(f"[用户发言] user={uname} | {text!r}")
        _inc_user_turns(uname)
        relation = _get_relation(uname)
        prompt   = f"[当前关系阶段：{relation}]\n[{uname}说]：{text}"
        _log("user", uname, text)
        # 立即广播用户气泡，不等 worker（所有在线用户实时看到弹幕）
        broadcast({"type": "user_msg", "username": uname, "text": text})
        _reply_queue.put({"username": uname, "prompt": prompt, "log_role": "user", "display_text": None})

    else:
        # 冷场触发（idle，前端只传 username）
        logger.info(f"[冷场触发] user={uname}")
        prompt = _next_idle_prompt(uname)
        _log("system", uname, "[冷场触发]")
        _reply_queue.put({"username": uname, "prompt": prompt, "log_role": "system", "display_text": None})

    return "", 204


@app.route("/leave", methods=["POST"])
def leave():
    """前端页面关闭/刷新时主动调用，精确注销该 tab 的连接。"""
    # sendBeacon 发的是 text/plain，需要手动解析
    try:
        raw = request.get_data(as_text=True)
        data = json.loads(raw)
    except Exception:
        data = {}
    client_id = data.get("clientId", "").strip()
    if client_id:
        _unregister_client(client_id)
        broadcast({"type": "online", "count": online_count()})
    return "", 204


@app.route("/audio_cache/<filename>")
def audio_cache(filename):
    if "/" in filename or ".." in filename:
        return "Bad request", 400
    path = os.path.join(_AUDIO_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    return send_file(path, mimetype="audio/wav")


@app.route("/seed_samples/<filename>")
def seed_sample(filename):
    sample_dir = os.path.join(_DIR, "..", "voice_chat", "seed_samples")
    return send_file(os.path.join(sample_dir, filename), mimetype="audio/wav")


@app.route("/tts_test")
def tts_test_page():
    return render_template_string(TTS_TEST_HTML)


@app.route("/tts_test/synth", methods=["POST"])
def tts_test_synth():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    fname = f"ttstest_{uuid.uuid4().hex[:8]}.wav"
    fpath = os.path.join(_AUDIO_DIR, fname)
    ok = _synth_to_file(text, fpath)
    if not ok:
        return jsonify({"error": "synth failed"}), 500
    return jsonify({"url": f"/audio_cache/{fname}", "cleaned": TTSEngine._clean_text(text)})


TTS_TEST_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>ChatTTS 标签测试</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 24px; max-width: 900px; margin: 0 auto; }
h1 { color: #a78bfa; margin-bottom: 6px; }
.subtitle { color: #6b7280; font-size: 0.85rem; margin-bottom: 24px; }
.section { margin-bottom: 32px; }
.section h2 { color: #7dd3fc; font-size: 1rem; margin-bottom: 12px; border-bottom: 1px solid #1f2937; padding-bottom: 6px; }
.cases { display: flex; flex-direction: column; gap: 8px; }
.case { background: #1a1a2e; border-radius: 10px; padding: 12px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.case-label { font-size: 0.82rem; color: #9ca3af; min-width: 130px; flex-shrink: 0; }
.case-text { flex: 1; font-size: 0.88rem; color: #c4b5fd; font-family: monospace; word-break: break-all; min-width: 180px; }
.case-right { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.play-btn { padding: 6px 14px; background: #7c3aed; border: none; border-radius: 6px; color: white;
            cursor: pointer; font-size: 0.85rem; white-space: nowrap; transition: background 0.2s; }
.play-btn:hover { background: #6d28d9; }
.play-btn:disabled { background: #374151; cursor: not-allowed; }
.status { font-size: 0.75rem; color: #6b7280; min-width: 50px; }
.feedback-input { padding: 4px 8px; background: #111827; border: 1px solid #374151; color: #e0e0e0;
                  border-radius: 6px; font-size: 0.8rem; width: 150px; outline: none; }
.feedback-input:focus { border-color: #7c3aed; }
.custom-area, .export-area { background: #1a1a2e; border-radius: 10px; padding: 16px; }
.export-area { margin-top: 32px; }
.export-area h2 { color: #7dd3fc; font-size: 1rem; margin-bottom: 12px; }
textarea { width: 100%; background: #111827; border: 1px solid #374151; color: #e0e0e0;
           border-radius: 6px; padding: 10px; font-size: 0.9rem; font-family: monospace;
           resize: vertical; min-height: 80px; outline: none; }
textarea:focus { border-color: #7c3aed; }
.btn-row { display: flex; gap: 10px; margin-top: 10px; align-items: center; }
.cleaned { font-size: 0.75rem; color: #4b5563; margin-top: 8px; word-break: break-all; }
#feedback-summary { min-height: 120px; }
</style>
</head>
<body>
<h1>🔬 ChatTTS 标签测试</h1>
<p class="subtitle">点播放听效果，右侧输入框填反馈（好/差/截断/无效），最后点生成汇总复制给我。</p>

<div class="section"><h2>🔇 停顿类</h2><div class="cases" id="cases-pause"></div></div>
<div class="section"><h2>😄 笑声类</h2><div class="cases" id="cases-laugh"></div></div>
<div class="section"><h2>🗣️ 口语强调 [oral_N]</h2><div class="cases" id="cases-oral"></div></div>
<div class="section"><h2>📝 组合示例</h2><div class="cases" id="cases-combo"></div></div>

<div class="section">
  <h2>✏️ 自定义测试</h2>
  <div class="custom-area">
    <textarea id="custom-text" placeholder="输入文本，支持标签，如：诶[uv_break]今儿才来呀[laugh_0]真的"></textarea>
    <div class="btn-row">
      <button class="play-btn" id="custom-btn">合成并播放</button>
      <span class="status" id="custom-status"></span>
    </div>
    <div class="cleaned" id="custom-cleaned"></div>
  </div>
</div>

<div class="export-area">
  <h2>📋 反馈汇总</h2>
  <div class="btn-row" style="margin-bottom:10px;margin-top:0">
    <button class="play-btn" onclick="genSummary()">生成汇总</button>
    <button class="play-btn" style="background:#065f46" onclick="copySummary()">复制</button>
  </div>
  <textarea id="feedback-summary" readonly placeholder="点生成汇总..."></textarea>
</div>

<script>
const CASES = {
  "cases-pause": [
    ["无标签（基准）",    "诶，今儿才来呀，想死你了"],
    ["[uv_break] 中间",  "诶[uv_break]今儿才来呀，想死你了"],
    ["[v_break] 中间",   "诶[v_break]今儿才来呀，想死你了"],
    ["[lbreak] 中间",    "诶，今儿才来呀[lbreak]想死你了"],
    ["[break_2] 中间",   "诶，今儿才来呀[break_2]想死你了"],
    ["[break_5] 中间",   "诶，今儿才来呀[break_5]想死你了"],
  ],
  "cases-laugh": [
    ["无笑声（基准）",    "昨天买东西花了好多钱，我自己都傻了"],
    ["[laugh_0] 中间",   "昨天买东西[laugh_0]花了好多钱，我自己都傻了"],
    ["[laugh_1] 中间",   "昨天买东西[laugh_1]花了好多钱，我自己都傻了"],
    ["[laugh_2] 中间",   "昨天买东西[laugh_2]花了好多钱，我自己都傻了"],
    ["[laugh] 通用 中间", "昨天买东西[laugh]花了好多钱，我自己都傻了"],
    ["[laugh_0] 句末⚠️", "昨天买东西花了好多钱[laugh_0]"],
  ],
  "cases-oral": [
    ["无标签（基准）",   "就二十八块五，老便宜了"],
    ["[oral_3] 前",     "就[oral_3]二十八块五，老便宜了"],
    ["[oral_5] 前",     "就[oral_5]二十八块五，老便宜了"],
    ["[oral_7] 前",     "就[oral_7]二十八块五，老便宜了"],
    ["[oral_9] 前",     "就[oral_9]二十八块五，老便宜了"],
    ["[oral_7] 句末⚠️", "就二十八块五，老便宜了[oral_7]"],
  ],
  "cases-combo": [
    ["欢迎A（纯文本）",   "诶，小可爱来啦，想死你了"],
    ["欢迎B（uv_break）", "诶[uv_break]小可爱来啦，想死你了"],
    ["欢迎C（laugh中间）","诶[uv_break]小可爱[laugh_0]来啦，想死你了"],
    ["购物A",             "昨天去超市[uv_break]本来就买个酱油"],
    ["购物B（oral+laugh）","结果[oral_5]花了一百二十八块[laugh_1]我自己都傻了"],
    ["冷场（oral+uv）",   "哎[uv_break]今儿中午我整了碗[oral_5]牛肉面，贼好吃"],
  ],
};

const feedbacks = {};

function buildCases() {
  for (const [id, items] of Object.entries(CASES)) {
    const container = document.getElementById(id);
    items.forEach(([label, text], i) => {
      const key = id + '_' + i;
      const div = document.createElement('div');
      div.className = 'case';
      const labelEl = document.createElement('span'); labelEl.className = 'case-label'; labelEl.textContent = label;
      const textEl  = document.createElement('span'); textEl.className  = 'case-text';  textEl.textContent  = text;
      const rightEl = document.createElement('div');  rightEl.className = 'case-right';
      const btn     = document.createElement('button'); btn.className = 'play-btn'; btn.textContent = '▶ 播放';
      const statusEl = document.createElement('span'); statusEl.className = 'status';
      const fbInput  = document.createElement('input'); fbInput.className = 'feedback-input'; fbInput.placeholder = '填反馈...';
      fbInput.addEventListener('input', () => { feedbacks[key] = {label, text, fb: fbInput.value}; });
      const audioEl = document.createElement('audio'); audioEl.style.display = 'none';
      btn.addEventListener('click', () => synthAndPlay(btn, text, statusEl, audioEl));
      rightEl.appendChild(btn); rightEl.appendChild(statusEl); rightEl.appendChild(fbInput);
      div.appendChild(labelEl); div.appendChild(textEl); div.appendChild(rightEl); div.appendChild(audioEl);
      container.appendChild(div);
    });
  }
}

async function synthAndPlay(btn, text, statusEl, audioEl) {
  btn.disabled = true; statusEl.textContent = '合成中...';
  try {
    const r = await fetch('/tts_test/synth', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
    const data = await r.json();
    if (data.error) { statusEl.textContent = 'ERR'; btn.disabled = false; return; }
    audioEl.src = data.url + '?t=' + Date.now(); audioEl.style.display = 'inline'; audioEl.play();
    statusEl.textContent = '▶'; audioEl.onended = () => { statusEl.textContent = 'OK'; };
  } catch(e) { statusEl.textContent = 'ERR'; }
  btn.disabled = false;
}

document.getElementById('custom-btn').addEventListener('click', async () => {
  const text = document.getElementById('custom-text').value.trim();
  if (!text) return;
  const statusEl = document.getElementById('custom-status');
  const cleanedEl = document.getElementById('custom-cleaned');
  statusEl.textContent = '合成中...';
  try {
    const r = await fetch('/tts_test/synth', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
    const data = await r.json();
    if (data.error) { statusEl.textContent = 'ERR'; return; }
    cleanedEl.textContent = '送入TTS: ' + data.cleaned;
    statusEl.textContent = '▶';
    const a = new Audio(data.url + '?t=' + Date.now()); a.play();
    a.onended = () => { statusEl.textContent = 'OK'; };
  } catch(e) { statusEl.textContent = 'ERR'; }
});

function genSummary() {
  const lines = ['=== ChatTTS 标签测试反馈 ==='];
  for (const [key, {label, text, fb}] of Object.entries(feedbacks)) {
    if (fb) lines.push('[' + label + '] ' + text + '\\n  反馈: ' + fb);
  }
  if (lines.length === 1) lines.push('（还没填写任何反馈）');
  document.getElementById('feedback-summary').value = lines.join('\\n');
}
function copySummary() {
  const el = document.getElementById('feedback-summary'); el.select(); document.execCommand('copy'); alert('已复制到剪贴板');
}

buildCases();
</script>
</body>
</html>"""


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
    print(f"🔬 TTS标签测试页：http://<服务器IP>:{port}/tts_test\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
