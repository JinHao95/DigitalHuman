"""
web_chat/app.py
Web 版数字人：浏览器输入文字 → LLM → TTS → 浏览器播放音频

架构：SSE 流式推送 + 流水线并行
  LLM stream → 按换行切段 → 每段立即提交线程池 TTS
  → TTS 完成 → SSE push 段索引 → 前端边收边播
"""

import io
import os
import sys
import json
import uuid
import time
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
# 同时输出到终端
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

# TTS 线程池：GPU 是串行资源，用单线程池保证顺序且不阻塞 Flask 主线程
_tts_pool = ThreadPoolExecutor(max_workers=1)


# ─────────────────────── 关系阶段 ───────────────────────

# 每个用户的对话轮次（user 消息次数），用于推进关系阶段
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
    """追加一条对话记录到当天日志文件。"""
    today = datetime.date.today().isoformat()
    path  = os.path.join(_LOG_DIR, f"{today}.jsonl")
    record = {
        "ts":       datetime.datetime.now().isoformat(timespec="seconds"),
        "role":     role,       # "user" / "ai" / "system"
        "username": username,
        "text":     text,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _next_idle_prompt(username: str) -> str:
    """冷场时让 LLM 自由发挥，自己想话题说。"""
    return (
        f"[系统通知]：直播间有一段时间没人说话了，「{username}」还在线。"
        "请你自己想一件最近发生的小事或者你的日常，随口分享出来，"
        "说话要自然、口语、慢一点，就像在跟朋友聊天，不要太正式，不要总结，不要升华。"
    )


# ─────────────────────── 工具 ───────────────────────

def _synth_to_file(text: str, path: str) -> bool:
    """TTS 合成并写入文件，返回是否成功。在 TTS 线程池中执行。"""
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


# ─────────────────────── 核心流式生成 ───────────────────────

def _stream_response(prompt_text: str, session_id: str, username: str, log_role: str):
    """
    生成器：LLM 流式 → 切段 → 线程池 TTS → SSE 逐段推送。

    流水线：LLM 每输出一个换行段，立刻提交 TTS；
    主线程同时尝试 yield 已完成的段（保证顺序），
    实现 LLM / TTS / 播放三者并行，首段延迟最低。

    SSE 事件格式：
      {"type": "seg",  "idx": 0, "text": "...", "url": "/audio_cache/xxx_0.wav"}
      {"type": "done", "total": 3}
      {"type": "error","msg": "..."}
    """
    req_start = time.time()
    logger.info(f"[REQ {session_id}] 开始 | user={username} | role={log_role} | prompt={prompt_text[:60]!r}")

    import re as _re
    _TAG_ONLY = _re.compile(r'^(\[[\w_]+\]|…+|～+|\s)*$')  # 纯标签/纯省略号，无实质文字

    buffer     = ""
    seg_idx    = 0
    pending    = []   # list of (idx, text, fname, future)
    all_segs   = []   # 收集所有段文字，用于日志
    carry_over = ""   # 纯标签段暂存，拼到下一个实质段前面
    llm_done_time = None

    def submit_seg(text, idx):
        logger.debug(f"[REQ {session_id}] 提交TTS seg{idx} | {text[:40]!r}")
        fname = f"{session_id}_{idx}.wav"
        fpath = os.path.join(_AUDIO_DIR, fname)
        fut   = _tts_pool.submit(_synth_to_file, text, fpath)
        pending.append((idx, text, fname, fut))
        all_segs.append(text)

    def try_emit(line):
        """判断是否实质段；纯标签暂存 carry_over，实质段拼上后提交。"""
        nonlocal seg_idx, carry_over
        if not line:
            return
        if _TAG_ONLY.match(line):
            # 纯标签/省略号：暂存，拼到下一实质段开头
            carry_over += line
            logger.debug(f"[REQ {session_id}] 暂存纯标签 {line!r}")
        else:
            full = (carry_over + line) if carry_over else line
            carry_over = ""
            submit_seg(full, seg_idx)
            seg_idx += 1

    try:
        for chunk in llm.chat_stream(prompt_text):
            buffer += chunk
            # 按换行 或 句末标点切段，不等 LLM 全部输出
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
                yield from _drain_pending(pending, wait=False)

        llm_done_time = time.time()
        logger.info(f"[REQ {session_id}] LLM完成 {llm_done_time - req_start:.2f}s | {seg_idx}段")

        last = buffer.strip()
        if last:
            try_emit(last)
        # carry_over 里还有未拼出的纯标签，直接丢弃（不提交，避免爆音）
        carry_over = ""

        yield from _drain_pending(pending, wait=True)

    except Exception as e:
        logger.error(f"[REQ {session_id}] 异常: {e}", exc_info=True)
        yield _sse({"type": "error", "msg": str(e)})
        return

    # 记录 AI 完整回复
    if all_segs:
        _log("ai", username, "\n".join(all_segs))

    total_time = time.time() - req_start
    logger.info(f"[REQ {session_id}] 完成 总耗时{total_time:.2f}s | {seg_idx}段推送完毕")
    yield _sse({"type": "done", "total": seg_idx})


def _drain_pending(pending: list, wait: bool):
    """
    按顺序 yield 已完成的段 SSE 事件。
    - 第一段（队列头）：始终阻塞等待，保证首段尽快推出去
    - 后续段：只有 wait=True 才阻塞；wait=False 时遇到未完成就停
    这样前端能在第一段 TTS 完成后立刻开始播，后续段在播放过程中陆续到达。
    """
    i = 0
    while i < len(pending):
        idx, text, fname, fut = pending[i]
        if i == 0 or wait:
            ok = fut.result()   # 第一段或收尾阶段：阻塞等待
        else:
            if not fut.done():
                break           # 后续段未完成：先推出去已有的，下轮再来
            ok = fut.result()
        if ok:
            yield _sse({"type": "seg", "idx": idx, "text": text,
                        "url": f"/audio_cache/{fname}"})
        i += 1
    del pending[:i]


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
const NICKNAMES = [
  '小可爱','宝贝','亲爱的','小甜心','小心肝','小宝贝',
  '暖暖','晴天','软糖','棉花糖','奶茶','芝芝','糖糖',
  '甜甜','暖心','小鹿','星星','小雨','夏天','初晴',
];
function genNick() {
  return NICKNAMES[Math.floor(Math.random() * NICKNAMES.length)];
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
nickInput.addEventListener('keydown', e => { if (e.key === 'Enter') confirmNick(); });
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
input.addEventListener('keydown', e => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); } });

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const label = document.createElement('span');
  label.className = role === 'user' ? 'user-label' : 'ai-label';
  label.textContent = role === 'user' ? username + '：' : '🎙 小晴：';
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
// 动次打次节奏：kick(0) hat(0.5) snare(1.0) hat(1.5) 每2拍循环，BPM≈80
let bgCtx = null;
let bgGain = null;
let bgStarted = false;
const BG_VOL_NORMAL = 0.18;   // 正常音量
const BG_VOL_DUCK   = 0.04;   // 说话时压低

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
  const bpm      = 82;
  const beat     = 60 / bpm;       // 一拍时长（秒）
  const barLen   = beat * 4;        // 一小节 = 4拍
  const BARS     = 2;               // 每次预排 2 小节，减少 GC

  for (let bar = 0; bar < BARS; bar++) {
    const t0 = startTime + bar * barLen;
    // kick  拍1 拍3
    _kick(t0);
    _kick(t0 + beat * 2);
    // snare 拍2 拍4
    _snare(t0 + beat);
    _snare(t0 + beat * 3);
    // hihat 每半拍
    for (let i = 0; i < 8; i++) _hat(t0 + i * beat * 0.5, i % 2 === 0 ? 0.55 : 0.35);
  }
  // 循环调度
  const nextBar = startTime + BARS * barLen - 0.05;
  const delay   = (nextBar - bgCtx.currentTime) * 1000;
  setTimeout(() => _scheduleBeat(startTime + BARS * barLen), Math.max(delay, 0));
}

function _kick(t) {
  const o = bgCtx.createOscillator();
  const g = bgCtx.createGain();
  o.connect(g); g.connect(bgGain);
  o.frequency.setValueAtTime(160, t);
  o.frequency.exponentialRampToValueAtTime(40, t + 0.15);
  g.gain.setValueAtTime(1.2, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
  o.start(t); o.stop(t + 0.35);
}
function _snare(t) {
  // noise burst
  const bufSize = Math.floor(bgCtx.sampleRate * 0.12);
  const buf  = bgCtx.createBuffer(1, bufSize, bgCtx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) data[i] = (Math.random() * 2 - 1);
  const src  = bgCtx.createBufferSource();
  src.buffer = buf;
  const flt  = bgCtx.createBiquadFilter();
  flt.type = 'highpass'; flt.frequency.value = 1800;
  const g   = bgCtx.createGain();
  src.connect(flt); flt.connect(g); g.connect(bgGain);
  g.gain.setValueAtTime(0.6, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
  src.start(t); src.stop(t + 0.12);
}
function _hat(t, vol) {
  const bufSize = Math.floor(bgCtx.sampleRate * 0.04);
  const buf  = bgCtx.createBuffer(1, bufSize, bgCtx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufSize; i++) data[i] = (Math.random() * 2 - 1);
  const src = bgCtx.createBufferSource();
  src.buffer = buf;
  const flt = bgCtx.createBiquadFilter();
  flt.type = 'highpass'; flt.frequency.value = 8000;
  const g = bgCtx.createGain();
  src.connect(flt); flt.connect(g); g.connect(bgGain);
  g.gain.setValueAtTime(vol * 0.4, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.04);
  src.start(t); src.stop(t + 0.04);
}

function bgDuck()   { if (bgGain) bgGain.gain.linearRampToValueAtTime(BG_VOL_DUCK,   bgCtx.currentTime + 0.3); }
function bgRestore(){ if (bgGain) bgGain.gain.linearRampToValueAtTime(BG_VOL_NORMAL, bgCtx.currentTime + 0.8); }

// ── 音频队列：边生成边播 ──
let audioQueue = [];   // 待播 URL 列表
let isPlaying  = false;
let streamDone = false;  // SSE 是否已结束
let currentReader = null; // 当前 SSE reader，用于中断
let pendingUserMsg = null; // 用户发消息时若 AI 正在讲，先缓存，讲完再处理

// 打断当前播放和 SSE 流（强制中断，用于必要时）
function interruptCurrent() {
  audioQueue = [];
  isPlaying  = false;
  streamDone = false;
  pendingUserMsg = null;
  bgRestore();
  if (currentReader) {
    currentReader.cancel();
    currentReader = null;
  }
}

// 浏览器 Autoplay 解锁
let userInteracted = false;
let pendingPlay = null;
function markInteracted() {
  if (userInteracted) return;
  userInteracted = true;
  _startBgMusic();   // 用户第一次交互后启动背景音乐
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
    bgRestore();           // 队列播完，恢复背景音量
    if (streamDone) onAllDone();
    return;
  }
  const playFn = () => {
    isPlaying = true;
    bgDuck();              // 开始播语音，背景压低
    const url = audioQueue.shift() + '?t=' + Date.now();
    const a = new Audio(url);
    let advanced = false;
    const advance = () => { if (!advanced) { advanced = true; setTimeout(tryPlayNext, 700); } };
    a.addEventListener('ended', advance);
    a.addEventListener('error', advance);
    // 保底：若 ended/error 都没触发（极少见），10s 后强制推进
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
const IDLE_SEC = 10;
let isResponding = false;  // AI 是否正在生成（用于冷场判断，不阻塞发消息）

function resetIdleTimer() {
  clearTimeout(idleTimer);
  if (isResponding || isPlaying) return;
  idleTimer = setTimeout(triggerIdle, IDLE_SEC * 1000);
}
function onAllDone() {
  isResponding = false;
  sendBtn.disabled = false;
  input.focus();
  // 若用户在 AI 讲话期间发了消息，现在处理
  if (pendingUserMsg) {
    const { text } = pendingUserMsg;
    pendingUserMsg = null;
    _doSend(text, false);  // 气泡已在发消息时加过
    return;
  }
  resetIdleTimer();
}

async function triggerIdle() {
  if (isResponding || isPlaying) { resetIdleTimer(); return; }
  isResponding = true;
  streamDone = false;
  await streamRequest('/stream', { username }, null);
}

// ── SSE 流式接收 ──
async function streamRequest(url, body, statusEl) {
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
  } catch(e) {
    if (statusEl) statusEl.remove();
    onAllDone();
    return;
  }
  if (!res.ok) {
    if (statusEl) statusEl.remove();
    addSys('❌ 请求失败');
    onAllDone();
    return;
  }
  const reader = res.body.getReader();
  currentReader = reader;
  const decoder = new TextDecoder();
  let buf = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }
        if (evt.type === 'seg') {
          if (statusEl) { statusEl.remove(); statusEl = null; }
          // 音频立即入队（不等气泡），气泡按段序延迟显示，制造逐句打出的真实感
          enqueueAudio(evt.url);
          const delay = evt.idx * 600;
          setTimeout(() => addMsg('ai', evt.text), delay);
        } else if (evt.type === 'done') {
          streamDone = true;
          if (!isPlaying && audioQueue.length === 0) onAllDone();
        } else if (evt.type === 'error') {
          if (statusEl) statusEl.remove();
          addSys('❌ ' + evt.msg);
          onAllDone();
        }
      }
    }
  } catch(e) {
    // reader 被 cancel() 打断，正常情况，不报错
  } finally {
    if (currentReader === reader) currentReader = null;
  }
}

// ── 发送消息 ──
async function sendMsg() {
  const text = input.value.trim();
  if (!text) return;
  markInteracted();
  input.value = '';
  clearTimeout(idleTimer);

  // 如果 AI 正在讲话，先缓存消息，等当前段讲完再回复
  if (isResponding || isPlaying) {
    pendingUserMsg = { text };
    addMsg('user', text);
    // 停止继续生成后续段（SSE），但不打断正在播的这段
    if (currentReader) { currentReader.cancel(); currentReader = null; }
    streamDone = true;
    // 若此刻既不在播放、队列也空，需要手动触发 onAllDone 避免死锁
    if (!isPlaying && audioQueue.length === 0) {
      setTimeout(onAllDone, 50);
    }
    return;
  }

  _doSend(text);
}

async function _doSend(text, showUserMsg = true) {
  isResponding = true;
  streamDone   = false;
  sendBtn.disabled = true;

  if (showUserMsg) addMsg('user', text);
  const status = addSys('小晴思考中...');
  await streamRequest('/stream', { text, username }, status);
}

// ── 进入直播间 ──
async function onEnter() {
  isResponding = true;
  streamDone   = false;
  audioQueue   = [];
  clearTimeout(idleTimer);
  try {
    await streamRequest('/stream', { enter: true, username }, null);
  } catch(e) {
    onAllDone();
  }
}

function enterRoom() {
  const overlay = document.getElementById('enter-overlay');
  overlay.style.display = 'none';
  markInteracted();   // 用户点击 = 交互解锁，autoplay 可用
  onEnter();
}

// 不再自动触发，等用户点"进入直播间"
</script>
</body>
</html>
"""


# ─────────────────────── API ───────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream", methods=["POST"])
def stream():
    """统一 SSE 流式接口：chat / enter / idle 都走这里。"""
    data     = request.get_json() or {}
    username = data.get("username", "宝贝").strip() or "宝贝"
    text     = data.get("text", "").strip()
    is_enter = data.get("enter", False)

    if is_enter:
        logger.info(f"[进入直播间] user={username}")
        prompt   = (
            f"[系统通知]：用户「{username}」刚刚进入了直播间。"
            "请像真实直播主播一样欢迎他，说3行，每行一句，换行分隔。"
            "第一行：欢迎他进入直播间，可以用『欢迎欢迎』『哎呀来啦』『诶～来了』等开场，带他的名字；"
            "第二行：一句暖场的话，比如夸他、问他哪里来的、说今天等他好久了之类；"
            "第三行：邀请他留下来聊，说法要自然撩人，不要太正式；"
            "语气热情、口语、有点黏，不要像机器人报幕，不要用感叹号，不要书面语。"
            "每行可以在中间插一个语气标签（[uv_break]或[laugh_0]），但不能放在行末。"
        )
        log_role = "system"
        _log("system", username, "进入直播间")
    elif text:
        logger.info(f"[用户发言] user={username} | {text!r}")
        _inc_user_turns(username)
        relation = _get_relation(username)
        prompt   = f"[当前关系阶段：{relation}]\n[{username}说]：{text}"
        log_role = "user"
        _log("user", username, text)
    else:
        logger.info(f"[冷场触发] user={username}")
        prompt   = _next_idle_prompt(username)
        log_role = "system"
        _log("system", username, "[冷场触发]")

    session_id = uuid.uuid4().hex[:8]

    return Response(
        stream_with_context(_stream_response(prompt, session_id, username, log_role)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/audio_cache/<filename>")
def audio_cache(filename):
    # 简单校验，防路径穿越
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
    """合成测试文本，返回音频 URL。"""
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

      const labelEl = document.createElement('span');
      labelEl.className = 'case-label';
      labelEl.textContent = label;

      const textEl = document.createElement('span');
      textEl.className = 'case-text';
      textEl.textContent = text;

      const rightEl = document.createElement('div');
      rightEl.className = 'case-right';

      const btn = document.createElement('button');
      btn.className = 'play-btn';
      btn.textContent = '▶ 播放';

      const statusEl = document.createElement('span');
      statusEl.className = 'status';

      const fbInput = document.createElement('input');
      fbInput.className = 'feedback-input';
      fbInput.placeholder = '填反馈...';
      fbInput.addEventListener('input', () => {
        feedbacks[key] = {label, text, fb: fbInput.value};
      });

      const audioEl = document.createElement('audio');
      audioEl.style.display = 'none';

      btn.addEventListener('click', () => synthAndPlay(btn, text, statusEl, audioEl));

      rightEl.appendChild(btn);
      rightEl.appendChild(statusEl);
      rightEl.appendChild(fbInput);
      div.appendChild(labelEl);
      div.appendChild(textEl);
      div.appendChild(rightEl);
      div.appendChild(audioEl);
      container.appendChild(div);
    });
  }
}

async function synthAndPlay(btn, text, statusEl, audioEl) {
  btn.disabled = true;
  statusEl.textContent = '合成中...';
  try {
    const r = await fetch('/tts_test/synth', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    if (data.error) { statusEl.textContent = 'ERR'; btn.disabled = false; return; }
    audioEl.src = data.url + '?t=' + Date.now();
    audioEl.style.display = 'inline';
    audioEl.play();
    statusEl.textContent = '▶';
    audioEl.onended = () => { statusEl.textContent = 'OK'; };
  } catch(e) {
    statusEl.textContent = 'ERR';
    console.error(e);
  }
  btn.disabled = false;
}

document.getElementById('custom-btn').addEventListener('click', async () => {
  const text = document.getElementById('custom-text').value.trim();
  if (!text) return;
  const statusEl = document.getElementById('custom-status');
  const cleanedEl = document.getElementById('custom-cleaned');
  statusEl.textContent = '合成中...';
  try {
    const r = await fetch('/tts_test/synth', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    if (data.error) { statusEl.textContent = 'ERR'; return; }
    cleanedEl.textContent = '送入TTS: ' + data.cleaned;
    statusEl.textContent = '▶';
    const a = new Audio(data.url + '?t=' + Date.now());
    a.play();
    a.onended = () => { statusEl.textContent = 'OK'; };
  } catch(e) { statusEl.textContent = 'ERR'; }
});

function genSummary() {
  const lines = ['=== ChatTTS 标签测试反馈 ==='];
  for (const [key, {label, text, fb}] of Object.entries(feedbacks)) {
    if (fb) lines.push('[' + label + '] ' + text + '\n  反馈: ' + fb);
  }
  if (lines.length === 1) lines.push('（还没填写任何反馈）');
  document.getElementById('feedback-summary').value = lines.join('\n');
}

function copySummary() {
  const el = document.getElementById('feedback-summary');
  el.select();
  document.execCommand('copy');
  alert('已复制到剪贴板');
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
    # threaded=True 支持 SSE 长连接 + 并发请求
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
