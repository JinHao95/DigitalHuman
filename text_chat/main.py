"""
main.py
文字输入版数字人主循环

流程：
  文字输入（命令行 / 弹幕）→ OpenAI LLM → TTS → 播放

运行方式：
    export OPENAI_API_KEY=sk-...
    python main.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import queue
import signal
import sys
import time
import threading
import numpy as np

# 复用 voice_chat 中的共享模块
_VOICE_CHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "voice_chat")
sys.path.insert(0, os.path.abspath(_VOICE_CHAT_DIR))

from llm import LLMClient
from tts import create_tts_engine
from audio_player import AudioPlayer
from input_source import CLIInputSource, DanmuInputSource

# ─────────────────────── 配置 ───────────────────────
# 消息合并窗口：积压消息 > 1 条时，等待多少秒再合并（0 = 不合并，逐条回复）
BATCH_WINDOW_SEC = float(os.getenv("BATCH_WINDOW_SEC", "0"))

# 输入模式：cli（命令行）/ danmu（弹幕队列）
INPUT_MODE = os.getenv("INPUT_MODE", "cli")

# ─────────────────────── 颜色输出 ───────────────────────
RESET  = "\033[0m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"

def print_status(text: str):
    print(f"{YELLOW}{text}{RESET}", flush=True)

# ─────────────────────── 优雅退出 ───────────────────────
_shutdown = False

def _handle_sigint(sig, frame):
    global _shutdown
    print(f"\n\n{YELLOW}[系统] 正在退出，再见！{RESET}")
    _shutdown = True
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_sigint)


# ─────────────────────── CLI 输入线程 ───────────────────────

def _cli_input_thread(msg_queue: queue.Queue):
    """
    子线程：持续读取命令行输入，推入消息队列。
    主线程处理队列，互不阻塞。
    """
    src = CLIInputSource()
    while not _shutdown:
        text = src.get_next()
        if text is None:
            msg_queue.put(None)  # 发送退出信号
            break
        msg_queue.put(text)


# ─────────────────────── 核心处理 ───────────────────────

def process_text(text: str, llm: LLMClient, tts, player: AudioPlayer) -> None:
    """LLM → TTS → 非阻塞播放。主循环可随时通过 player.stop() 打断。"""
    print(f"\n{BOLD}{CYAN}[输入]{RESET} {text}")

    # LLM 流式生成
    print_status("[LLM ] 正在生成回复...")
    print(f"{BOLD}{GREEN}[AI  ]{RESET} ", end="", flush=True)
    t0 = time.time()
    ai_text = ""
    try:
        for chunk in llm.chat_stream(text):
            print(chunk, end="", flush=True)
            ai_text += chunk
    except Exception as e:
        print(f"\n[LLM 错误] {e}")
        return
    print()
    print_status(f"      (LLM {time.time()-t0:.2f}s)")

    if not ai_text.strip():
        return

    # TTS 合成
    print_status("[TTS ] 正在合成语音...")
    t1 = time.time()
    try:
        audio = tts.synthesize(ai_text)
    except Exception as e:
        print(f"[TTS 错误] {e}")
        return
    print_status(f"      (TTS {time.time()-t1:.2f}s，音频 {len(audio)/tts.sample_rate:.1f}s)")

    # 非阻塞播放，立即返回，主循环继续监听新消息
    print_status("[播放] 正在播放...（发新消息可打断）")
    player.play(audio, sample_rate=tts.sample_rate, blocking=False)


# ─────────────────────── 主逻辑 ───────────────────────

def main():
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}    文字对话系统 (文字输入 + LLM + TTS){RESET}")
    print(f"{BOLD}{'='*50}{RESET}\n")

    print_status("[初始化] 连接 LLM...")
    try:
        llm = LLMClient()
    except ValueError as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)

    print_status("[初始化] 加载 TTS...")
    tts = create_tts_engine()

    player = AudioPlayer()

    # ── 启动输入源 ──
    msg_queue: queue.Queue = queue.Queue()

    if INPUT_MODE == "danmu":
        danmu_src = DanmuInputSource()
        print_status("[输入] 弹幕模式，通过 danmu_src.push(text) 推入消息\n")
    else:
        danmu_src = None
        # CLI 输入放子线程，主线程不阻塞
        t = threading.Thread(target=_cli_input_thread, args=(msg_queue,), daemon=True)
        t.start()
        print_status("[输入] 命令行模式，输入文字后回车发送（exit 退出）\n")

    print(f"{BOLD}{GREEN}✅ 所有组件就绪！{RESET}\n")

    while not _shutdown:
        # ── 取消息 ──
        if INPUT_MODE == "danmu":
            text = danmu_src.get_next()
            if text is None:
                time.sleep(0.05)
                continue
        else:
            try:
                text = msg_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if text is None:
                print(f"\n{YELLOW}[系统] 再见！{RESET}")
                break

        # ── 弹幕模式：视积压情况决定是否合并 ──
        if INPUT_MODE == "danmu" and danmu_src is not None:
            pending = danmu_src.pending_count()
            if pending > 0 and BATCH_WINDOW_SEC > 0:
                time.sleep(BATCH_WINDOW_SEC)
                more = danmu_src.get_batch(max_count=10)
                all_msgs = [text] + more
                text = "；".join(all_msgs)
                print_status(f"[合并] 合并 {len(all_msgs)} 条弹幕")
            elif pending > 1:
                # 积压较多且不合并：跳过旧消息，只回复最新一条
                danmu_src.get_batch(max_count=pending - 1)
                latest = danmu_src.get_next()
                if latest:
                    print_status(f"[队列] 跳过积压消息，处理最新一条")
                    text = latest

        # ── 有新消息：打断当前播放 ──
        if player.is_playing:
            print_status("[打断] 新消息，打断当前播放")
            player.stop()
            time.sleep(0.05)

        t0 = time.time()
        process_text(text, llm, tts, player)

        # 等待播放结束（或被下一条消息打断）
        while player.is_playing and not _shutdown:
            # CLI 模式：检查是否有新消息进来
            if INPUT_MODE != "danmu" and not msg_queue.empty():
                break
            # 弹幕模式：检查队列
            if INPUT_MODE == "danmu" and danmu_src and danmu_src.pending_count() > 0:
                break
            time.sleep(0.05)

        print_status(f"      (本轮总耗时 {time.time()-t0:.2f}s)\n")


if __name__ == "__main__":
    main()
