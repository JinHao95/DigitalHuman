"""
main.py
实时语音对话系统主循环

流程：
  麦克风 → VAD 断句 → Whisper ASR → OpenAI LLM → ChatTTS → 播放（可打断）

运行方式：
    export OPENAI_API_KEY=sk-...
    python main.py
"""

# 修复 torch + ctranslate2 重复加载 libiomp5.dylib 导致 abort 的问题
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import signal
import sys
import time
import numpy as np
import sounddevice as sd

# ─────────────────────── 组件导入 ───────────────────────
from vad_recorder import VoiceRecorder, SAMPLE_RATE, FRAME_SIZE, FRAME_DURATION, INPUT_DEVICE
from asr import ASREngine
from llm import LLMClient
from tts import create_tts_engine
from audio_player import AudioPlayer

# ─────────────────────── 颜色输出 ───────────────────────
RESET  = "\033[0m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"

def print_user(text: str):
    print(f"\n{BOLD}{CYAN}[用户]{RESET} {text}")

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


# ─────────────────────── 打断式播放（共享麦克风流方案）───────────────────────
_INTERRUPT_POLL_MS   = 200   # 检测间隔（ms）
_INTERRUPT_NOISE_MS  = 600   # 播放前采底噪时长（ms）
_INTERRUPT_SNRATIO   = 2.0   # RMS 超过底噪 N 倍才算人声
_INTERRUPT_MIN_ABS   = 0.040 # 绝对最低阈值
_INTERRUPT_DEBUG     = True

_MIC_DEVICE = INPUT_DEVICE if INPUT_DEVICE >= 0 else None

# 全局共享的麦克风流（录音结束后启动，一直保持到程序退出）
_shared_mic_frames = []
_shared_mic_stream = None

def start_shared_mic():
    """启动全局麦克风流，此后黄灯持续亮着。"""
    global _shared_mic_stream, _shared_mic_frames
    if _shared_mic_stream is not None:
        return
    _shared_mic_frames = []
    def _cb(indata, f, t, status):
        _shared_mic_frames.append(indata[:, 0].copy())
    try:
        _shared_mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME_SIZE, device=_MIC_DEVICE, callback=_cb)
        _shared_mic_stream.start()
        print_status(f"[麦克风] 共享流已启动 active={_shared_mic_stream.active}")
    except Exception as e:
        print_status(f"[麦克风] 启动失败: {e}")
        _shared_mic_stream = None

def stop_shared_mic():
    """关闭全局麦克风流。"""
    global _shared_mic_stream
    if _shared_mic_stream is not None:
        try:
            _shared_mic_stream.stop()
            _shared_mic_stream.close()
        except Exception:
            pass
        _shared_mic_stream = None


def play_with_interrupt(audio: np.ndarray, sample_rate: int) -> tuple[bool, np.ndarray]:
    """
    连续播放音频，用全局共享麦克风流检测打断。
    先用已采集的帧计算底噪，播放中超阈值则立即停止。
    """
    global _shared_mic_frames

    if _shared_mic_stream is None or not _shared_mic_stream.active:
        # 流不可用，直接播完
        print_status("  [打断] 麦克风流不可用，直接播放")
        sd.play(audio.reshape(-1, 1) if audio.ndim == 1 else audio, samplerate=sample_rate)
        sd.wait()
        return False, np.zeros(1, dtype=np.float32)

    # 先清空积压帧，采 _INTERRUPT_NOISE_MS 底噪
    _shared_mic_frames.clear()
    time.sleep(_INTERRUPT_NOISE_MS / 1000.0)
    noise_frames = list(_shared_mic_frames)
    _shared_mic_frames.clear()

    if noise_frames:
        noise_rms = max(float(np.sqrt(np.mean(np.concatenate(noise_frames) ** 2))), 1e-6)
    else:
        noise_rms = 0.01
    threshold = max(noise_rms * _INTERRUPT_SNRATIO, _INTERRUPT_MIN_ABS)
    if _INTERRUPT_DEBUG:
        print(f"  [打断] 底噪 RMS={noise_rms:.5f}  打断阈值={threshold:.5f}", flush=True)

    interrupted = False

    # 开始播放
    audio_2d = audio.reshape(-1, 1) if audio.ndim == 1 else audio
    sd.play(audio_2d, samplerate=sample_rate)

    poll_interval = _INTERRUPT_POLL_MS / 1000.0
    while sd.get_stream().active:
        time.sleep(poll_interval)
        if not _shared_mic_frames:
            continue
        recent = np.concatenate(list(_shared_mic_frames))
        _shared_mic_frames.clear()
        rms = float(np.sqrt(np.mean(recent ** 2)))
        if _INTERRUPT_DEBUG:
            marker = " <<< 说话" if rms >= threshold else ""
            print(f"  [打断] 麦克风 RMS={rms:.5f}  阈值={threshold:.5f}{marker}", flush=True)
        if rms >= threshold:
            sd.stop()
            interrupted = True
            break

    sd.wait()

    if interrupted:
        print(f"  [打断] 检测到人声，打断播放！", flush=True)
        return True, np.zeros(1, dtype=np.float32)

    return False, np.zeros(1, dtype=np.float32)


# ─────────────────────── 主逻辑 ───────────────────────
def main():
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}    实时语音对话系统 (VAD + Whisper + LLM + TTS){RESET}")
    print(f"{BOLD}{'='*50}{RESET}")
    print("按 Ctrl+C 退出\n")

    print_status("[初始化] 加载 VAD 录音器...")
    recorder = VoiceRecorder()

    print_status("[初始化] 加载 ASR（Whisper）...")
    asr = ASREngine()

    print_status("[初始化] 连接 LLM（OpenAI）...")
    try:
        llm = LLMClient()
    except ValueError as e:
        print(f"\n❌ {e}\n")
        sys.exit(1)

    print_status("[初始化] 加载 TTS...")
    tts = create_tts_engine()

    print(f"\n{BOLD}{GREEN}✅ 所有组件就绪，开始对话！{RESET}\n")
    print_status("提示：说话即可，AI 回复时打断说话可立即接管。\n")

    round_num = 0

    while not _shutdown:
        round_num += 1
        t0 = time.time()

        # ① 录音
        print_status(f"─── 第 {round_num} 轮 | 正在聆听... ───")
        try:
            # pre_close_callback: VAD 检测到句子结束、但录音流尚未关闭时启动共享麦克风流
            # 这样可以无缝接管 CoreAudio 麦克风，保持黄灯持续亮着
            audio_np = recorder.record_once(pre_close_callback=start_shared_mic)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[录音错误] {e}")
            import traceback; traceback.print_exc()
            time.sleep(0.5)
            continue

        if audio_np is None or len(audio_np) < 1600:
            print_status("[跳过] 音频太短，重新聆听...")
            stop_shared_mic()
            continue

        # ② ASR
        print_status("[ASR ] 正在识别...")
        user_text = asr.transcribe(audio_np)
        asr_time = time.time() - t0

        if not user_text or not user_text.strip():
            print_status(f"[ASR ] 未识别到文本（{asr_time:.2f}s），重新聆听...")
            continue

        print_user(user_text)
        print_status(f"      (ASR {asr_time:.2f}s)")

        # ③ LLM
        print_status("[LLM ] 正在生成回复...")
        print(f"{BOLD}{GREEN}[AI  ]{RESET} ", end="", flush=True)
        t1 = time.time()
        ai_text = ""
        try:
            for chunk in llm.chat_stream(user_text):
                print(chunk, end="", flush=True)
                ai_text += chunk
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"\n[LLM 错误] {e}")
            continue
        print()
        print_status(f"      (LLM {time.time()-t1:.2f}s)")

        if not ai_text.strip():
            continue

        # ④ TTS
        print_status("[TTS ] 正在合成语音...")
        t2 = time.time()
        try:
            audio_out = tts.synthesize(ai_text)
        except Exception as e:
            print(f"[TTS 错误] {e}")
            continue
        print_status(f"      (TTS {time.time()-t2:.2f}s，音频 {len(audio_out)/tts.sample_rate:.1f}s)")

        # ⑤ 播放（可打断）
        print_status("[播放] 正在播放...（说话可打断）")
        try:
            was_interrupted, captured = play_with_interrupt(audio_out, tts.sample_rate)
        except Exception as e:
            print(f"[播放错误] {e}")
            continue

        if was_interrupted:
            print_status("[打断] 用户打断了播放，重新聆听...")
        else:
            pending_audio = None

        # 播放结束，关闭共享麦克风流（下一轮录音前会重新开）
        stop_shared_mic()

        print_status(f"      (本轮总耗时 {time.time()-t0:.2f}s)\n")


if __name__ == "__main__":
    main()
