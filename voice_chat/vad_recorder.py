"""
vad_recorder.py
实时麦克风录音 + WebRTC VAD 自动断句模块
采样率：16000 Hz，帧长：30ms
"""

import collections
import os
import threading
import time
import numpy as np
import sounddevice as sd
import webrtcvad

# ─────────────────────────── 配置常量 ───────────────────────────
SAMPLE_RATE    = 16000          # webrtcvad 要求 8/16/32 kHz
FRAME_DURATION = 30             # 每帧时长 ms（10 / 20 / 30）
FRAME_SIZE     = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # 480 samples

INPUT_DEVICE   = int(os.getenv("MIC_DEVICE", "-1"))  # -1 = 系统默认; 改为设备ID可指定
VAD_AGGRESSIVENESS = 3          # 0~3，越大越激进（滤掉非语音越多）
SILENCE_THRESHOLD  = 1.0        # 静音持续多少秒视为句子结束
MIN_SPEECH_DURATION = 0.5       # 最短有效语音（秒），防误触发
MAX_RECORD_DURATION = 30.0      # 最长单次录音（秒），防卡死

# 滑动窗口：记录最近 N 帧的 VAD 结果，用于平滑判断
RING_BUFFER_FRAMES = int(SILENCE_THRESHOLD * 1000 / FRAME_DURATION)


class VoiceRecorder:
    """
    使用 webrtcvad 实现基于 VAD 的自动断句录音器。

    用法：
        recorder = VoiceRecorder()
        audio_np = recorder.record_once()   # 阻塞，直到检测到完整一句
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION,
        vad_aggressiveness: int = VAD_AGGRESSIVENESS,
        silence_threshold: float = SILENCE_THRESHOLD,
        min_speech_duration: float = MIN_SPEECH_DURATION,
        max_record_duration: float = MAX_RECORD_DURATION,
    ):
        self.sample_rate        = sample_rate
        self.frame_size         = int(sample_rate * frame_duration_ms / 1000)
        self.silence_threshold  = silence_threshold
        self.min_speech_frames  = int(min_speech_duration * 1000 / frame_duration_ms)
        self.max_record_frames  = int(max_record_duration * 1000 / frame_duration_ms)

        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self._ring_size = int(silence_threshold * 1000 / frame_duration_ms)

        # 内部状态（每次 record_once 重置）
        self._audio_buffer: list[np.ndarray] = []
        self._ring: collections.deque = collections.deque(maxlen=self._ring_size)        # VAD bool 结果
        self._audio_ring: collections.deque = collections.deque(maxlen=self._ring_size)  # 对应音频帧
        self._frame_queue: collections.deque = collections.deque()
        self._stop_event   = threading.Event()
        self._result_audio: np.ndarray | None = None

    # ──────────────────── 公开 API ────────────────────

    def record_once(self, pre_close_callback=None) -> np.ndarray:
        """
        阻塞，直到检测到一段完整语音（有声 → 静音结束）。
        返回：float32 numpy array，范围 [-1, 1]，采样率 16000 Hz

        pre_close_callback: 可选，在关闭麦克风流之前调用（用于无缝接管麦克风）
        """
        self._reset()

        print("[VAD] 正在聆听... (按 Ctrl+C 退出)")

        # 在子线程中开启音频流，主线程处理帧
        stream_thread = threading.Thread(target=self._stream_audio, daemon=True)
        stream_thread.start()

        result = self._process_frames()

        # 在关闭录音流之前先调用回调（无缝接管麦克风，避免 CoreAudio 释放再重新申请）
        if pre_close_callback is not None:
            try:
                pre_close_callback()
            except Exception:
                pass

        self._stop_event.set()
        stream_thread.join(timeout=3.0)
        # 额外等待 CoreAudio 完全释放设备
        time.sleep(0.1)

        return result

    # ──────────────────── 私有方法 ────────────────────

    def _reset(self):
        self._audio_buffer = []
        self._ring.clear()
        self._audio_ring.clear()
        self._frame_queue.clear()
        self._stop_event.clear()
        self._result_audio = None

    def _stream_audio(self):
        """sounddevice 回调：把原始 PCM 帧推入队列。"""
        def callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                pass  # 忽略轻微 underrun/overrun
            # indata shape: (frames, channels)，转为单声道 int16
            mono = indata[:, 0].copy()
            self._frame_queue.append(mono)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            device=INPUT_DEVICE if INPUT_DEVICE >= 0 else None,
            callback=callback,
        ):
            # 等待主线程发出停止信号
            self._stop_event.wait()

    def _process_frames(self) -> np.ndarray:
        """
        状态机：
          WAITING  → 等待语音开始
          RECORDING → 检测到语音，持续录制
          (静音超过阈值) → 结束录音
        """
        state = "WAITING"
        voiced_frames: list[np.ndarray] = []
        speech_frame_count = 0
        total_frames = 0
        silence_frame_count = 0  # 当前连续静音帧数
        silence_limit = int(self.silence_threshold * 1000 / FRAME_DURATION)

        while True:
            if self._stop_event.is_set():
                break

            # 从队列取帧（忙等 + 短暂 sleep 防止 CPU 100%）
            if not self._frame_queue:
                time.sleep(0.005)
                continue

            frame_float = self._frame_queue.popleft()

            # float32 → int16（webrtcvad 需要 int16 bytes）
            frame_int16 = (frame_float * 32767).astype(np.int16)
            frame_bytes = frame_int16.tobytes()

            # 确保帧长度正确（webrtcvad 严格要求）
            expected_bytes = self.frame_size * 2  # int16 = 2 bytes/sample
            if len(frame_bytes) != expected_bytes:
                continue

            try:
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            except Exception:
                is_speech = False

            total_frames += 1

            if state == "WAITING":
                self._ring.append(is_speech)
                self._audio_ring.append(frame_float)
                # 当滑动窗口中语音帧占多数时，认为用户开始说话
                if sum(self._ring) >= len(self._ring) * 0.75:
                    state = "RECORDING"
                    print("[VAD] 检测到说话，开始录音...")
                    # 把音频 ring buffer 里的历史帧加进去（避免截掉开头）
                    voiced_frames.extend(list(self._audio_ring))
                    speech_frame_count = len(voiced_frames)
                    silence_frame_count = 0

            elif state == "RECORDING":
                voiced_frames.append(frame_float)
                speech_frame_count += 1

                if is_speech:
                    silence_frame_count = 0
                else:
                    silence_frame_count += 1

                # 超过静音阈值 → 句子结束
                if silence_frame_count >= silence_limit:
                    if speech_frame_count >= self.min_speech_frames:
                        print("[VAD] 检测到静音，录音结束。")
                        break
                    else:
                        # 语音太短，重置，继续等待
                        print("[VAD] 语音过短，忽略，继续聆听...")
                        state = "WAITING"
                        voiced_frames = []
                        speech_frame_count = 0
                        silence_frame_count = 0
                        self._ring.clear()
                        self._audio_ring.clear()

                # 防止超长录音卡死
                if speech_frame_count >= self.max_record_frames:
                    print("[VAD] 达到最大录音时长，强制结束。")
                    break

        if not voiced_frames:
            # 没有有效语音，返回空数组
            return np.zeros(self.frame_size, dtype=np.float32)

        # 拼接成完整音频
        audio = np.concatenate(voiced_frames).astype(np.float32)
        # 裁剪掉末尾静音（保留少量）
        trim_frames = max(0, len(audio) - int(self.silence_threshold * 0.5 * self.sample_rate))
        audio = audio[:trim_frames] if trim_frames > 0 else audio

        return audio


# ─────────────────────────── 独立测试 ───────────────────────────
if __name__ == "__main__":
    import scipy.io.wavfile as wav
    recorder = VoiceRecorder()
    print("=== VAD 录音测试 ===")
    audio = recorder.record_once()
    print(f"录音完成，共 {len(audio)/SAMPLE_RATE:.2f} 秒")
    wav.write("test_output.wav", SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print("已保存为 test_output.wav")
