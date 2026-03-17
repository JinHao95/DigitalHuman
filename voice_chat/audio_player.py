"""
audio_player.py
播放 numpy 音频数组（使用 sounddevice）
支持阻塞播放和非阻塞播放
"""

import threading
import numpy as np
import sounddevice as sd


class AudioPlayer:
    """
    音频播放器，基于 sounddevice。

    用法：
        player = AudioPlayer()
        player.play(audio_np, sample_rate=24000)          # 阻塞直到播放完
        player.play_async(audio_np, sample_rate=24000)    # 非阻塞
        player.stop()                                      # 立即停止
    """

    def __init__(self):
        self._lock   = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._playing = threading.Event()

    # ──────────────────── 公开 API ────────────────────

    def play(self, audio: np.ndarray, sample_rate: int = 24000, blocking: bool = True):
        """
        播放音频。

        参数：
            audio:       float32 numpy array
            sample_rate: 采样率
            blocking:    True = 阻塞直到播放完成
        """
        if audio is None or len(audio) == 0:
            return

        audio_f32 = self._prepare(audio)

        if blocking:
            self._play_blocking(audio_f32, sample_rate)
        else:
            t = threading.Thread(
                target=self._play_blocking,
                args=(audio_f32, sample_rate),
                daemon=True,
            )
            t.start()

    def play_async(self, audio: np.ndarray, sample_rate: int = 24000):
        """非阻塞播放（立即返回）。"""
        self.play(audio, sample_rate, blocking=False)

    def stop(self):
        """立即停止当前播放。"""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        self._playing.clear()

    def wait(self):
        """等待当前异步播放完成。"""
        self._playing.wait()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    # ──────────────────── 私有方法 ────────────────────

    def _play_blocking(self, audio: np.ndarray, sample_rate: int):
        """实际阻塞播放逻辑。"""
        self._playing.set()
        try:
            # 确保是 2D (frames, channels) 或 1D (frames,)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            with self._lock:
                # 使用 sounddevice 的简单 play/wait 接口
                pass  # 不在锁内持有流，避免死锁

            sd.play(audio, samplerate=sample_rate)
            sd.wait()  # 阻塞直到播放完成

        except Exception as e:
            print(f"[Player] 播放出错: {e}")
        finally:
            self._playing.clear()

    @staticmethod
    def _prepare(audio: np.ndarray) -> np.ndarray:
        """确保音频格式正确：float32，范围 [-1, 1]。"""
        audio = np.array(audio, dtype=np.float32)
        # 如果是 int16
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32767.0
        # 防止爆音
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak * 0.98
        return audio


# ─────────────────────── 便捷函数 ───────────────────────
_default_player = None

def play_audio(audio: np.ndarray, sample_rate: int = 24000, blocking: bool = True):
    """
    模块级便捷函数，复用同一个播放器实例。
    """
    global _default_player
    if _default_player is None:
        _default_player = AudioPlayer()
    _default_player.play(audio, sample_rate, blocking=blocking)


# ─────────────────────────── 独立测试 ───────────────────────────
if __name__ == "__main__":
    import math
    print("[Player] 播放 440Hz 正弦波测试音（1秒）...")
    sr = 24000
    t_arr = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (np.sin(2 * math.pi * 440 * t_arr) * 0.5).astype(np.float32)
    play_audio(tone, sr)
    print("[Player] 播放完成。")
