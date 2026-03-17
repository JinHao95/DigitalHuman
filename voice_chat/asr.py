"""
asr.py
基于 faster-whisper 的语音识别模块（无 numba 依赖，Apple Silicon 友好）
输入：numpy float32 音频（16kHz）
输出：识别文本字符串
"""

import os
import numpy as np
from faster_whisper import WhisperModel

# 模型选择：tiny / base / small / medium / large-v3
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# 语言提示（填写可加速，留空自动检测）
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "zh")

# Apple Silicon：用 cpu + int8 量化，速度快且无兼容问题
DEVICE    = os.getenv("WHISPER_DEVICE",    "cpu")
COMPUTE   = os.getenv("WHISPER_COMPUTE",   "int8")


class ASREngine:
    """
    faster-whisper ASR 封装。

    用法：
        asr = ASREngine()
        text = asr.transcribe(audio_np)  # audio_np: float32, 16kHz
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        language: str   = LANGUAGE,
        device: str     = DEVICE,
        compute: str    = COMPUTE,
    ):
        print(f"[ASR] 加载 faster-whisper 模型: {model_name} (device={device}, compute={compute}) ...")
        self.model    = WhisperModel(model_name, device=device, compute_type=compute)
        self.language = language or None
        print("[ASR] 模型加载完成。")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        将音频 numpy 数组转换为文本。

        参数：
            audio: float32 numpy array，采样率 16000 Hz，范围 [-1, 1]
        返回：
            识别文本（str），失败时返回空字符串
        """
        if audio is None or len(audio) == 0:
            return ""

        audio_f32 = audio.astype(np.float32)

        # 振幅太小（几乎无声）直接跳过
        if np.abs(audio_f32).max() < 1e-4:
            return ""

        try:
            kwargs = {
                "beam_size": 5,
                "vad_filter": True,          # 内置 VAD 过滤静音，减少幻觉
                "vad_parameters": {"min_silence_duration_ms": 500},
            }
            if self.language:
                kwargs["language"] = self.language

            segments, _info = self.model.transcribe(audio_f32, **kwargs)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text

        except Exception as e:
            print(f"[ASR] 识别出错: {e}")
            return ""


# ─────────────────────────── 独立测试 ───────────────────────────
if __name__ == "__main__":
    import sys
    import scipy.io.wavfile as wav

    asr = ASREngine()
    if len(sys.argv) > 1:
        sr, data = wav.read(sys.argv[1])
        audio = data.astype(np.float32) / 32767.0 if data.dtype == np.int16 else data.astype(np.float32)
        text = asr.transcribe(audio)
    else:
        dummy = np.zeros(16000, dtype=np.float32)
        text = asr.transcribe(dummy)
    print(f"[ASR] 识别结果: {text!r}")
