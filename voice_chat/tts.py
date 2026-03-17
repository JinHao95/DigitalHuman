"""
tts.py
使用 ChatTTS 生成语音
输入：文本字符串
输出：numpy float32 音频数组（24kHz）

ChatTTS 安装：pip install chattts
项目主页：https://github.com/2noise/ChatTTS
"""

import os
import re
import numpy as np

# ─────────────────────────── 配置 ───────────────────────────
# 音色种子（固定种子 = 固定音色，设为 None 则随机）
SPEAKER_SEED = int(os.getenv("CHATTTS_SPEAKER_SEED", "2222"))

# 推理参数
TEMPERATURE   = float(os.getenv("CHATTTS_TEMPERATURE",   "0.3"))
TOP_P         = float(os.getenv("CHATTTS_TOP_P",         "0.7"))
TOP_K         = int(os.getenv("CHATTTS_TOP_K",           "20"))
SPEED         = int(os.getenv("CHATTTS_SPEED",           "5"))    # 1~10

# 是否使用 oral（口语化）/ laugh（笑声）/ break（停顿）增强
USE_ORAL    = int(os.getenv("CHATTTS_ORAL",    "2"))   # 0~9
USE_LAUGH   = int(os.getenv("CHATTTS_LAUGH",   "0"))   # 0~2
USE_BREAK   = int(os.getenv("CHATTTS_BREAK",   "6"))   # 0~7

CHATTTS_SAMPLE_RATE = 24000   # ChatTTS 输出采样率固定为 24kHz


class TTSEngine:
    """
    ChatTTS TTS 封装。

    用法：
        tts = TTSEngine()
        audio_np = tts.synthesize("你好，我是你的语音助手。")
    """

    def __init__(self, speaker_seed: int | None = SPEAKER_SEED):
        print("[TTS] 加载 ChatTTS 模型...")
        try:
            import ChatTTS
            self.chat = ChatTTS.Chat()
            self.chat.load(compile=False)   # compile=True 可提速但首次耗时
            print("[TTS] ChatTTS 加载完成。")
        except ImportError:
            raise ImportError(
                "ChatTTS 未安装，请执行：pip install chattts\n"
                "或访问 https://github.com/2noise/ChatTTS"
            )

        # 固定音色
        self._spk = self._make_speaker(speaker_seed)
        self.sample_rate = CHATTTS_SAMPLE_RATE

    # ──────────────────── 公开 API ────────────────────

    def synthesize(self, text: str) -> np.ndarray:
        """
        将文本合成为音频。

        参数：
            text: 要合成的文本
        返回：
            float32 numpy array，采样率 24000 Hz，范围 [-1, 1]
            失败时返回空数组
        """
        if not text or not text.strip():
            return np.zeros(1, dtype=np.float32)

        # 清理不适合 TTS 的字符
        clean_text = self._clean_text(text)
        if not clean_text:
            return np.zeros(1, dtype=np.float32)

        # 构建 infer_code 参数（控制语速、情感）
        params_infer = self.chat.InferCodeParams(
            spk_emb=self._spk,
            temperature=TEMPERATURE,
            top_P=TOP_P,
            top_K=TOP_K,
            manual_seed=None,
        )

        # 构建 refine_text 参数（口语化增强）
        params_refine = self.chat.RefineTextParams(
            prompt=f"[oral_{USE_ORAL}][laugh_{USE_LAUGH}][break_{USE_BREAK}]",
            top_P=TOP_P,
            top_K=TOP_K,
            temperature=TEMPERATURE,
        )

        try:
            wavs = self.chat.infer(
                texts=[clean_text],
                params_refine_text=params_refine,
                params_infer_code=params_infer,
                use_decoder=True,
                skip_refine_text=False,
            )

            if wavs and len(wavs) > 0 and wavs[0] is not None:
                audio = np.array(wavs[0], dtype=np.float32)
                # 归一化（防止爆音）
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio / peak * 0.9
                return audio
            else:
                print("[TTS] 合成失败：输出为空")
                return np.zeros(1, dtype=np.float32)

        except Exception as e:
            print(f"[TTS] 合成出错: {e}")
            return np.zeros(1, dtype=np.float32)

    # ──────────────────── 私有方法 ────────────────────

    def _make_speaker(self, seed: int | None) -> object:
        """生成固定音色 embedding。"""
        try:
            import torch
            if seed is not None:
                torch.manual_seed(seed)
            return self.chat.sample_random_speaker()
        except Exception as e:
            print(f"[TTS] 生成音色失败: {e}，将使用随机音色")
            return self.chat.sample_random_speaker()

    @staticmethod
    def _clean_text(text: str) -> str:
        """移除 Markdown、emoji、特殊符号等不适合朗读的内容。"""
        # 去掉 emoji 及所有 Unicode 符号类字符（Emoji、杂项符号、箭头等）
        text = re.sub(r"[\U00010000-\U0010ffff]", "", text)   # 辅助平面（大部分 emoji）
        text = re.sub(r"[\u2600-\u27BF]", "", text)            # 杂项符号、箭头、Dingbats
        text = re.sub(r"[\u2300-\u23FF]", "", text)            # 技术符号
        text = re.sub(r"[\uFE00-\uFE0F]", "", text)            # 变体选择符
        # 去掉 Markdown 标记
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"`+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [text](url) → text
        # 去掉多余空白
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ─────────────────────────── 备用 TTS（edge-tts / pyttsx3）───────────────────────────
class FallbackTTSEngine:
    """
    当 ChatTTS 不可用时的备用方案：使用 edge-tts（微软在线 TTS）。
    需要安装：pip install edge-tts
    """

    VOICE    = "zh-CN-XiaoxiaoNeural"
    SAMPLE_RATE = 24000

    def __init__(self):
        print("[TTS-Fallback] 使用 edge-tts 作为备用 TTS")
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise ImportError("备用 TTS 也不可用，请安装：pip install edge-tts")
        self.sample_rate = self.SAMPLE_RATE

    def synthesize(self, text: str) -> np.ndarray:
        import asyncio
        import io
        import edge_tts
        import soundfile as sf

        clean = TTSEngine._clean_text(text)
        if not clean:
            return np.zeros(1, dtype=np.float32)

        async def _run():
            communicate = edge_tts.Communicate(clean, self.VOICE)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf

        try:
            buf = asyncio.run(_run())
            audio, sr = sf.read(buf, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            # 如果采样率不同，简单重采样
            if sr != self.SAMPLE_RATE:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, self.SAMPLE_RATE)
                audio = resample_poly(audio, self.SAMPLE_RATE // g, sr // g)
            return audio.astype(np.float32)
        except Exception as e:
            print(f"[TTS-Fallback] 合成出错: {e}")
            return np.zeros(1, dtype=np.float32)


def create_tts_engine() -> TTSEngine | FallbackTTSEngine:
    """
    工厂函数：优先使用 ChatTTS，不可用则自动降级到 edge-tts。
    """
    try:
        return TTSEngine()
    except Exception as e:
        print(f"[TTS] ChatTTS 初始化失败({e})，尝试 edge-tts 备用方案...")
        return FallbackTTSEngine()


# ─────────────────────────── 独立测试 ───────────────────────────
if __name__ == "__main__":
    import scipy.io.wavfile as wav
    tts = create_tts_engine()
    text = "你好，我是你的语音助手，很高兴认识你！"
    print(f"[TTS] 合成文本: {text}")
    audio = tts.synthesize(text)
    print(f"[TTS] 音频长度: {len(audio)/tts.sample_rate:.2f} 秒")
    out_path = "tts_test.wav"
    wav.write(out_path, tts.sample_rate, (audio * 32767).astype(np.int16))
    print(f"[TTS] 已保存到 {out_path}")
