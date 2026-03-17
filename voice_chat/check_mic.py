"""
check_mic.py
快速测量麦克风音量，用于校准打断门限
运行：python check_mic.py
对着麦克风说话，看"说话时"的音量数值
"""
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME_SIZE  = 480   # 30ms

print("测量麦克风音量（说话 5 秒，看数值范围）")
print("安静时看底噪，说话时看峰值")
print("Ctrl+C 退出\n")

frame_count = [0]

def callback(indata, frames, time_info, status):
    mono = indata[:, 0]
    vol = float(np.abs(mono).max())
    rms = float(np.sqrt(np.mean(mono**2)))
    bar = int(vol * 80)
    frame_count[0] += 1
    print(f"  峰值={vol:.4f}  RMS={rms:.4f}  {'█'*min(bar,80)}", end="\r", flush=True)

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=FRAME_SIZE, callback=callback):
    try:
        input()
    except KeyboardInterrupt:
        pass

print(f"\n共采集 {frame_count[0]} 帧")
