import os
import time
import numpy as np
import soundfile as sf

def wav_writer_proc(record_queue, filename, sample_rate=24000):
    """音频录制子进程"""
    abs_filename = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
    try:
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    except:
        # 如果文件被占用，尝试加时间戳
        abs_filename = abs_filename.replace(".wav", f"_{int(time.time())}.wav")
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
        
    try:
        while True:
            chunk = record_queue.get()
            if chunk is None: break
            if isinstance(chunk, str) and chunk == "CLEAR": continue
            f.write(chunk.flatten().astype(np.float32))
            f.flush()
    except: 
        pass
    finally: 
        f.close()
