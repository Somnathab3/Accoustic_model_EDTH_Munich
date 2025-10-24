import soundfile as sf
import numpy as np

def load_wav(path, target_sr=None):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:  # mono-ize
        y = np.mean(y, axis=1)
    if target_sr and sr != target_sr:
        # Lazy resample using librosa for simplicity
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype("float32"), sr
