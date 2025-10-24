import numpy as np
import librosa

def melspec(y, sr, n_mels=64, n_fft=1024, hop_length=320):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)
