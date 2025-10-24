import json, torch, numpy as np
from .models.cnn_small import CNNSmall
from .utils.audio_io import load_wav
from .features.melspec import melspec

class InferenceModel:
    def __init__(self, model_path="models/cnn_small.pt", labels_path="models/labels.json", sr=16000, n_mels=64, n_fft=1024, hop=320, win_sec=2.0):
        self.labels = json.load(open(labels_path))["labels"]
        self.model = CNNSmall(len(self.labels))
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.sr=sr; self.n_mels=n_mels; self.n_fft=n_fft; self.hop=hop; self.window=int(win_sec*sr)

    def predict_path(self, path):
        y, sr = load_wav(path, target_sr=self.sr)
        if len(y) < self.window:
            y = np.pad(y, (0, self.window-len(y)))
        else:
            y = y[:self.window]
        S = melspec(y, self.sr, self.n_mels, self.n_fft, self.hop)
        x = torch.from_numpy(S).unsqueeze(0).unsqueeze(0).float()  # [1,1,M,T]
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy().tolist()
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
