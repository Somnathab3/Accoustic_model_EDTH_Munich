from fastapi import FastAPI, UploadFile, File
from ..infer import InferenceModel
import tempfile, shutil

app = FastAPI(title="Acoustic Drone Detector")

# Lazy-load model on first request
_model = None
def get_model():
    global _model
    if _model is None:
        _model = InferenceModel()
    return _model

@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp.flush()
        m = get_model()
        probs = m.predict_path(tmp.name)
    return {"ok": True, "probs": probs}
