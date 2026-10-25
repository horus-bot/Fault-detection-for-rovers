from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.io import loadmat
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "src" / "models" / "bearing_cnn.pth"
X_PATH = ROOT / "data" / "output" / "X.npy"
GROUPS_PATH = ROOT / "data" / "output" / "groups.npy"
STATIC_PATH = Path(__file__).resolve().parent / "static"


class BearingCNN(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_checkpoint() -> tuple[nn.Module, dict[int, str], int]:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    class_names = {int(k): str(v) for k, v in checkpoint["class_names"].items()}
    window_size = int(checkpoint["window_size"])
    model = BearingCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, window_size


model, CLASS_NAMES, WINDOW_SIZE = load_checkpoint()
SAMPLES = np.load(X_PATH).astype(np.float32) if X_PATH.exists() else np.empty((0, 1, WINDOW_SIZE))
GROUPS = np.load(GROUPS_PATH, allow_pickle=True).astype(str) if GROUPS_PATH.exists() else np.array([])

app = FastAPI(title="BearingSense API", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


class PredictionRequest(BaseModel):
    signal: list[float] | None = Field(default=None, min_length=1)
    sample_index: int | None = Field(default=None, ge=0)


def prepare_signal(values: Any) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float32).reshape(-1)
    if signal.size != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly {WINDOW_SIZE} signal values, received {signal.size}.",
        )
    if not np.isfinite(signal).all():
        raise HTTPException(status_code=422, detail="Signal contains NaN or infinite values.")
    return signal


def explain(label: str, confidence: float, stats: dict[str, float]) -> str:
    if label == "Normal":
        return (
            "The vibration pattern is consistent with healthy bearing operation. "
            f"The model is {confidence:.1f}% confident; continue routine monitoring."
        )
    fault_advice = {
        "Inner-race fault": "Inspect the inner race and lubrication, and schedule a focused inspection.",
        "Outer-race fault": "Inspect the outer race, load zone, and mounting for localized damage.",
        "Ball fault": "Inspect rolling elements, cage condition, and lubrication for impact damage.",
    }
    return (
        f"The model detected a pattern associated with {label.lower()} "
        f"({confidence:.1f}% confidence). {fault_advice[label]} "
        f"The signal RMS is {stats['rms']:.3f} and its peak-to-peak range is "
        f"{stats['peak_to_peak']:.3f}."
    )


def predict(signal: np.ndarray, source: str | None = None) -> dict[str, Any]:
    tensor = torch.from_numpy(signal.reshape(1, 1, WINDOW_SIZE))
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()
    index = int(np.argmax(probabilities))
    label = CLASS_NAMES[index]
    stats = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "peak_to_peak": float(np.ptp(signal)),
        "crest_factor": float(np.max(np.abs(signal)) / max(np.sqrt(np.mean(signal**2)), 1e-8)),
    }
    return {
        "prediction": {"index": index, "label": label, "confidence": float(probabilities[index])},
        "probabilities": [
            {"label": CLASS_NAMES[i], "value": float(probabilities[i])}
            for i in range(len(probabilities))
        ],
        "signal": signal.tolist(),
        "stats": stats,
        "explanation": explain(label, float(probabilities[index] * 100), stats),
        "source": source,
        "window_size": WINDOW_SIZE,
    }


@app.get("/", response_class=FileResponse)
def index() -> Path:
    return STATIC_PATH / "index.html"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": True, "window_size": WINDOW_SIZE, "classes": CLASS_NAMES}


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    return {
        "count": int(len(SAMPLES)),
        "window_size": WINDOW_SIZE,
        "samples": [
            {"index": i, "source": GROUPS[i] if i < len(GROUPS) else f"Sample {i + 1}"}
            for i in range(len(SAMPLES))
        ],
    }


@app.post("/api/predict")
def predict_json(request: PredictionRequest) -> dict[str, Any]:
    if request.sample_index is not None:
        if request.sample_index >= len(SAMPLES):
            raise HTTPException(status_code=404, detail="Sample index is out of range.")
        return predict(SAMPLES[request.sample_index, 0], str(GROUPS[request.sample_index]))
    if request.signal is None:
        raise HTTPException(status_code=422, detail="Provide either signal or sample_index.")
    return predict(prepare_signal(request.signal), "Custom signal")


@app.post("/api/predict-file")
async def predict_file(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    suffix = Path(file.filename or "").suffix.lower()
    try:
        if suffix == ".npy":
            values = np.load(io.BytesIO(raw), allow_pickle=False)
        elif suffix == ".mat":
            mat = loadmat(io.BytesIO(raw))
            arrays = [v for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.size]
            if not arrays:
                raise ValueError("No numeric array found in MAT file.")
            values = max(arrays, key=lambda item: item.size)
        else:
            values = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter=",")
        return predict(prepare_signal(values), file.filename or "Uploaded signal")
    except (UnicodeDecodeError, ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not read signal file: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
