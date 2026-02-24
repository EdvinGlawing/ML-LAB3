from pathlib import Path
from typing import List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -----------------------
# App Setup
# -----------------------
app = FastAPI(title="CIFAR10 Inference API")


# -----------------------
# Load Model (CPU only)
# -----------------------
MODEL_PATH = Path("artifacts/model_scripted.pt")

if not MODEL_PATH.exists():
    raise RuntimeError(
        "TorchScript model not found.\n"
        "Run: uv run python export_model.py"
    )

model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
model.eval()


# -----------------------
# Request Schema
# -----------------------
class PredictRequest(BaseModel):
    image: List[float] = Field(
        ...,
        description="Flattened CIFAR10 image (3072 float values: 3x32x32)"
    )


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):

    if len(request.image) != 3 * 32 * 32:
        raise HTTPException(
            status_code=400,
            detail="Image must contain exactly 3072 float values."
        )

    image_array = np.array(request.image, dtype=np.float32)
    image_array = image_array.reshape(1, 3, 32, 32)

    with torch.no_grad():
        input_tensor = torch.from_numpy(image_array)
        outputs = model(input_tensor)
        predicted_class = int(torch.argmax(outputs, dim=1))

    return {"prediction": predicted_class}