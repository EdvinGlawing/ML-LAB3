# CIFAR-10 Deep Learning Pipeline & Deployment

This project implements a complete Deep Learning workflow using PyTorch, DVC, and FastAPI.

It includes:

- Modular PyTorch training pipeline
- Dataset tracking with DVC
- Experiment tracking
- Model export using TorchScript
- FastAPI inference service

---

# Project Structure

```
src/                   # ML modules (dataset, model, train, evaluate)
main.py                # Training entry point
export_model.py        # TorchScript export script
smoke_test_export.py   # Export verification script
app.py                 # FastAPI inference service
artifacts/             # Model binaries (gitignored)
experiments/           # Experiment results
data.dvc               # DVC dataset tracking
```

---

# Training

Train the model:

```bash
uv run python main.py --epochs 5 --batch_size 64 --lr 0.001
```

If you want to save weights for export:

```bash
uv run python main.py --epochs 5 --save-weights
```

This will:

- Train the CIFAR-10 model
- Save weights to `artifacts/model.pth`

---

# Model Export (TorchScript)

After training and saving weights, export the model for inference:

```bash
uv run python export_model.py
```

This generates:

```
artifacts/model_scripted.pt
```

The exported model is CPU-portable and ready for inference.

Export is done using `torch.jit.trace`.

---

# Smoke Test the Exported Model

```bash
uv run python smoke_test_export.py
```

Expected output:

```
Smoke test OK.
Input shape: torch.Size([2, 3, 32, 32])
Output shape: torch.Size([2, 10])
```

---

# FastAPI Inference Service

The project exposes a REST API using FastAPI for model inference.

## Start the API

```bash
uv run uvicorn app:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

Swagger UI provides interactive API testing.

---

## Available Endpoints

### GET /health

Health check endpoint.

Returns:

```json
{
  "status": "ok"
}
```

---

### POST /predict

Runs inference using the exported TorchScript model.

### Request Body (JSON)

```json
{
  "image": [3072 float values]
}
```

- Input must contain exactly 3072 floats
- Represents a flattened CIFAR-10 image (3x32x32)

---

### Response

```json
{
  "prediction": 0
}
```

- Returns the predicted class index (0–9)

---

# DVC Dataset Tracking

Dataset is tracked using DVC.

Raw dataset files are not stored in Git.

To restore dataset:

```bash
uv run dvc pull
```

---

# Experiments

Experiment results are stored in:

```
experiments/results.csv
```

This allows tracking different hyperparameter configurations and corresponding accuracies.

---

# Requirements

- Python 3.12
- uv
- PyTorch
- FastAPI
- Uvicorn
- DVC