## Model Export (TorchScript)

This project supports exporting the trained CIFAR-10 model to TorchScript for deployment/inference.

1️⃣ Train and Save Weights
```bash
uv run python main.py --epochs 5 --save-weights
```

This will:

-Train the model
-Save weights to artifacts/model.pth

2️⃣ Export to TorchScript
```bash
uv run python export_model.py
```

This generates:

artifacts/model_scripted.pt

The exported model is CPU-portable and ready for inference.

3️⃣ Smoke Test the Exported Model
```bash
uv run python smoke_test_export.py
```

Expected output:

Smoke test OK.
Input shape: torch.Size([2, 3, 32, 32])
Output shape: torch.Size([2, 10])


Notes

-Model artifacts are stored in artifacts/

-Artifacts are excluded from Git via .gitignore

-Export is done using TorchScript (torch.jit.trace)

-The exported model does not require training code for inference