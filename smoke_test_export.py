"""
smoke_test_export.py

Loads the TorchScript model and runs a forward pass to verify export works.
"""

from pathlib import Path
import torch


def main():
    model_path = Path("artifacts/model_scripted.pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run:\n"
            f"  uv run python export_model.py"
        )

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    print("Smoke test OK.")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)  # should be [2, 10]


if __name__ == "__main__":
    main()