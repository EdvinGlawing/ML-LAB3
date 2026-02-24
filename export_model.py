"""
export_model.py

Exports the trained CIFAR-10 model to TorchScript for deployment/inference.

Expected input:
- artifacts/model.pth (state_dict) created by running:
    uv run python main.py --save-weights

Output:
- artifacts/model_scripted.pt
"""

from pathlib import Path

import torch

from src.model import SimpleCNN


def export_torchscript():
    artifacts_dir = Path("artifacts")
    weights_path = artifacts_dir / "model.pth"
    output_path = artifacts_dir / "model_scripted.pt"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Train and save weights first:\n"
            f"  uv run python main.py --save-weights"
        )

    # Export on CPU to be maximally portable
    device = torch.device("cpu")

    model = SimpleCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Example input for tracing (CIFAR shape)
    example_input = torch.randn(1, 3, 32, 32, device=device)

    traced = torch.jit.trace(model, example_input)
    traced.save(str(output_path))

    print(f"Exported TorchScript model to: {output_path}")


if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    export_torchscript()