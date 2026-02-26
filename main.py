import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import CIFAR10Dataset
from src.model import SimpleCNN
from src.train import train
from src.evaluate import evaluate


def get_safe_device():
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major <= 9:
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        except:
            return torch.device("cpu")
    return torch.device("cpu")


def main(lr, batch_size, epochs, save_weights):
    device = get_safe_device()
    print("Using device:", device)

    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleCNN()

    model = train(
        model=model,
        train_dataset=train_dataset,
        device=device,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
    )

    acc = evaluate(model, test_loader, device)
    print("Final Test Accuracy:", acc)

    if save_weights:
        Path("artifacts").mkdir(exist_ok=True)
        weights_path = Path("artifacts/model.pth")
        torch.save(model.state_dict(), weights_path)
        print(f"Saved weights to: {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save-weights", action="store_true")

    args = parser.parse_args()

    main(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_weights=args.save_weights,
    )