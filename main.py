import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CIFAR10Dataset
from src.model import SimpleCNN
from src.train import train_one_epoch
from src.evaluate import evaluate


def main(lr: float, batch_size: int, epochs: int, save_weights: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss:.4f} | "
            f"Test Accuracy: {accuracy:.4f}"
        )

    print("\nFinal Test Accuracy:", accuracy)

    # Save weights for export/deployment if requested
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

    # PR1 addition
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Save trained model weights to artifacts/model.pth",
    )

    args = parser.parse_args()

    main(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        save_weights=args.save_weights,
    )