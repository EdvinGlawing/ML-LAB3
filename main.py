import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import CIFAR10Dataset
from src.model import SimpleCNN
from src.train import train
from src.evaluate import evaluate


def main(lr, batch_size, epochs, save_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)

    # Your train() builds its own train_loader internally,
    # so we only need the test_loader here.
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleCNN()

    # Train using your existing train() function
    model = train(
        model=model,
        train_dataset=train_dataset,
        device=device,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
    )

    # Evaluate after training
    acc = evaluate(model, test_loader, device)
    print("Final Test Accuracy:", acc)

    # Save weights if requested
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