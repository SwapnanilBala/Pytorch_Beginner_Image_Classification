"""
train.py  (Beginner-friendly CIFAR-10 training script)

What this script does:
1) Downloads CIFAR-10 (if not already present)
2) Builds a small CNN (from model.py)
3) Trains for N epochs (default 5)
4) Evaluates after each epoch on the test set
5) Saves the best model weights to checkpoints/best.pt
6) Shows visuals:
   - A grid of sample images from the training set (optional)
   - Training curves (loss + accuracy) at the end
"""

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN


# ----------------------------
# Helper function: accuracy
# ----------------------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: model raw outputs shaped (batch_size, num_classes)
    y:      true labels shaped (batch_size,)

    We convert logits -> predicted class using argmax,
    then compute the fraction of correct predictions.
    """
    preds = logits.argmax(dim=1)            # predicted class indices
    return (preds == y).float().mean().item()  # convert True/False -> 1/0 -> average


# ----------------------------
# Helper function: evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    Evaluate model on a dataset loader (usually validation/test).

    @torch.no_grad() means:
    - Don't track gradients (faster + less memory)
    - Because we're NOT training here, only evaluating
    """
    model.eval()  # important: turns OFF dropout randomness, etc.

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    loss_fn = nn.CrossEntropyLoss()

    for xb, yb in loader:
        # Move batch to GPU/CPU device
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass
        logits = model(xb)

        # Compute loss
        loss = loss_fn(logits, yb)

        # Accumulate stats
        bs = xb.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs

    # Return average loss and average accuracy
    return total_loss / n, total_acc / n


# ----------------------------
# Helper function: visualize a batch
# ----------------------------
def show_batch(images, labels, classes, mean, std, n=16):
    """
    Show a grid of images from a batch (for visualization).

    images: (B, 3, 32, 32) tensors that are currently NORMALIZED
    labels: (B,) class indices
    classes: list of label names (CIFAR-10 provides this)
    mean/std: used to "denormalize" images so they look correct on screen
    n: number of images to show (must be a perfect square like 16, 25, 36)
    """
    # Denormalize for display: img = img * std + mean
    # (We do this because Normalize(mean, std) changes the pixel values.)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)

    imgs = images.cpu() * std_t + mean_t  # move to CPU for plotting and denormalize
    imgs = imgs.clamp(0, 1)              # keep pixel values in [0,1] for display

    k = int(n ** 0.5)  # grid will be k x k
    fig, axes = plt.subplots(k, k, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(k * k):
        # PyTorch images are (C, H, W) but matplotlib expects (H, W, C)
        axes[i].imshow(imgs[i].permute(1, 2, 0), interpolation="nearest")
        axes[i].set_title(classes[labels[i].item()], fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # ----------------------------
    # 1) Read command-line arguments
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="How many epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data", help="Where to store CIFAR-10 data")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--show-batch", action="store_true", help="If set, show a sample image grid")
    args = parser.parse_args()

    # ----------------------------
    # 2) Pick device: GPU if available, else CPU
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        print("cuda version (torch):", torch.version.cuda)

    # ----------------------------
    # 3) Define transforms (preprocessing)
    # ----------------------------
    # CIFAR-10 "standard" normalization stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms include random augmentation
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # slight random crop
        transforms.RandomHorizontalFlip(),         # random flip
        transforms.ToTensor(),                     # convert PIL -> Tensor [0,1]
        transforms.Normalize(mean, std),           # normalize
    ])

    # Test transforms should be deterministic (no randomness)
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ----------------------------
    # 4) Load CIFAR-10 datasets (downloads first time)
    # ----------------------------
    data_dir = Path(args.data_dir)
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)

    # Clean preview set (no random augmentation)
    preview_ds = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_tfms)
    preview_loader = DataLoader(preview_ds, batch_size=16, shuffle=True)
    classes = preview_ds.classes
    xb, yb = next(iter(preview_loader))
    show_batch(xb, yb, classes, mean, std, n=16)


    print("train set size:", len(train_ds))
    print("test  set size:", len(test_ds))

    # ----------------------------
    # 5) Create DataLoaders (batching + shuffling)
    # ----------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,                        # shuffle training data
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),  # speeds CPU->GPU copies
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optional: show a batch of images so you can SEE the dataset
    if args.show_batch:
        classes = train_ds.classes  # CIFAR-10 label names
        xb, yb = next(iter(train_loader))
        show_batch(xb[:16], yb[:16], classes, mean, std, n=16)

    # ----------------------------
    # 6) Build model, loss, optimizer
    # ----------------------------
    model = SmallCNN(num_classes=10).to(device)

    # Print where the model lives (CPU or GPU)
    print("model device:", next(model.parameters()).device)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------
    # 7) Create checkpoint folder (for saving best model)
    # ----------------------------
    checkpoints = Path("checkpoints")
    checkpoints.mkdir(exist_ok=True)

    # ----------------------------
    # 8) Lists to store training history (for plots)
    # ----------------------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_acc = 0.0
    t0 = time.time()

    # ----------------------------
    # 9) Training loop
    # ----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()  # enables training mode (dropout ON, etc.)

        running_loss = 0.0
        running_acc = 0.0
        n = 0

        # Loop through batches
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # 1) Clear old gradients
            opt.zero_grad(set_to_none=True)

            # 2) Forward pass (predictions)
            logits = model(xb)

            # 3) Compute loss
            loss = loss_fn(logits, yb)

            # 4) Backward pass (compute gradients)
            loss.backward()

            # 5) Update weights
            opt.step()

            # Track metrics
            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(logits, yb) * bs
            n += bs

        # Averages over the whole training set
        train_loss = running_loss / n
        train_acc = running_acc / n

        # Evaluate on test set
        val_loss, val_acc = evaluate(model, test_loader, device)

        # Save history for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print epoch summary
        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Save best model so far
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "val_acc": best_acc, "epoch": epoch},
                checkpoints / "best.pt",
            )
            print(f"  saved checkpoints/best.pt (best_acc={best_acc:.4f})")

    # ----------------------------
    # 10) Final print + show training curves
    # ----------------------------
    print("done. best_acc:", round(best_acc, 4), "| elapsed:", round(time.time() - t0, 1), "sec")

    # Plot curves
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="train acc")
    plt.plot(val_accs, label="val acc")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



# To run effectively put this in the pycharm terminal : python train.py --epochs 5 --show-batch
