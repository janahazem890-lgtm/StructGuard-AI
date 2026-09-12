"""
Reads logs/training_metrics.json (produced by train.py) and generates
loss / IoU / Dice curves (train vs. validation) as PNG files.

Usage:
    python plot_training_curves.py
Outputs:
    logs/eda/loss_curve.png
    logs/eda/iou_curve.png
    logs/eda/dice_curve.png
"""
import os
import json
import matplotlib.pyplot as plt


def plot_training_curves(metrics_path="logs/training_metrics.json", output_dir="logs/eda"):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"{metrics_path} not found. Run train.py first to generate real training metrics."
        )

    with open(metrics_path, "r") as f:
        data = json.load(f)

    epochs_data = data.get("epochs", [])
    if not epochs_data:
        raise ValueError(f"No per-epoch records found in {metrics_path}.")

    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    train_iou = [e["train_iou"] for e in epochs_data]
    val_iou = [e["val_iou"] for e in epochs_data]
    train_dice = [e["train_dice"] for e in epochs_data]
    val_dice = [e["val_dice"] for e in epochs_data]

    # ---- Loss curve ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss", color="#38bdf8")
    plt.plot(epochs, val_loss, marker="o", label="Validation Loss", color="#f472b6")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCE + Dice)")
    plt.title("Training vs Validation Loss", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # ---- IoU curve ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_iou, marker="o", label="Train IoU", color="#38bdf8")
    plt.plot(epochs, val_iou, marker="o", label="Validation IoU", color="#f472b6")
    best_val_iou = data.get("best_val_iou")
    if best_val_iou is not None:
        plt.axhline(best_val_iou, color="#22c55e", linestyle="--",
                     label=f"Best Val IoU ({best_val_iou:.4f})")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training vs Validation IoU", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    iou_path = os.path.join(output_dir, "iou_curve.png")
    plt.savefig(iou_path, dpi=150)
    plt.close()

    # ---- Dice curve ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_dice, marker="o", label="Train Dice", color="#38bdf8")
    plt.plot(epochs, val_dice, marker="o", label="Validation Dice", color="#f472b6")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Training vs Validation Dice", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    dice_path = os.path.join(output_dir, "dice_curve.png")
    plt.savefig(dice_path, dpi=150)
    plt.close()

    print(f"Saved: {loss_path}")
    print(f"Saved: {iou_path}")
    print(f"Saved: {dice_path}")
    print(f"Total epochs trained: {data.get('total_epochs')}")
    print(f"Best validation IoU achieved: {data.get('best_val_iou')}")


if __name__ == "__main__":
    plot_training_curves()
