import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CrackDataset, get_data_splits
from models.unet import ResNet18UNet

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)

def compute_batch_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    intersection = (preds * targets).sum(dim=(2, 3))
    total_preds = preds.sum(dim=(2, 3))
    total_targets = targets.sum(dim=(2, 3))
    union = total_preds + total_targets - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (total_preds + total_targets + eps)
    
    return iou.mean().item(), dice.mean().item()

def train(num_epochs=12, batch_size=16, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Starting training on device: {device}')

    train_pairs, val_pairs, test_pairs = get_data_splits()
    print(f'Dataset splits: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}')

    train_ds = CrackDataset(train_pairs, is_train=True)
    val_ds = CrackDataset(val_pairs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNet18UNet(pretrained=True).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    metrics_history = []
    best_val_iou = 0.0
    best_model_path = os.path.join('models', 'best_model.pth')

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        # Train phase
        model.train()
        train_loss = 0.0
        train_iou_sum = 0.0
        train_dice_sum = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            iou, dice = compute_batch_metrics(logits, masks)
            train_loss += loss.item() * images.size(0)
            train_iou_sum += iou * images.size(0)
            train_dice_sum += dice * images.size(0)

        scheduler.step()

        train_loss /= len(train_ds)
        train_iou = train_iou_sum / len(train_ds)
        train_dice = train_dice_sum / len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou_sum = 0.0
        val_dice_sum = 0.0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                iou, dice = compute_batch_metrics(logits, masks)

                val_loss += loss.item() * images.size(0)
                val_iou_sum += iou * images.size(0)
                val_dice_sum += dice * images.size(0)

        val_loss /= len(val_ds)
        val_iou = val_iou_sum / len(val_ds)
        val_dice = val_dice_sum / len(val_ds)

        epoch_duration = time.time() - epoch_start
        print(f'Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_duration:.1f}s) - '
              f'Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f} | '
              f'Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}')

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_iou': round(train_iou, 4),
            'train_dice': round(train_dice, 4),
            'val_loss': round(val_loss, 4),
            'val_iou': round(val_iou, 4),
            'val_dice': round(val_dice, 4),
            'lr': float(optimizer.param_groups[0]['lr']),
            'duration_sec': round(epoch_duration, 2)
        }
        metrics_history.append(epoch_record)

        # Checkpoint best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice
            }, best_model_path)
            print(f'  --> Checkpoint saved: New best Val IoU: {val_iou:.4f}')

    total_time = time.time() - start_time
    print(f'Training complete in {total_time:.1f}s. Best Val IoU: {best_val_iou:.4f}')

    # Save metrics log
    metrics_path = os.path.join('logs', 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_val_iou': round(best_val_iou, 4),
            'total_epochs': num_epochs,
            'total_time_sec': round(total_time, 2),
            'epochs': metrics_history
        }, f, indent=2)
    print(f'Training metrics saved to {metrics_path}')

if __name__ == '__main__':
    train(num_epochs=12, batch_size=16)
