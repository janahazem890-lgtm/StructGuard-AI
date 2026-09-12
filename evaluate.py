import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CrackDataset, get_data_splits
from models.unet import ResNet18UNet

def evaluate(checkpoint_path='models/best_model.pth', threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on device: {device}')

    _, _, test_pairs = get_data_splits()
    print(f'Test dataset size: {len(test_pairs)} samples')

    test_ds = CrackDataset(test_pairs, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Load model
    model = ResNet18UNet(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    sample_ious = []
    sample_dices = []

    eps = 1e-7

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            for p, m in zip(preds, masks):
                p_flat = p.view(-1)
                m_flat = m.view(-1)

                tp = ((p_flat == 1) & (m_flat == 1)).sum().item()
                fp = ((p_flat == 1) & (m_flat == 0)).sum().item()
                fn = ((p_flat == 0) & (m_flat == 1)).sum().item()
                tn = ((p_flat == 0) & (m_flat == 0)).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                inter = tp
                union = tp + fp + fn
                iou = (inter + eps) / (union + eps)
                dice = (2.0 * inter + eps) / (2.0 * tp + fp + fn + eps)

                sample_ious.append(iou)
                sample_dices.append(dice)

    total_pixels = total_tp + total_fp + total_fn + total_tn
    pixel_accuracy = (total_tp + total_tn) / (total_pixels + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    mean_iou = float(np.mean(sample_ious))
    mean_dice = float(np.mean(sample_dices))

    report = {
        'test_samples_count': len(test_pairs),
        'pixel_accuracy': round(float(pixel_accuracy), 4),
        'mean_iou': round(mean_iou, 4),
        'mean_dice': round(mean_dice, 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'confusion_counts': {
            'tp': int(total_tp),
            'fp': int(total_fp),
            'fn': int(total_fn),
            'tn': int(total_tn)
        }
    }

    print('\n===== Test Set Evaluation Report =====')
    print('Test Samples:   ', report['test_samples_count'])
    print('Pixel Accuracy: ', report['pixel_accuracy'])
    print('Mean IoU:       ', report['mean_iou'])
    print('Mean Dice:      ', report['mean_dice'])
    print('Precision:      ', report['precision'])
    print('Recall:         ', report['recall'])
    print('F1 Score:       ', report['f1_score'])
    print('======================================\n')

    os.makedirs('logs', exist_ok=True)
    report_path = os.path.join('logs', 'eval_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report successfully saved to {report_path}')
    return report

if __name__ == '__main__':
    evaluate()
