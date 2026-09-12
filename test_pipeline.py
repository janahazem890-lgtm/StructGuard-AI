import os
import json
import torch
import joblib
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from models.unet import ResNet18UNet
from feature_extraction import extract_crack_features, create_inspection_overlay
from severity_rules import compute_visual_severity
from chatbot import get_chat_response
from dataset import get_data_splits

def run_integration_test():
    print("=== StructGuard AI - End-to-End Pipeline Integration Test ===")
    os.makedirs('logs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load trained model
    ckpt_path = 'models/best_model.pth'
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    model = ResNet18UNet(pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    print("ResNet18-UNet model loaded successfully.")

    # 2. Load trained RF classifier
    rf_path = 'models/rf_classifier.joblib'
    assert os.path.exists(rf_path), f"RF model not found at {rf_path}"
    rf_model = joblib.load(rf_path)
    print("Random Forest validation model loaded successfully.")

    # 3. Select 3 test samples from held-out test split
    _, _, test_pairs = get_data_splits()
    test_samples = test_pairs[:3]
    print(f"Testing {len(test_samples)} held-out samples...")

    test_results = []

    for i, (img_path, mask_path) in enumerate(test_samples):
        print(f"\n--- Testing Sample {i+1}: {os.path.basename(img_path)} ---")
        img_pil = Image.open(img_path).convert('RGB')
        img_resized = img_pil.resize((256, 256), Image.Resampling.BILINEAR)

        img_t = TF.to_tensor(img_resized)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)

        # Predict mask
        with torch.no_grad():
            logits = model(img_t)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs > 0.5).astype(np.uint8) * 255

        # Feature Extraction
        feats = extract_crack_features(pred_mask)
        print(f"  Length: {feats['length_px']} px, Max Width: {feats['max_width_px']} px, Area: {feats['affected_area_pct']}%")

        # Severity Logic
        sev_info = compute_visual_severity(feats)
        print(f"  Visual Severity: {sev_info['severity']} (Score: {sev_info['score']})")

        # RF Classification Check
        import pandas as pd
        feat_df = pd.DataFrame([{
            'length_px': feats['length_px'],
            'max_width_px': feats['max_width_px'],
            'affected_area_pct': feats['affected_area_pct'],
            'branch_points': feats['branch_points'],
            'component_count': feats['component_count']
        }])
        rf_pred = rf_model.predict(feat_df)[0]
        print(f"  Random Forest Check: {rf_pred} (Agrees: {rf_pred == sev_info['severity']})")

        # Inspection context for Chatbot
        inspection_data = {
            **{k: v for k, v in feats.items() if not isinstance(v, np.ndarray)},
            **{k: v for k, v in sev_info.items() if isinstance(v, (str, float, int, dict))},
            'rf_prediction': rf_pred
        }

        # Chatbot test call
        chat_q = f"Why is this crack rated {sev_info['severity']}?"
        chat_reply = get_chat_response(chat_q, inspection_data)
        print(f"  Chatbot query test succeeded (response length: {len(chat_reply)} chars)")

        # Create visual inspection composite
        orig_np = np.array(img_resized)
        overlay = create_inspection_overlay(orig_np, pred_mask, feats.get('skeleton_mask'), alpha=0.55)
        
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        gt_overlay = create_inspection_overlay(orig_np, gt_resized, alpha=0.55)

        # Combine horizontally: Original | Ground Truth | Prediction Overlay
        composite = np.hstack([orig_np, cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)])
        composite_path = f"logs/integration_test_sample_{i+1}.png"
        cv2.imwrite(composite_path, composite)
        print(f"  Saved verification composite to {composite_path}")

        test_results.append({
            'sample_index': i + 1,
            'file_name': os.path.basename(img_path),
            'features': {k: v for k, v in feats.items() if not isinstance(v, np.ndarray)},
            'severity': sev_info['severity'],
            'score': sev_info['score'],
            'rf_prediction': rf_pred,
            'rf_agrees': bool(rf_pred == sev_info['severity']),
            'chatbot_sample_reply': chat_reply[:200] + '...'
        })

    results_path = 'logs/integration_test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nAll 3 sample tests passed successfully! Detailed results logged to {results_path}")

if __name__ == '__main__':
    run_integration_test()
