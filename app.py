import os
import sys
import json
import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torchvision.transforms.functional as TF

from models.unet import ResNet18UNet
from feature_extraction import extract_crack_features, create_inspection_overlay
from severity_rules import compute_visual_severity
from chatbot import get_chat_response

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(
    page_title="StructGuard AI | Visual Crack Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Dark Engineering Dashboard Theme */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header decoration */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Stat Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #7ee787;
        margin-top: 4px;
    }

    /* Severity Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-none {
        background-color: rgba(100, 116, 139, 0.2);
        color: #94a3b8;
        border: 1px solid #64748b;
        padding: 4px 14px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------- DYNAMIC LOGS LOADER ----------------
@st.cache_data
def load_all_metrics():
    """
    Dynamically loads actual logged metrics from JSON files.
    Eliminates all hardcoded values from the UI.
    """
    eval_metrics = {
        'pixel_accuracy': 0.0,
        'mean_iou': 0.0,
        'mean_dice': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'test_samples_count': 0
    }
    ml_metrics = {
        'rf_accuracy': 0.0,
        'rf_macro_f1': 0.0,
        'svm_accuracy': 0.0,
        'feature_importances': {}
    }
    train_metrics = {
        'best_val_iou': 0.0,
        'total_epochs': 0
    }

    eval_file = 'logs/eval_report.json'
    if os.path.exists(eval_file):
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_metrics.update(json.load(f))

    ml_file = 'logs/ml_classifier_report.json'
    if os.path.exists(ml_file):
        with open(ml_file, 'r', encoding='utf-8') as f:
            ml_data = json.load(f)
            rf_part = ml_data.get('random_forest', {})
            svm_part = ml_data.get('svm_comparison', {})
            ml_metrics['rf_accuracy'] = float(rf_part.get('test_accuracy', 0.0))
            ml_metrics['rf_macro_f1'] = float(rf_part.get('macro_f1', 0.0))
            ml_metrics['svm_accuracy'] = float(svm_part.get('test_accuracy', 0.0))
            ml_metrics['feature_importances'] = rf_part.get('feature_importances', {})

    train_file = 'logs/training_metrics.json'
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            train_metrics['best_val_iou'] = float(train_data.get('best_val_iou', 0.0))
            train_metrics['total_epochs'] = int(train_data.get('total_epochs', 0))

    return eval_metrics, ml_metrics, train_metrics

eval_m, ml_m, train_m = load_all_metrics()

# ---------------- MODEL CACHING ----------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18UNet(pretrained=False).to(device)
    
    ckpt_path = 'models/best_model.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.eval()

    rf_model = None
    rf_path = 'models/rf_classifier.joblib'
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)

    return model, rf_model, device

model, rf_model, device = load_models()

# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.markdown("## ⚙️ Inspection Settings")
    
    # Pre-loaded sample images from test split
    samples = []
    splits_file = 'data/splits.json'
    if os.path.exists(splits_file):
        with open(splits_file, 'r', encoding='utf-8') as f:
            data_splits = json.load(f)
            samples = [pair[0] for pair in data_splits.get('test', [])[:8]]

    input_mode = st.radio("Select Input Source", ["Preloaded Test Samples", "Upload Image"])
    
    selected_img_path = None
    uploaded_file = None

    if input_mode == "Preloaded Test Samples" and samples:
        sample_names = [os.path.basename(p) for p in samples]
        choice = st.selectbox("Choose a sample surface", sample_names)
        selected_img_path = samples[sample_names.index(choice)]
    else:
        uploaded_file = st.file_uploader("Upload wall/surface photo", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown("### 📏 Physical Calibration")
    use_calib = st.checkbox("Enable Physical Scale (mm/px)")
    px_ratio = None
    if use_calib:
        px_ratio = st.number_input("Scale ratio (mm per pixel)", min_value=0.01, max_value=10.0, value=0.25, step=0.05)
    
    st.markdown("---")
    st.markdown("### 🎛️ Detection Sensitivity")
    threshold = st.slider("Segmentation Threshold", min_value=0.2, max_value=0.8, value=0.5, step=0.05)
    overlay_alpha = st.slider("Crack Overlay Opacity", min_value=0.1, max_value=0.9, value=0.55, step=0.05)

    st.markdown("---")
    st.markdown("### 🤖 Chat Assistant Model")
    llm_provider = st.selectbox(
        "Select AI Provider",
        [
            "Local Engineering AI (No API Key required - Free & Offline)",
            "Google Gemini (Free API)",
            "Groq (Fast Llama 3 - Free API)",
            "OpenAI (Optional)"
        ]
    )

    api_key_input = None
    if "gemini" in llm_provider.lower():
        api_key_input = st.text_input("Gemini API Key (Google AI Studio)", type="password", placeholder="AIzaSy...", help="Get a free key at aistudio.google.com")
    elif "groq" in llm_provider.lower():
        api_key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="Get a free key at console.groq.com")
    elif "openai" in llm_provider.lower():
        api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

# ---------------- INFERENCE PIPELINE ----------------
def run_inspection(image_pil, threshold_val, calib_ratio):
    img_resized = image_pil.resize((256, 256), Image.Resampling.BILINEAR)
    img_t = TF.to_tensor(img_resized)
    img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred_mask = (probs > threshold_val).astype(np.uint8) * 255

    features = extract_crack_features(pred_mask, px_to_mm_ratio=calib_ratio)
    severity_info = compute_visual_severity(features)

    rf_pred = severity_info['severity']
    if rf_model is not None and features['has_crack']:
        feat_vector = pd.DataFrame([{
            'length_px': features['length_px'],
            'max_width_px': features['max_width_px'],
            'affected_area_pct': features['affected_area_pct'],
            'branch_points': features['branch_points'],
            'component_count': features['component_count']
        }])
        rf_pred = rf_model.predict(feat_vector)[0]

    orig_np = np.array(img_resized)
    overlay_img = create_inspection_overlay(
        orig_np, pred_mask, features.get('skeleton_mask'), alpha=overlay_alpha
    )

    combined_result = {
        **features,
        **severity_info,
        'rf_prediction': rf_pred,
        'rf_agrees': (rf_pred == severity_info['severity']),
        'overlay_np': overlay_img,
        'mask_np': pred_mask,
        'orig_np': orig_np
    }
    return combined_result

# ---------------- LOAD ACTIVE IMAGE ----------------
active_img = None
img_label = "Sample"

if uploaded_file is not None:
    active_img = Image.open(uploaded_file).convert('RGB')
    img_label = uploaded_file.name
elif selected_img_path and os.path.exists(selected_img_path):
    active_img = Image.open(selected_img_path).convert('RGB')
    img_label = os.path.basename(selected_img_path)

# ---------------- MAIN DASHBOARD DISPLAY ----------------
st.markdown('<div class="main-title">StructGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Automated Visual Crack Inspection & Quantitative Risk Assessment | '
    f'<strong>Live Metrics from Logs:</strong> Test IoU: <code>{eval_m["mean_iou"]:.4f}</code> | '
    f'Best Val IoU: <code>{train_m["best_val_iou"]:.4f}</code> | '
    f'RF Consistency: <code>{ml_m["rf_accuracy"]*100:.2f}%</code></div>',
    unsafe_allow_html=True
)

tab_inspect, tab_analytics = st.tabs(["🔍 Active Crack Inspection", "📈 Model Training Curves & Analytics"])

with tab_analytics:
    st.markdown("### 📈 ResNet18-UNet Model Training Convergence")
    st.caption(f"Dynamic metrics extracted from logs across {train_m['total_epochs']} epochs on the DeepCrack dataset.")

    # 3 curves side-by-side
    g_col1, g_col2, g_col3 = st.columns(3)
    loss_img = "logs/eda/loss_curve.png"
    iou_img = "logs/eda/iou_curve.png"
    dice_img = "logs/eda/dice_curve.png"

    with g_col1:
        if os.path.exists(loss_img):
            st.image(loss_img, caption="Combined BCE + Dice Loss Curve", use_container_width=True)
    with g_col2:
        if os.path.exists(iou_img):
            st.image(iou_img, caption="Intersection over Union (IoU) Progression", use_container_width=True)
    with g_col3:
        if os.path.exists(dice_img):
            st.image(dice_img, caption="Dice Similarity Coefficient Progression", use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 📊 Held-Out Test Set Benchmark ({eval_m['test_samples_count']} Unseen Test Samples)")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Test Pixel Accuracy", f"{eval_m['pixel_accuracy']*100:.2f}%", f"+{eval_m['pixel_accuracy']:.4f}")
    with m2:
        st.metric("Test Mean IoU", f"{eval_m['mean_iou']*100:.2f}%", f"Best Val: {train_m['best_val_iou']*100:.2f}%")
    with m3:
        st.metric("Test Mean Dice (F1)", f"{eval_m['mean_dice']*100:.2f}%", f"F1: {eval_m['f1_score']*100:.2f}%")
    with m4:
        st.metric("Crack Sensitivity (Recall)", f"{eval_m['recall']*100:.2f}%", f"Precision: {eval_m['precision']*100:.2f}%")

    st.markdown("---")
    st.markdown("### 🔬 Feature Importance & Correlation Analysis")
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        rf_fi = "logs/eda/rf_feature_importance.png"
        if os.path.exists(rf_fi):
            st.image(rf_fi, caption=f"Random Forest Validation ({ml_m['rf_accuracy']*100:.2f}% Agreement with Rules)", use_container_width=True)
    with eda_col2:
        corr_hm = "logs/eda/correlation_heatmap.png"
        if os.path.exists(corr_hm):
            st.image(corr_hm, caption="Correlation Matrix of Morphological Features", use_container_width=True)

with tab_inspect:
    if active_img is not None:
        result = run_inspection(active_img, threshold, px_ratio)
        st.session_state['current_inspection'] = result

        # 1. Top Inspection Stat Cards
        unit_str = result['unit']
        len_display = f"{result['length_calibrated']:.1f} {unit_str}" if result['calibrated'] else f"{result['length_px']} px"
        wid_display = f"{result['max_width_calibrated']:.2f} {unit_str}" if result['calibrated'] else f"{result['max_width_px']:.1f} px"
        area_display = f"{result['affected_area_pct']:.2f} %"
        sev = result['severity']

        badge_class = f"badge-{sev.lower()}"
        badge_html = f'<span class="{badge_class}">{sev.upper()}</span>'

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Estimated Length</div>
                    <div class="metric-value">{len_display}</div>
                    <div class="metric-sub">{result['component_count']} component(s)</div>
                </div>""",
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Max Opening Width</div>
                    <div class="metric-value">{wid_display}</div>
                    <div class="metric-sub">Mean: {result['mean_width_px']:.1f} px</div>
                </div>""",
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Surface Impact Area</div>
                    <div class="metric-value">{area_display}</div>
                    <div class="metric-sub">{result['branch_points']} branch node(s)</div>
                </div>""",
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Visual Severity Rating</div>
                    <div class="metric-value" style="font-size: 1.5rem; margin-top: 4px;">{badge_html}</div>
                    <div class="metric-sub" style="color: #8b949e;">Score: {result['score']:.2f} / 1.00</div>
                </div>""",
                unsafe_allow_html=True
            )

        # 2. Side-by-Side Visual Inspection
        st.markdown("### 🖼️ Visual Crack Localization")
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(result['orig_np'], caption=f"Original Surface Image ({img_label})", use_container_width=True)
        with col_right:
            st.image(
                result['overlay_np'],
                caption="AI Segmentation Overlay (Red = Crack Body, Yellow = Medial Skeleton Axis)",
                use_container_width=True
            )

        # 3. Transparent Explainability Section ("Why?" Accordion)
        with st.expander("🔍 **Why this rating? (Transparent Engineering Breakdown & ML Cross-Validation)**", expanded=False):
            st.markdown("#### Step-by-Step Scoring Rationale")
            st.text(result['rationale'])
            
            st.markdown("#### Machine Learning Validation Layer")
            agreement_text = "✅ **Consistent**" if result['rf_agrees'] else "⚠️ **Slight Deviation**"
            
            fi_items = ml_m['feature_importances']
            fi_desc = ", ".join([f"{k}: {v*100:.1f}%" for k, v in fi_items.items()]) if fi_items else "Area and width dominate"
            
            st.markdown(
                f"- **Random Forest Prediction:** `{result['rf_prediction']}` ({agreement_text} with explicit rules).\n"
                f"- **Dynamic Validation Accuracy:** `{ml_m['rf_accuracy']*100:.2f}%` overall agreement loaded from `logs/ml_classifier_report.json`.\n"
                f"- **Feature Importance Hierarchy:** {fi_desc}."
            )

            st.markdown("#### Engineering Recommendation")
            st.info(result['recommendation'])

            fi_chart = "logs/eda/rf_feature_importance.png"
            if os.path.exists(fi_chart):
                st.image(fi_chart, caption="Random Forest Feature Importance Analysis across Dataset", width=550)

        # 4. Contextual Chatbot Panel
        st.markdown("---")
        st.markdown("### 💬 StructGuard Engineering Chat Assistant")
        st.caption(f"Active Provider: **{llm_provider}** | Tailored for structural crack diagnostics.")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": f"Hello! I have loaded the inspection results for `{img_label}`. The crack is currently rated **{sev}** (Score: {result['score']:.2f}). How can I assist you with this analysis?"
                }
            ]

        q_col1, q_col2, q_col3 = st.columns(3)
        quick_query = None
        if q_col1.button(f"❓ Why is this rated {sev}?"):
            quick_query = f"Why is this crack rated {sev}?"
        if q_col2.button("📋 Generate Inspection Report"):
            quick_query = "Generate a full visual inspection report for this crack."
        if q_col3.button("🔬 Recommended Tests"):
            quick_query = "What further engineering tests should be performed on-site?"

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask a question about the current inspection...") or quick_query
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing inspection context..."):
                    reply = get_chat_response(
                        user_input,
                        result,
                        chat_history=st.session_state["messages"],
                        provider=llm_provider,
                        api_key=api_key_input
                    )
                    st.markdown(reply)
                    st.session_state["messages"].append({"role": "assistant", "content": reply})

    else:
        st.info("👈 Please select a pre-loaded sample from the sidebar or upload an image to begin inspection.")
