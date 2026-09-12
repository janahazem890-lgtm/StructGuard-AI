# StructGuard AI — Visual Crack Inspection System
## Implementation Plan (imp.md) for Autonomous Agent Execution

---

## 0. IMPORTANT — READ BEFORE STARTING

- This document is the single source of truth for the project. Follow it step by step, in order. Do not skip steps or invent requirements that are not listed here.
- After EVERY major step (data setup, training, evaluation, feature extraction, chatbot, GUI), write a short status report to `logs/progress.md` describing what was done, what worked, what failed, and any assumptions made.
- Do NOT report success or accuracy numbers unless they come from an actual saved log file (e.g. `logs/training_metrics.json`, `logs/eval_report.json`). Never state metrics from memory or estimation.
- If a required credential, dataset, or resource is missing, STOP and write the blocker clearly in `logs/blockers.md` instead of guessing or fabricating a workaround.
- Time is limited. Prefer working end-to-end (even if simple) over a perfect but incomplete component. A finished simple pipeline beats a broken complex one.

---

## 1. Project Goal

Build a **Visual Crack Inspection System**: the user uploads an image of a wall/structural surface, and the system:
1. Detects and segments the crack in the image (pixel-level, not just "crack present/absent").
2. Extracts measurable features from the segmentation: approximate length (px), max width (px), affected area (% of image or region).
3. Produces a **Visual Severity Assessment** (Low / Medium / High) using an explicit, human-readable rule/scoring logic based on the extracted features — NOT a second opaque model that outputs severity directly from the raw image.
4. Displays results in a polished web GUI with an overlay of the detected crack.
5. Provides a chatbot that has the inspection result as context and can explain the classification, generate a report, or advise on what further data would improve the assessment.

## 2. Explicit Non-Goals / Constraints

- **No black-box severity model.** The severity label must be derivable from a documented formula or rule-set applied to the extracted numeric features (length, width, affected area, etc.), so it can always be explained ("why High?" → point to the exact numbers/thresholds).
- Do NOT claim the system predicts structural failure or building safety. Language must stay at "visual assessment of the crack's visible characteristics; further inspection recommended," never "the building is unsafe" or similar.
- Pixel measurements are NOT real-world units (mm/cm) unless a calibration reference is provided in the image. Do not silently convert px → mm.

## 3. Tech Stack (fixed — do not substitute without noting it in blockers.md)

- **Language:** Python 3.10+
- **CV/DL framework:** PyTorch + torchvision
- **Segmentation model:** A lightweight U-Net (or MobileNet-based encoder + U-Net decoder) trained via transfer learning. Prefer a pretrained encoder (e.g., resnet18/mobilenet_v2 pretrained on ImageNet) to reduce training time.
- **Classical CV fallback:** OpenCV (for measurement extraction from the predicted mask: contour length, skeleton-based length, max width via distance transform, area ratio).
- **Classical ML (for classification/validation of severity):** scikit-learn (RandomForestClassifier primarily; SVM optional as a comparison model). Used as a data-driven complement to the rule-based severity logic, NOT as a replacement for it — see Section 5, Steps 7 and 8.
- **Data analysis / visualization:** pandas, matplotlib, seaborn (for EDA and reporting plots).
- **Backend:** Python (FastAPI or Flask) serving inference + feature extraction + severity logic.
- **Frontend:** Streamlit (fastest path to a clean, good-looking UI given time constraints) — dark theme, card-based layout, image side-by-side (original vs. overlay), metrics displayed as styled stat cards, not raw text dumps.
- **Chatbot:** LLM API call (specify provider/key placeholder — e.g., `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` env variable) with a system prompt that injects the current inspection's structured result (JSON) as context before every user question.

## 4. Dataset

### 4.1 Requirement
Need a crack dataset with **pixel-level segmentation masks** (not just classification labels), since length/width/area extraction requires knowing exactly which pixels are the crack.

### 4.2 Candidates to search for on Kaggle (in this priority order)
1. "Crack Segmentation Dataset" (combined dataset merging Crack500, CFD, DeepCrack, etc. — commonly available on Kaggle under this name)
2. "DeepCrack" dataset
3. "CFD" (Crack Forest Dataset)
4. "Crack500"

### 4.3 Prerequisite (must be done by the human before the agent can download from Kaggle)
- The user must place a valid `kaggle.json` API token in `~/.kaggle/kaggle.json` (or provide `KAGGLE_USERNAME` / `KAGGLE_KEY` as env vars) BEFORE the agent attempts any Kaggle download.
- If no valid credentials are found, STOP and write this exact blocker to `logs/blockers.md`, then attempt fallback: search for the same datasets mirrored on GitHub/Zenodo (no login required) and use those instead.

### 4.4 Fallback plan if no masked dataset is found
Use a classification-only crack dataset (crack/no-crack patches) for a basic crack presence detector, and use classical OpenCV techniques (Canny edge detection + morphological filtering + connected components) directly on the full image for measurement extraction instead of a trained segmentation mask. Document this substitution clearly in `logs/progress.md` — do not silently downgrade without noting it.

### 4.5 Dataset size / time constraint
Given limited compute/time: use a maximum of ~1,000–2,000 training images (subsample if the dataset is larger), image size resized to 256x256, batch size appropriate for CPU/limited GPU, and a modest epoch count (start with 10–15 epochs, checkpoint best model by validation IoU, stop early if no improvement for 3 epochs).

## 5. Pipeline Steps (build and verify in this order)

1. **Environment setup**: create `requirements.txt`, virtual environment, verify PyTorch + OpenCV import correctly.
2. **Data acquisition**: download dataset per Section 4, verify image/mask pairs load and align correctly (visualize 3 samples with mask overlay to confirm masks are correct before training).
3. **Preprocessing**: resize, normalize, train/val/test split (e.g. 70/15/15), basic augmentation (flip, rotation, brightness jitter) to help with the small dataset.
4. **Model training**: train the U-Net segmentation model. Save:
   - `models/best_model.pth`
   - `logs/training_metrics.json` (loss/IoU/Dice per epoch)
5. **Evaluation**: compute on held-out test set and save to `logs/eval_report.json`:
   - Pixel accuracy
   - IoU (Intersection over Union)
   - Dice coefficient
   - Precision / Recall / F1
   (Do not report only "accuracy" — segmentation accuracy alone is misleading on imbalanced crack/background pixels.)
6. **Feature extraction module** (`feature_extraction.py`): given a predicted mask, compute:
   - Crack length (skeletonize mask, measure skeleton pixel length)
   - Max width (distance transform along the skeleton, take 2x max distance)
   - Affected area % (crack pixels / total image pixels, or / region of interest if defined)
   - Optional: number of separate crack branches
7. **Data Analysis (EDA) module** (`data_analysis.py` / `notebooks/eda.ipynb`): Run feature extraction (Step 6) across the ENTIRE dataset (not just single images) and analyze the resulting table of {length, max_width, affected_area, branch_count} per image. Produce and save to `logs/eda/`:
   - Summary statistics (mean/median/std/min/max) for each feature, saved as `logs/eda/feature_stats.csv`
   - Histograms/distributions of each feature (`logs/eda/*_distribution.png`)
   - Correlation matrix/heatmap between features (`logs/eda/correlation_heatmap.png`)
   - If proxy/pseudo severity labels exist (see Step 8), a breakdown of feature distributions per severity class (boxplots)
   This EDA output is also what justifies the threshold choices (T1, T2) used in the rule-based logic below — thresholds should be picked with reference to the actual distribution of the data (e.g. percentiles), not arbitrary round numbers.

8. **Severity logic module** (`severity_rules.py`): explicit, documented thresholds, e.g.:
   ```
   score = w1*normalized_length + w2*normalized_width + w3*affected_area
   if score < T1: Low
   elif score < T2: Medium
   else: High
   ```
   Document the chosen thresholds/weights and the reasoning in comments — these should be justifiable, not arbitrary magic numbers. If no ground-truth severity labels exist to calibrate against, state clearly that thresholds are a reasonable engineering default pending expert calibration.
9. **Classical ML model — Random Forest (primary), SVM (optional comparison)** (`ml_classifier.py`):
   - Build a labeled table where each row = one image's {length, max_width, affected_area, branch_count} → severity label. Since no expert-annotated severity ground truth exists, use the **rule-based logic's own output (Step 8) as pseudo-labels** for this table, and state this explicitly in `logs/progress.md` (this is a documented, honest limitation — not hidden).
   - Train a `RandomForestClassifier` (scikit-learn) on this table (train/test split, e.g. 80/20) to predict severity from the same features.
   - Report accuracy, precision/recall/F1, and a confusion matrix to `logs/ml_classifier_report.json`.
   - Report and save **feature importances** (`logs/eda/rf_feature_importance.png`) — this is the interpretability payoff: it shows which measurable properties (length vs. width vs. area) drive severity most, grounding the rule-based weights in data rather than pure guesswork.
   - Optionally train an `SVC` (SVM) on the same data as a secondary comparison model, report its metrics alongside the Random Forest's for contrast (do not need to pick a "winner" — the point is showing both a transparent rule engine AND a data-driven classical ML model agree/complement each other).
   - **Important:** the Random Forest/SVM model is a validation and insight layer, not the final authority — the deployed system still uses the explicit rule-based logic (Step 8) as the actual severity shown to the user, per Section 2's non-goals. The ML classifier's agreement rate with the rules can be reported in the GUI as a "confidence/consistency check" if time allows.
10. **Optional calibration feature**: if a reference object of known size is marked in the image, convert px measurements to real-world units (mm/cm). If not implemented due to time, explicitly state measurements are in pixels only.
11. **Chatbot integration**: build the context-injection logic — every chat session starts with the current inspection's JSON result injected as system/context message, so the bot can answer "why High?" or generate a report using this data, not general knowledge.
12. **GUI (Streamlit)**:
    - Upload image → show original + overlay side-by-side
    - Stat cards: Length / Max Width / Affected Area / Severity (color-coded: green/yellow/red)
    - "Why?" expandable section showing the rule-based reasoning AND the Random Forest's feature importance / agreement note
    - Chat panel below, pre-loaded with inspection context
    - Clean typography, consistent color palette, no default Streamlit look — custom CSS for a polished feel
13. **Final integration test**: run the full flow end-to-end on 3 sample images, confirm outputs are consistent and logged.

## 6. Deliverables Checklist

- [ ] `logs/training_metrics.json`, `logs/eval_report.json` — real numbers, not estimates
- [ ] `models/best_model.pth`
- [ ] `feature_extraction.py`, `severity_rules.py` — documented logic
- [ ] `logs/eda/feature_stats.csv` + distribution/correlation plots — real EDA on the dataset
- [ ] `ml_classifier.py`, `logs/ml_classifier_report.json`, `logs/eda/rf_feature_importance.png` — Random Forest (+ optional SVM) results
- [ ] Streamlit app running locally with working upload → analysis → chat flow
- [ ] `logs/progress.md` — narrative of what was built, what was substituted, and why
- [ ] `logs/blockers.md` — anything that couldn't be completed and needs human input (e.g. missing API keys)

## 7. Known Risks (do not silently work around these — log them)

- No GPU available → training may be slow; reduce dataset/epochs as needed and note the tradeoff.
- Kaggle requires authentication → needs `kaggle.json` from the user first.
- Masked (segmentation) datasets are less common than classification-only ones → fallback plan in 4.4 applies if none found.
- Chatbot requires an LLM API key (OpenAI/Anthropic) — must be provided as an environment variable before the chatbot step can be tested.
- The Random Forest/SVM classifier is trained on pseudo-labels generated by the rule-based logic itself (no independent expert-labeled severity ground truth exists). This means its "accuracy" reflects how well it reproduces the rules, not independent real-world correctness — report this honestly, do not present it as validated against ground truth.
