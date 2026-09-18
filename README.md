# StructGuard AI — Visual Crack Inspection System

An end-to-end AI system that detects, measures, and assesses the visual severity of structural cracks from a single image — combining deep learning segmentation, classical computer vision, explainable rule-based logic, and classical machine learning, with a context-aware chatbot on top.

Built as an applied Computer Vision / Machine Learning project for the **Huawei HCIA-AI (H13-311)** track.

> ⚠️ **Disclaimer:** This is a visual assessment tool based on 2D surface imagery only. It does **not** certify structural safety, predict building failure, or replace an on-site inspection by a licensed structural engineer.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Design Principles](#design-principles)
- [Course Context](#course-context)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Overview

StructGuard AI takes a photograph of a wall or concrete surface and produces a complete visual inspection report:

1. **Locates** the crack at pixel level (semantic segmentation, not just classification)
2. **Measures** it — length, maximum width, affected surface area, number of branches
3. **Rates its severity** (Low / Medium / High) through a transparent, documented formula
4. **Cross-checks** that rating against a classical ML model trained on the same measurements
5. **Explains itself** through a chatbot that has the inspection's data as live context
6. Presents everything in a clean, interactive dashboard

The core design goal was **explainability**: every severity rating a user sees can be traced back to exact numbers and documented thresholds — never an opaque model output with no justification.

---

## How It Works

```
┌─────────┐    ┌───────────────┐    ┌────────────────────┐    ┌──────────────────┐
│  Image  │ →  │  Segmentation │ →  │ Feature Extraction  │ →  │  Severity Rating  │
│  Input  │    │ (U-Net/ResNet)│    │ (OpenCV, classical) │    │  (rule-based)     │
└─────────┘    └───────────────┘    └────────────────────┘    └──────────────────┘
                                                                          │
                                                                          ▼
                                                      ┌──────────────────────────────┐
                                                      │  Random Forest / SVM check    │
                                                      │  (validation layer)           │
                                                      └──────────────────────────────┘
                                                                          │
                                                                          ▼
                                          ┌────────────────────────────────────────────┐
                                          │  Streamlit Dashboard + Context-Aware Chatbot │
                                          └────────────────────────────────────────────┘
```

### 1. Segmentation
A **U-Net** with a **ResNet18 encoder** (pretrained on ImageNet, fine-tuned via transfer learning) takes the input image and outputs a per-pixel probability of "crack." A sigmoid activation converts logits to probabilities, thresholded at 0.5 to produce a binary mask.

### 2. Feature Extraction (classical CV)
From the predicted mask:
- **Skeletonization** (morphological thinning) reduces the crack to a 1-pixel-wide medial line, used to measure **length**
- **Distance transform** along the skeleton gives the **maximum width**
- **Connected components** count separate crack fragments
- A neighbor-counting pass on the skeleton detects **branch points**
- **Affected area %** = crack pixels ÷ total image pixels

### 3. Exploratory Data Analysis (EDA)
Feature extraction is run across the *entire* dataset (not just one image) to produce summary statistics, distributions, and a correlation matrix — this grounds the severity thresholds below in the actual shape of the data (percentiles), not arbitrary numbers.

### 4. Severity Scoring (rule-based, explainable)
```
score = 0.45 × normalized(max_width)
      + 0.30 × normalized(length)
      + 0.25 × normalized(affected_area)

score < 0.40         → Low
0.40 ≤ score < 0.70   → Medium
score ≥ 0.70          → High
```
Weights reflect standard visual inspection priorities (width is the primary risk indicator). Every rating comes with a full breakdown of how each sub-score contributed.

### 5. Classical ML Validation Layer
A **Random Forest** (and a **SVM** for comparison) is trained on the same extracted features to predict severity, using the rule engine's own output as pseudo-labels (no independent expert-labeled ground truth was available). This is **not** the final decision-maker — it's a consistency check, and its feature importances (Area 46.6%, Width 38.0%) independently confirm the reasoning behind the rule-based weights.

### 6. Context-Aware Chatbot
Before any user question is sent to the LLM, the current inspection's full structured result (measurements, severity, rule breakdown, RF prediction) is injected as system context. This means the chatbot answers *from this specific inspection's data*, not from generic knowledge. A local rule-based fallback handles common questions (why this rating, generate a report, what further tests are needed) when no LLM API key is configured.

---

## Architecture

**Segmentation model — ResNet18-UNet:**
- Encoder: ResNet18 (pretrained on ImageNet), stages at H/2, H/4, H/8, H/16, H/32
- Decoder: transposed convolutions with skip connections at every encoder stage, restoring full resolution
- Output: single-channel logit map → sigmoid → binary mask
- Loss: combined **BCE + Dice loss** (handles severe foreground/background pixel imbalance)
- Optimizer: AdamW, Cosine Annealing LR schedule
- Training: 12 epochs, batch size 16, best checkpoint selected by validation IoU

---

## Dataset

**[DeepCrack](https://github.com/yhlleo/DeepCrack)** — a public dataset of real concrete surface photographs with pixel-level crack segmentation masks (chosen because most Kaggle "crack" datasets are classification-only and don't provide the masks needed for measurement).

- Split: 70% train / 15% validation / 15% test
- Images resized to 256×256, with horizontal/vertical flip, rotation, and brightness augmentation on the training set only
- Subsampled to ~300–500 training images to keep CPU training time manageable

---

## Results

### Segmentation (test set, 82 held-out images)

| Metric | Value |
|---|---|
| Pixel Accuracy | 98.82% |
| Mean IoU | 0.6771 |
| Mean Dice | 0.7936 |
| Precision | 76.59% |
| Recall | 93.32% |
| F1 Score | 0.8413 |

*Pixel accuracy alone is misleading here since most pixels are background — IoU/Dice/F1 are the meaningful metrics for this imbalanced segmentation task. Higher recall than precision means the model tends to slightly over-predict crack area (favoring catching real cracks over avoiding false positives).*

### Severity Classification Validation (Random Forest / SVM)

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 98.15% | 98.07% |
| SVM (comparison) | 95.37% | 95.37% |

*These accuracies reflect agreement with the rule engine's own pseudo-labels, not independent ground-truth validation — see [Limitations](#limitations).*

**Feature importance (Random Forest):** Affected Area (46.6%) and Max Width (38.0%) are the dominant drivers, consistent with the manually chosen rule-based weights.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Classical Computer Vision | OpenCV |
| Classical Machine Learning | scikit-learn (RandomForestClassifier, SVC) |
| Data Analysis | pandas, matplotlib, seaborn |
| Frontend | Streamlit |
| Chatbot | OpenAI API, with local rule-based fallback |
| Model persistence | joblib (ML models), PyTorch checkpoints (segmentation) |

---

## Project Structure

```
.
├── data/
│   ├── raw/                       # DeepCrack images + masks
│   └── splits.json                # train/val/test split
├── models/
│   ├── unet.py                    # ResNet18-UNet architecture
│   ├── train.py                   # Training loop (loss, metrics, checkpointing)
│   ├── evaluate.py                # Held-out test set evaluation
│   └── best_model.pth             # Saved checkpoint
├── dataset.py                     # PyTorch Dataset / DataLoader / splitting
├── feature_extraction.py          # Skeletonization, distance transform, measurements
├── data_analysis.py               # EDA across the full dataset
├── severity_rules.py              # Transparent rule-based severity scoring
├── ml_classifier.py                # Random Forest / SVM validation layer
├── chatbot.py                     # Context-injection chatbot logic
├── app.py                         # Streamlit dashboard
├── test_pipeline.py               # End-to-end integration test
├── plot_training_curves.py        # Generates loss/IoU/Dice curves from training logs
├── requirements.txt
└── logs/
    ├── training_metrics.json
    ├── eval_report.json
    ├── ml_classifier_report.json
    └── eda/                        # Distribution plots, correlation heatmap, feature importance
```

---

## Installation

```bash
git clone <repo-url>
cd structguard-ai
pip install -r requirements.txt
```

Requires Python 3.10+. GPU is optional — the project is designed to also train reasonably on CPU with a reduced dataset/epoch count.

Optional: set an OpenAI API key for full conversational chatbot capability (otherwise a local rule-based fallback is used):
```bash
export OPENAI_API_KEY=your_key_here
```

---

## Usage

```bash
# 1. Train the segmentation model
python models/train.py

# 2. Evaluate on the held-out test set
python models/evaluate.py

# 3. Run exploratory data analysis
python data_analysis.py

# 4. Train the Random Forest / SVM validation layer
python ml_classifier.py

# 5. Generate training curve plots
python plot_training_curves.py

# 6. Run the full pipeline integration test
python test_pipeline.py

# 7. Launch the dashboard
streamlit run app.py
```

In the dashboard: upload an image (or pick a preloaded test sample), optionally enable pixel-to-mm calibration if a known-size reference object is visible, and use the chat panel to ask about the result.

---

## Design Principles

- **No black-box severity decisions.** The final severity label always traces back to a documented formula applied to measurable features — never a second opaque model predicting severity directly from raw pixels.
- **Honest metric reporting.** Pixel accuracy alone is not reported as "the" accuracy for segmentation, given class imbalance — IoU, Dice, precision, and recall are all reported together.
- **Documented limitations over hidden assumptions.** Where ground truth was unavailable (e.g., expert-labeled severity), the project uses clearly-stated proxy labels rather than presenting them as validated fact.
- **Calibration is opt-in, not assumed.** Pixel measurements are never silently converted to millimeters without an explicit in-image reference.

---

## Course Context

This project was built to demonstrate applied coverage of the Huawei HCIA-AI (H13-311) syllabus:

| Syllabus Area | Weight | Where in this project |
|---|---|---|
| Deep Learning Overview | 25% | U-Net/ResNet18 segmentation, transfer learning, CNNs |
| Machine Learning Overview | 20% | Random Forest, SVM, train/test methodology |
| AI Development Framework | 20% | PyTorch model definition, training loop |
| AI Overview | 15% | Classification vs. segmentation, system design, reporting |
| Cutting-edge AI Applications | 6% | The applied structural-inspection use case itself |

---

## Limitations

- **Dataset size.** Trained on ~300–500 images (subsampled from DeepCrack) due to CPU/time constraints; a larger dataset and longer training would likely improve IoU.
- **No independent severity ground truth.** No expert structural engineer labeled the images by severity; the Random Forest/SVM validation layer is trained on the rule engine's own output as pseudo-labels, so its accuracy reflects internal consistency, not independent correctness.
- **Pixel-only measurements by default.** Real-world units (mm/cm) require an in-image calibration reference; without one, all measurements remain in pixels.
- **2D visual assessment only.** Cannot detect subsurface issues, rebar condition, crack depth, or structural load context — physical inspection is still required for any actionable safety decision.

## Future Improvements

- Expand training data and epochs given more compute
- Add crack-depth and additional defect types (spalling, corrosion staining) as detection classes
- Collect expert-labeled severity ratings for genuine ground-truth validation of the ML layer
- Add temporal tracking (compare the same crack across repeat inspections to detect propagation)
