"""
Test_gbc.py
--------------------------

Evaluate Gradient Boosting Meta-Models for EEG Abnormality Detection
====================================================================

This script uses trained meta-models (from `train_meta_model_8xai.py`) to predict 
abnormal time intervals in EEG windows using both EEG signals and features aggregated 
from eight Explainable AI (XAI) methods:

Grad-CAM, SHAP, LIME, Integrated Gradients, SmoothGrad, DDT, RISE, and Occlusion.

Workflow:
---------
1. Load trained Gradient Boosting models and the saved feature scaler.
2. Load XAI result CSVs containing predicted segments for each method.
3. Combine EEG features and aggregated XAI features to predict bin-wise abnormalities.
4. Reconstruct continuous abnormal time intervals.
5. Compute precision, coverage, and IoU metrics per EEG window.
6. Save the overall results to a `.csv` file.

Expected Directory Structure:
-----------------------------
model_output_8xai/
    scaler.joblib
    gbm_bin_0.joblib
    gbm_bin_1.joblib
    ...
data/
    eeg_windows_<subject_id>/
    results_gradcam_<subject_id>.csv
    results_shap_<subject_id>.csv
    ...
    results_occlusion_<subject_id>.csv

Run Command:
------------
    python predict_meta_model_8xai.py

Author: [Your Name]
Date: [Month, Year]
"""

import os
import numpy as np
import pandas as pd
import joblib


# ============================================================
# 1. Configuration
# ============================================================

TIME_WINDOW = 2.0  # seconds
BIN_SIZE = 0.1     # seconds per bin
BINS = np.arange(0, TIME_WINDOW + BIN_SIZE, BIN_SIZE)
NUM_BINS = len(BINS) - 1

# Model directory (must match training output)
MODEL_DIR = "model_output_8xai"

# Load trained scaler and per-bin models
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
models = [
    joblib.load(os.path.join(MODEL_DIR, f"gbm_bin_{i}.joblib"))
    for i in range(NUM_BINS)
]


# ============================================================
# 2. Helper Functions
# ============================================================

def preprocess_eeg_signal(file_path):
    """Load and normalize raw EEG signal from a .npy file."""
    eeg_signal = np.load(file_path)
    return (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)


def generate_features_from_segments(predicted_segments_xai, bins):
    """Aggregate XAI features across all methods into statistical summaries."""
    xai_features = []
    for predicted_segments in predicted_segments_xai.values():
        bin_counts = [
            sum(start < bins[i + 1] and end > bins[i] for start, end in predicted_segments)
            for i in range(len(bins) - 1)
        ]
        xai_features.append(bin_counts)

    xai_features = np.array(xai_features)
    return np.hstack([
        np.mean(xai_features, axis=0),
        np.max(xai_features, axis=0),
        np.var(xai_features, axis=0),
    ])


def get_intervals(binary_labels, bins):
    """Convert binary predictions into continuous abnormal time intervals."""
    intervals = []
    start = None
    for i, label in enumerate(binary_labels):
        if label == 1 and start is None:
            start = bins[i]
        elif label == 0 and start is not None:
            intervals.append((start, bins[i]))
            start = None
    if start is not None:
        intervals.append((start, bins[-1]))
    return intervals


def predict_abnormal_intervals(predicted_segments_xai, eeg_signal_file):
    """Predict abnormal intervals using EEG + XAI features."""
    raw_eeg_features = preprocess_eeg_signal(eeg_signal_file)
    features_xai = generate_features_from_segments(predicted_segments_xai, BINS)

    # Scale only XAI-derived features
    features_xai_scaled = scaler.transform([features_xai])[0]

    # Concatenate raw EEG and scaled XAI features
    combined_features = np.hstack([raw_eeg_features, features_xai_scaled])

    # Predict binary abnormality labels per bin
    binary_labels = [model.predict([combined_features])[0] for model in models]
    return get_intervals(binary_labels, BINS)


def calculate_metrics(predicted_intervals, actual_start, actual_end):
    """Compute Precision, Coverage, and IoU."""
    correctly_predicted = sum(
        max(0, min(end, actual_end) - max(start, actual_start))
        for start, end in predicted_intervals
    )
    total_predicted = sum(end - start for start, end in predicted_intervals)
    total_actual = actual_end - actual_start

    precision = correctly_predicted / total_predicted if total_predicted > 0 else 0
    coverage = correctly_predicted / total_actual if total_actual > 0 else 0
    iou = correctly_predicted / (
        total_predicted + total_actual - correctly_predicted
    ) if total_predicted + total_actual - correctly_predicted > 0 else 0

    return precision, coverage, iou


# ============================================================
# 3. XAI File Setup
# ============================================================

XAI_FILE_PATHS = {
    'gradcam': 'results_gradcam_subjectA.csv',
    'shap': 'results_shap_subjectA.csv',
    'lime': 'results_lime_subjectA.csv',
    'integrated_gradients': 'results_ig_subjectA.csv',
    'smoothgrad': 'results_smoothgrad_subjectA.csv',
    'ddt': 'results_ddt_subjectA.csv',
    'rise': 'results_rise_subjectA.csv',
    'occlusion': 'results_occlusion_subjectA.csv'
}

# EEG signal folder (replace with your actual folder)
EEG_SIGNAL_FOLDER = "/path/to/eeg_windows_subjectA/"

# Load available XAI result CSVs
xai_dfs = {
    method: pd.read_csv(path)
    for method, path in XAI_FILE_PATHS.items()
    if os.path.exists(path)
}

if "gradcam" not in xai_dfs:
    raise FileNotFoundError("Grad-CAM CSV not found — it is required as reference.")


# ============================================================
# 4. Prediction and Evaluation
# ============================================================

results = []
for _, row in xai_dfs["gradcam"].iterrows():
    file_name = row["Filename"]
    actual_start, actual_end = row["Actual_Start"], row["Actual_End"]

    # Skip files missing in any XAI dataframe
    if not all(file_name in df["Filename"].values for df in xai_dfs.values()):
        print(f"⚠️ Skipping {file_name}: missing in one or more XAI results.")
        continue

    # Load predicted segments from all XAI methods
    predicted_segments_xai = {}
    for method, df in xai_dfs.items():
        if file_name in df["Filename"].values:
            predicted_str = df.loc[df["Filename"] == file_name, "Predicted_Segments_Formatted"].values[0]
            if pd.isna(predicted_str):
                continue
            predicted_segments_xai[method] = eval(predicted_str)

    eeg_signal_file = os.path.join(EEG_SIGNAL_FOLDER, file_name)
    if not os.path.exists(eeg_signal_file):
        print(f"⚠️ Skipping {file_name}: EEG file not found.")
        continue

    predicted_intervals = predict_abnormal_intervals(predicted_segments_xai, eeg_signal_file)
    precision, coverage, iou = calculate_metrics(predicted_intervals, actual_start, actual_end)

    results.append({
        "File": file_name,
        "Predicted_Intervals": predicted_intervals,
        "Precision": precision,
        "Coverage": coverage,
        "IoU": iou,
    })


# ============================================================
# 5. Save Evaluation Results
# ============================================================

results_df = pd.DataFrame(results)
print("\nResults for Each File:")
print(results_df)
print("\nAverage Metrics:")
print(results_df[["Precision", "Coverage", "IoU"]].mean())

SAVE_PATH = "final_xai_combined_results_subjectA.csv"
results_df.to_csv(SAVE_PATH, index=False)
print(f"\n💾 Results saved successfully at: {SAVE_PATH}")
