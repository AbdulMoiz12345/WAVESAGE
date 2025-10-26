"""
Train_gbc.py
------------------------

Train Gradient Boosting Meta-Models for EEG Abnormality Detection
=================================================================

This script trains a meta-model that combines multiple Explainable AI (XAI) methods
(Grad-CAM, SHAP, LIME, Integrated Gradients, SmoothGrad, DDT, RISE, and Occlusion)
to detect abnormal segments within EEG windows.

Each EEG window is divided into smaller time bins (e.g., 0.1 seconds), and for each bin,
a Gradient Boosting classifier is trained using both EEG signal features and aggregated
XAI-derived features.

Workflow:
---------
1. Load and normalize EEG signals from `.npy` files.
2. Load XAI method outputs (e.g., SHAP, Grad-CAM, LIME) from `.csv` files.
3. Aggregate features across all XAI methods for each EEG window.
4. Train Gradient Boosting classifiers (one per time bin).
5. Save the trained models and feature scaler for future inference.

Required Data Layout:
---------------------
data/
    eeg_windows_<subject_id>/
        <filename>.npy
    results_gradcam_<subject_id>.csv
    shap_results_<subject_id>.csv
    lime_results_<subject_id>.csv
    ...

Run Command:
------------
    python train_meta_model_8xai.py

Author: [Your Name]
Date: [Month, Year]
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. EEG and XAI Data Preprocessing
# ============================================================

def preprocess_eeg_signal(file_path):
    """
    Load and normalize a raw EEG signal from a .npy file.
    Returns a zero-mean, unit-variance signal.
    """
    eeg_signal = np.load(file_path)
    eeg_signal = (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)
    return eeg_signal


def preprocess_file(file_path, required_columns):
    """
    Load and validate an XAI result CSV, ensuring it contains the required columns.
    """
    data = pd.read_csv(file_path)
    return data[required_columns]


def bin_labels(start, end, bins):
    """
    Generate binary labels for bins overlapping with a ground-truth abnormal segment.
    """
    labels = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if start < bins[i + 1] and end > bins[i]:
            labels[i] = 1
    return labels


def aggregate_xai_features(xai_features):
    """
    Aggregate XAI features across all methods for each bin.
    Combines mean, max, and variance statistics into a single feature vector.
    """
    return np.hstack([
        np.mean(xai_features, axis=0),
        np.max(xai_features, axis=0),
        np.var(xai_features, axis=0),
    ])


# ============================================================
# 2. Configuration
# ============================================================

TIME_WINDOW = 2.0  # seconds per EEG window
BIN_SIZE = 0.1     # seconds per bin
BINS = np.arange(0, TIME_WINDOW + BIN_SIZE, BIN_SIZE)
NUM_BINS = len(BINS) - 1

# Example placeholders (replace with actual file naming convention)
XAI_FILE_PATHS = {
    'gradcam': 'results_gradcam_subject{}.csv',
    'shap': 'results_shap_subject{}.csv',
    'lime': 'results_lime_subject{}.csv',
    'integrated_gradients': 'results_ig_subject{}.csv',
    'smoothgrad': 'results_smoothgrad_subject{}.csv',
    'ddt': 'results_ddt_subject{}.csv',
    'rise': 'results_rise_subject{}.csv',
    'occlusion': 'results_occlusion_subject{}.csv'
}

# Base EEG folder (replace with actual path)
EEG_BASE_FOLDER = '/path/to/eeg_windows_subject_'  # anonymized

# List of EEG dataset identifiers (anonymized)
SUBJECT_IDS = ['A01', 'A02', 'A03', 'A04', 'A05']  # example anonymized identifiers


# ============================================================
# 3. Feature Extraction Loop
# ============================================================

all_features = []
all_labels = []

for subject_id in SUBJECT_IDS:
    print(f"\n🔍 Processing XAI files for subject: {subject_id}")
    xai_files = {key: path.format(subject_id) for key, path in XAI_FILE_PATHS.items()}
    signal_folder = f"{EEG_BASE_FOLDER}{subject_id}/"

    # Grad-CAM used as the reference for available filenames
    gradcam_path = xai_files['gradcam']
    if not os.path.exists(gradcam_path):
        print(f"⚠️ Grad-CAM file missing: {gradcam_path}")
        continue

    gradcam_data = preprocess_file(gradcam_path, ["Filename", "Actual_Start", "Actual_End", "Predicted_Segments_Formatted"])

    for _, row in gradcam_data.iterrows():
        file_name = row["Filename"]
        signal_path = os.path.join(signal_folder, file_name)

        if not os.path.exists(signal_path):
            print(f"⚠️ Missing EEG file: {signal_path}")
            continue

        # === EEG Signal ===
        eeg_features = preprocess_eeg_signal(signal_path)
        num_eeg_features = eeg_features.shape[0]

        # === Ground Truth Labels ===
        y_label = bin_labels(row["Actual_Start"], row["Actual_End"], BINS)
        all_labels.append(y_label)

        # === Collect XAI Features Across Methods ===
        xai_feature_matrix = []
        for xai_name, xai_path in xai_files.items():
            if not os.path.exists(xai_path):
                continue
            xai_data = preprocess_file(xai_path, ["Filename", "Predicted_Segments_Formatted"])
            row_match = xai_data[xai_data["Filename"] == file_name]
            if row_match.empty:
                continue

            predicted_segments = eval(row_match.iloc[0]["Predicted_Segments_Formatted"])
            bin_counts = [
                sum(start < BINS[i + 1] and end > BINS[i] for start, end in predicted_segments)
                for i in range(NUM_BINS)
            ]
            xai_feature_matrix.append(bin_counts)

        if len(xai_feature_matrix) == 0:
            continue

        # === Aggregate Across XAI Methods ===
        aggregated_features = aggregate_xai_features(np.array(xai_feature_matrix))
        combined_features = np.hstack([eeg_features, aggregated_features])
        all_features.append(combined_features)


# ============================================================
# 4. Model Training
# ============================================================

features = np.array(all_features)
labels = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Scale only the XAI-derived portion of the features
scaler = StandardScaler()
X_train[:, num_eeg_features:] = scaler.fit_transform(X_train[:, num_eeg_features:])
X_test[:, num_eeg_features:] = scaler.transform(X_test[:, num_eeg_features:])

# Train one Gradient Boosting classifier per bin
models = [
    GradientBoostingClassifier(n_estimators=85, learning_rate=0.1, max_depth=3, random_state=42)
    for _ in range(NUM_BINS)
]

print("\n🚀 Training Gradient Boosting models (one per bin)...")
for i, model in enumerate(models):
    model.fit(X_train, y_train[:, i])

print("✅ Training complete.")


# ============================================================
# 5. Save Trained Models and Scaler
# ============================================================

OUTPUT_DIR = "model_output_8xai"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save feature scaler
scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)

# Save one model per bin
for i, model in enumerate(models):
    model_path = os.path.join(OUTPUT_DIR, f"gbm_bin_{i}.joblib")
    joblib.dump(model, model_path)

print(f"✅ All models and scaler saved to: {OUTPUT_DIR}")
