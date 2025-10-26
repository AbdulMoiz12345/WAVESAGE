"""
Test_gbc.py
---------------
Evaluates trained Gradient Boosting Classifiers (GBCs) on new EEG windows
to detect abnormal time intervals and compute performance metrics.

Each trained bin-level model predicts whether its temporal bin (e.g., 0.0–0.1s)
contains an abnormal event. Predictions are aggregated to form continuous
abnormal intervals, and Precision, Coverage, and IoU are computed
against the ground-truth labeled interval.

Before running:
---------------
1. Make sure the trained models and scaler are available in 'model_output/'.
2. Update the 'CONFIGURATION' section with your local test data paths.
3. Ensure EEG and SHAP files have matching names.
4. SHAP files must end with '_shap.npy'.

Author: [Your Name or Lab]
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

# ======================================================
# 1. CONFIGURATION
# ======================================================

TIME_WINDOW = 2.0   # seconds per EEG window
BIN_SIZE = 0.1      # seconds per bin
BINS = np.arange(0, TIME_WINDOW + BIN_SIZE, BIN_SIZE)
NUM_BINS = len(BINS) - 1

# Base directories (replace with your dataset structure)
EEG_BASE_FOLDER = "path/to/abnormal_windows_with_labels_"
SHAP_BASE_FOLDER = "path/to/shap_values_"

# Folder ID for test set (anonymous placeholder)
TEST_ID = "test01"

# Output directory for trained models
MODEL_DIR = "model_output"

# ======================================================
# 2. LOAD TRAINED MODELS AND SCALER
# ======================================================
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"❌ Scaler not found at: {scaler_path}")

scaler = joblib.load(scaler_path)

models = []
for i in range(NUM_BINS):
    model_path = os.path.join(MODEL_DIR, f"gradient_boosting_bin_{i}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model file: {model_path}")
    models.append(joblib.load(model_path))

print(f"✅ Loaded {len(models)} trained models and scaler.")


# ======================================================
# 3. HELPER FUNCTIONS
# ======================================================
def preprocess_eeg_signal(file_path):
    """Load and normalize a raw EEG signal from a .npy file."""
    eeg_signal = np.load(file_path)
    return (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)


def bin_labels(start, end, bins):
    """Generate binary bin labels (1 if bin overlaps with abnormal interval)."""
    labels = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if start < bins[i + 1] and end > bins[i]:
            labels[i] = 1
    return labels


def get_intervals(binary_labels, bins):
    """Convert binary labels into continuous abnormal intervals."""
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


def calculate_metrics(predicted_intervals, actual_start, actual_end):
    """Compute Precision, Coverage, and IoU metrics for predicted intervals."""
    correctly_predicted = sum(
        max(0, min(end, actual_end) - max(start, actual_start))
        for start, end in predicted_intervals
    )
    total_predicted = sum(end - start for start, end in predicted_intervals)
    total_actual = actual_end - actual_start

    precision = correctly_predicted / total_predicted if total_predicted > 0 else 0
    coverage = correctly_predicted / total_actual if total_actual > 0 else 0
    iou = (
        correctly_predicted
        / (total_predicted + total_actual - correctly_predicted)
        if total_predicted + total_actual - correctly_predicted > 0
        else 0
    )

    return precision, coverage, iou


# ======================================================
# 4. LOAD TEST DATA
# ======================================================
eeg_folder = f"{EEG_BASE_FOLDER}{TEST_ID}/"
shap_folder = f"{SHAP_BASE_FOLDER}{TEST_ID}/"

print(f"\n📂 Testing with EEG folder:  {eeg_folder}")
print(f"📂 SHAP folder: {shap_folder}")

if not os.path.exists(eeg_folder) or not os.path.exists(shap_folder):
    raise FileNotFoundError(f"❌ Missing EEG or SHAP folder for test ID {TEST_ID}")

eeg_files = [f for f in os.listdir(eeg_folder) if f.endswith(".npy")]

# ======================================================
# 5. PREDICT AND EVALUATE
# ======================================================
results = []

for file_name in eeg_files:
    eeg_path = os.path.join(eeg_folder, file_name)
    shap_path = os.path.join(shap_folder, file_name.replace(".npy", "_shap.npy"))

    if not os.path.exists(shap_path):
        print(f"⚠️ Skipping {file_name}: missing SHAP file.")
        continue

    # Extract ground truth start/end from filename
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.npy$', file_name)
    if not match:
        print(f"⚠️ Skipping {file_name}: could not extract start/end times.")
        continue

    actual_start, actual_end = float(match.group(1)), float(match.group(2))

    # Load and preprocess EEG + SHAP features
    eeg_features = preprocess_eeg_signal(eeg_path)
    shap_features = np.load(shap_path)
    combined_features = np.hstack([eeg_features, shap_features])

    # Scale and predict
    scaled_features = scaler.transform([combined_features])
    binary_labels = [model.predict(scaled_features)[0] for model in models]

    predicted_intervals = get_intervals(binary_labels, BINS)
    precision, coverage, iou = calculate_metrics(predicted_intervals, actual_start, actual_end)

    results.append({
        "File": file_name,
        "Predicted_Intervals": predicted_intervals,
        "Precision": precision,
        "Coverage": coverage,
        "IoU": iou
    })

# ======================================================
# 6. SUMMARY AND SAVE
# ======================================================
results_df = pd.DataFrame(results)

print("\n🧩 Results for Each EEG Window:")
print(results_df)

print("\n📊 Average Metrics:")
print(results_df[["Precision", "Coverage", "IoU"]].mean())

# Save results
OUTPUT_CSV = f"evaluation_results_{TEST_ID}.csv"
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n💾 Results saved successfully at: {OUTPUT_CSV}")
