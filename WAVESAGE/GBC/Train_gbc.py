"""
train_gbc.py
------------
Trains Gradient Boosting Classifiers (GBCs) to detect abnormal EEG activity
within short temporal bins using combined EEG + SHAP features.

Each EEG window (default: 2 seconds) is divided into small bins (default: 0.1s each),
and a separate GBC model is trained per bin to predict whether the bin is abnormal.

Before running:
---------------
1. Update the 'CONFIGURATION' section with your own folder paths.
2. Ensure that EEG and SHAP files have matching names.
3. SHAP files must end with '_shap.npy'.

Author: [Your Name or Lab]
"""

import os
import re
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================================================
# 1. HELPER FUNCTIONS
# ======================================================
def preprocess_eeg_signal(file_path):
    """
    Load and normalize raw EEG signal from a .npy file.
    Normalization: (signal - mean) / std
    """
    eeg_signal = np.load(file_path)
    return (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)


def bin_labels(start, end, bins):
    """
    Create binary labels for bins based on the abnormal interval.
    A bin is labeled '1' if it overlaps with the abnormal region.
    """
    labels = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if start < bins[i + 1] and end > bins[i]:
            labels[i] = 1
    return labels


# ======================================================
# 2. CONFIGURATION
# ======================================================
TIME_WINDOW = 2.0     # seconds per EEG window
BIN_SIZE = 0.1        # duration of each bin in seconds
BINS = np.arange(0, TIME_WINDOW + BIN_SIZE, BIN_SIZE)
NUM_BINS = len(BINS) - 1  # e.g., 20 bins for 2-second window

# 🔒 User should replace these with their local dataset paths
EEG_BASE_FOLDER = "path/to/abnormal_windows_with_labels_"
SHAP_BASE_FOLDER = "path/to/shap_values_"

# Anonymous example folder IDs — users should update these
FILE_IDS = ["set01", "set02", "set03", "set04", "set05"]


# ======================================================
# 3. LOAD EEG + SHAP DATA
# ======================================================
def load_data():
    """Load and combine EEG + SHAP features, along with bin-based labels."""
    all_features, all_labels = [], []

    for file_id in FILE_IDS:
        signal_folder = f"{EEG_BASE_FOLDER}{file_id}/"
        shap_folder = f"{SHAP_BASE_FOLDER}{file_id}/"

        print(f"\n📂 Processing dataset: {file_id}")
        print(f"EEG folder:  {signal_folder}")
        print(f"SHAP folder: {shap_folder}")

        if not os.path.exists(signal_folder) or not os.path.exists(shap_folder):
            print(f"❌ Missing folder for dataset '{file_id}'. Skipping.")
            continue

        eeg_files = [f for f in os.listdir(signal_folder) if f.endswith('.npy')]

        for file_name in eeg_files:
            eeg_path = os.path.join(signal_folder, file_name)
            shap_path = os.path.join(shap_folder, file_name.replace('.npy', '_shap.npy'))

            if not os.path.exists(eeg_path) or not os.path.exists(shap_path):
                print(f"⚠️ Missing EEG or SHAP file for: {file_name}")
                continue

            # Extract abnormal start/end times from filename
            match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.npy$', file_name)
            if not match:
                print(f"⚠️ Could not extract start/end times from: {file_name}")
                continue

            start_time = float(match.group(1))
            end_time = float(match.group(2))

            # Load EEG & SHAP data
            eeg_features = preprocess_eeg_signal(eeg_path)
            shap_features = np.load(shap_path)

            # Combine EEG (400) + SHAP (139) features
            combined_features = np.hstack([eeg_features, shap_features])
            all_features.append(combined_features)

            # Create bin-level labels
            labels = bin_labels(start_time, end_time, BINS)
            all_labels.append(labels)

    features = np.array(all_features)
    labels = np.array(all_labels)

    print(f"\n✅ Total samples: {len(features)}")
    print(f"✅ Feature shape: {features.shape}")
    print(f"✅ Label shape:   {labels.shape} (bins per window)")
    return features, labels


# ======================================================
# 4. TRAINING PIPELINE
# ======================================================
def train_gbc_models(features, labels, output_dir="model_output"):
    """
    Train a separate Gradient Boosting Classifier for each temporal bin.
    Saves all trained models and the fitted scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize GBC models for each bin
    models = [
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        for _ in range(NUM_BINS)
    ]

    print("\n🚀 Training Gradient Boosting models for each bin...")
    for i, model in enumerate(models):
        model.fit(X_train, y_train[:, i])
        print(f"✅ Trained model for bin {i + 1}/{NUM_BINS}")

    # Save models and scaler
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))

    for i, model in enumerate(models):
        model_path = os.path.join(output_dir, f"gradient_boosting_bin_{i}.joblib")
        joblib.dump(model, model_path)

    print("\n🎯 Training complete. Models and scaler saved to:", output_dir)


# ======================================================
# 5. SCRIPT ENTRY POINT
# ======================================================
if __name__ == "__main__":
    features, labels = load_data()
    train_gbc_models(features, labels)
