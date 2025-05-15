import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.load_data import load_balanced_filepaths
from src.extract_features import extract_logmel_features

def prepare_dataset():
    print("📁 Loading balanced data paths...")
    filepaths, labels = load_balanced_filepaths()

    print("🔄 Extracting features...")
    X = []
    for fp in filepaths:
        features = extract_logmel_features(fp)
        if features is not None:
            X.append(features)

    X = np.array(X)
    print(f"✅ Feature shape: {X.shape}")  # (samples, time, mel, channels)

    print("🔤 Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    y = np.array(y)

    return X, y, encoder