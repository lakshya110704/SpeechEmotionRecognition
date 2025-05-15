import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prepare_data import prepare_dataset
from src.model import create_cnn_model

def train_and_save_model():
    # Load data
    X, y, encoder = prepare_dataset()
    num_classes = len(np.unique(y))
    input_shape = X.shape[1:]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Create model
    model = create_cnn_model(input_shape, num_classes)

    # Prepare saving directory
    os.makedirs("models", exist_ok=True)

    # Checkpoint
    checkpoint = ModelCheckpoint("models/best_cnn_model.keras", monitor="val_accuracy", save_best_only=True)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint]
    )

    # Save encoder
    joblib.dump(encoder, "models/label_encoder.joblib")

    print("✅ Model and label encoder saved!")

if __name__ == "__main__":
    train_and_save_model()