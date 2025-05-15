import librosa
import numpy as np

def extract_logmel_features(filepath, sr=22050, duration=3.0, n_mels=64):
    try:
        y, _ = librosa.load(filepath, sr=sr, duration=duration)
        max_len = int(sr * duration)
        y = np.pad(y, (0, max(0, max_len - len(y))))[:max_len]
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        logmel = librosa.power_to_db(mel)

        delta = librosa.feature.delta(logmel)
        delta2 = librosa.feature.delta(logmel, order=2)

        combined = np.stack([logmel, delta, delta2], axis=-1)  # shape: (time, freq, 3)
        return combined
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return np.zeros((int(sr * duration / 512), n_mels, 3))  # default frame size