import sounddevice as sd
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import time

# Load model and label encoder
model = load_model("models/best_cnn_model.keras")
encoder = joblib.load("models/label_encoder.joblib")

# Config
DURATION = 3
SAMPLE_RATE = 22050
N_MELS = 64

def extract_logmel_features_from_array(y, sr=SAMPLE_RATE, n_mels=N_MELS):
    y = y[:sr * DURATION]
    y = np.pad(y, (0, sr * DURATION - len(y)))
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    combined = np.stack([logmel, delta, delta2], axis=-1)
    return combined.reshape(1, *combined.shape)  # for CNN input

def record_audio():
    print("🎙️ Speak now...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return recording.flatten()

responses = {
    "angry": "😠 I hear you're upset. Take a breath.",
    "sad": "😢 I'm here for you. Want to talk about it?",
    "happy": "😊 Great to hear you're happy!",
    "neutral": "😐 How can I assist you today?",
    "fearful": "😨 It's okay to be scared. You're safe.",
    "disgust": "🤢 That sounds unpleasant.",
    "surprised": "😲 That caught you off guard!",
    "calm": "😌 You sound calm. How can I help?"
}

def predict_emotion():
    audio = record_audio()
    features = extract_logmel_features_from_array(audio)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    response = responses.get(predicted_label, "🤖 I'm here to help however I can.")
    print(f"🧠 Detected Emotion: {predicted_label}")
    print("🤖 Assistant:", response)

if __name__ == "__main__":
    while True:
        predict_emotion()
        if input("🔁 Press Enter to try again or type 'q' to quit: ").lower() == "q":
            break