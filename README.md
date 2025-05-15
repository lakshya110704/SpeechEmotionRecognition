# 🎙️ Speech Emotion Recognition Assistant

A smart assistant that listens to your voice and responds empathetically based on your emotion — powered by deep learning and audio signal processing.

---

## 🔍 Features

- 🎧 Real-time emotion detection from microphone input
- 🔊 Trained on the RAVDESS dataset (balanced)
- 🧠 CNN model with log-mel spectrogram + delta features
- 🤖 Assistant-style responses to detected emotions
- 💾 Model + label encoder saved for reuse

---

## 📦 Tech Stack

- Python 3
- TensorFlow / Keras
- Librosa
- Sounddevice
- NumPy, Scikit-learn
- RAVDESS dataset (locally loaded)

---

## 🚀 How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train the model**

```bash
PYTHONPATH=. python src/train.py
```

3. **Run the assistant**

```bash
PYTHONPATH=. python src/predict.py
```

---

## 🎯 Emotions Detected

- Neutral 😐
- Calm 😌
- Happy 😊
- Sad 😢
- Angry 😠
- Fearful 😨
- Disgust 🤢
- Surprised 😲

---

## 📁 Folder Structure

```
SpeechEmotionRecognition/
├── data/               # RAVDESS .wav files
├── models/             # Saved model (.keras) and encoder
├── src/
│   ├── load_data.py
│   ├── extract_features.py
│   ├── prepare_data.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
└── README.md
```

---

## 🧠 Future Ideas

- 🎛️ Streamlit web UI
- 🗣️ Text-to-speech voice replies
- 📈 Confusion matrix / per-class metrics
- 🌐 REST API for integration

---

## 💡 Credits

- Dataset: [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
- Developed by [Lakshya Mehta](https://github.com/lakshya110704)