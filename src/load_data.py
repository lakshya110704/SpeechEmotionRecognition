import os
import random
from collections import defaultdict

# Mapping from filename code to emotion label
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_emotion(filename):
    try:
        emotion_code = filename.split("-")[2]
        return emotion_map.get(emotion_code)
    except:
        return None

def load_balanced_filepaths(data_dir="data"):
    paths_by_emotion = defaultdict(list)

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = parse_emotion(file)
                if emotion:
                    full_path = os.path.join(root, file)
                    paths_by_emotion[emotion].append(full_path)

    # Balance all emotions to min count
    min_count = min(len(v) for v in paths_by_emotion.values())
    balanced_paths = []
    balanced_labels = []

    for emotion, files in paths_by_emotion.items():
        sampled = random.sample(files, min_count)
        balanced_paths.extend(sampled)
        balanced_labels.extend([emotion] * min_count)

    return balanced_paths, balanced_labels