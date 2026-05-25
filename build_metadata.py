from pathlib import Path
import pandas as pd

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "ps": 5,
    "sad": 6
}

rows = []

for file in Path("data/TESS_female").rglob("*.wav"):

    parts = file.stem.split("_")

    if len(parts) != 3:
        continue

    speaker_group = "OAF" if parts[0] == "OA" else parts[0]
    word = parts[1]
    emotion = parts[2]

    rows.append({
        "path": str(file),
        "word": word,
        "emotion": emotion,
        "speaker_group": speaker_group,
        "label": emotion_map[emotion]
    })

df = pd.DataFrame(rows)

print(df.head())
print()
print(df["emotion"].value_counts())
print()
print(df["speaker_group"].value_counts())

df.to_csv("metadata.csv", index=False)

print("\nmetadata.csv created successfully")