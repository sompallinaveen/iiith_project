# Multimodal Emotion Recognition using Speech and Text

## Project Overview

This project implements a multimodal emotion recognition system that predicts human emotions from speech and text modalities.

The system was developed as part of the IIIT-H Speech Analytics internship evaluation project and explores:

- Speech-based emotion recognition
- Text-based emotion recognition
- Multimodal fusion of speech and text
- Comparative analysis of different architectures

The project evaluates how emotional information can be extracted from audio signals and textual content and investigates whether combining both modalities improves performance.

---

## Dataset

### TESS Dataset (Toronto Emotional Speech Set)

The dataset contains recordings from:

- OAF (Older Adult Female)
- YAF (Younger Adult Female)

Emotions:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Pleasant Surprise
7. Sad

Each audio file follows the format:

```

OAF\_back\_angry.wav
YAF\_chair\_happy.wav

```

Metadata extracted:

| Column | Description |
|----------|----------|
| path | Audio file path |
| word | Spoken word |
| emotion | Emotion category |
| speaker_group | OAF / YAF |
| label | Numerical class label |

---

## Project Structure

```text
project/
│
├── data/
│   ├── TESS_female/
│   ├── metadata.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   │
│   ├── speech_pipeline/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train_baseline.py
│   │   ├── train_wav2vec.py
│   │   ├── evaluate.py
│   │   ├── evaluate_wav2vec.py
│   │   └── tests/
│   │
│   ├── text_pipeline/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── tests/
│   │
│   └── fusion_pipeline/
│       ├── dataset.py
│       ├── collate_fn.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── tests/
│
├── Results/
│
├── requirements.txt
└── README.md
```

---

## Architectures

### 1. Speech Baseline

```text
Audio
↓
MFCC Extraction
↓
CNN
↓
Softmax
↓
Emotion Label
```

### 2. Speech Final Model

```text
Audio
↓
Preprocessing
↓
Wav2Vec2
↓
BiLSTM
↓
Dense
↓
Softmax
↓
Emotion Label
```

### 3. Text Model

```text
Word
↓
DistilBERT
↓
BiLSTM
↓
Dense
↓
Softmax
↓
Emotion Label
```

### 4. Multimodal Fusion Model

```text
Speech
↓
Wav2Vec2
↓
BiLSTM
↓
Speech Embedding
        +
Text
↓
DistilBERT
↓
BiLSTM
↓
Text Embedding
        ↓
Concatenation
        ↓
Dense
        ↓
Softmax
        ↓
Emotion Label
```

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate:

### Windows PowerShell

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Speech Baseline

```bash
python models/speech_pipeline/train_baseline.py
```

### Wav2Vec2 Speech Model

```bash
python models/speech_pipeline/train_wav2vec.py
```

### Text Model

```bash
python models/text_pipeline/train.py
```

### Fusion Model

```bash
python models/fusion_pipeline/train.py
```

---

## Evaluation

### Speech Baseline

```bash
python models/speech_pipeline/evaluate.py
```

### Speech Wav2Vec2

```bash
python models/speech_pipeline/evaluate_wav2vec.py
```

### Text Pipeline

```bash
python models/text_pipeline/evaluate.py
```

### Fusion Pipeline

```bash
python models/fusion_pipeline/evaluate.py
```

---

## Results

| Model | Modality | Accuracy | Weighted F1 |
|---------|---------|---------|---------|
| MFCC + CNN | Speech | 99.82% | 99.82% |
| DistilBERT + BiLSTM | Text | 13.04% | 3.87% |
| Fusion (Frozen Encoders) | Speech + Text | 89.46% | 89.16% |

---

## Key Findings

- Speech contains the strongest emotional information in the TESS dataset.
- Text-only emotion recognition performs poorly because the dataset contains isolated words rather than meaningful sentences.
- Freezing pretrained Wav2Vec2 and DistilBERT encoders significantly stabilizes multimodal training.
- Multimodal fusion improves robustness but does not outperform the best speech-only model on this dataset.

---

## Author

Naveen Kumar

IIIT-H Speech Analytics Internship Evaluation Project