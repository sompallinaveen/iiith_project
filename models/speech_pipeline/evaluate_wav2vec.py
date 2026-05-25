import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

from wav2vec_dataset import Wav2VecDataset
from wav2vec_model import Wav2Vec2BiLSTM
from collate_fn import collate_fn


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# Load test data
test_df = pd.read_csv(
    "../../test.csv"
)

test_dataset = Wav2VecDataset(
    test_df
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

# Load model
model = Wav2Vec2BiLSTM()

model.load_state_dict(
    torch.load(
        "../../Results/wav2vec_bilstm.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

preds = []
targets = []

with torch.no_grad():

    for audio, labels in test_loader:

        audio = audio.to(device)

        outputs = model(audio)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        preds.extend(
            predictions.cpu().numpy()
        )

        targets.extend(
            labels.numpy()
        )

acc = accuracy_score(
    targets,
    preds
)

f1 = f1_score(
    targets,
    preds,
    average="weighted"
)

cm = confusion_matrix(
    targets,
    preds
)
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Wav2Vec2 + BiLSTM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "../../Results/wav2vec_confusion_matrix.png"
)

plt.close()

print("\nAccuracy:", acc)
print("Weighted F1:", f1)

print("\nConfusion Matrix:")
print(cm)

with open(
    "../../Results/wav2vec_results.txt",
    "w"
) as f:

    f.write(
        f"Accuracy: {acc}\n"
    )

    f.write(
        f"Weighted F1: {f1}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )