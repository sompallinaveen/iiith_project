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

from dataset import TESSTextDataset
from model import DistilBERTBiLSTM


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

test_df = pd.read_csv("../../test.csv")

test_dataset = TESSTextDataset(
    test_df
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

model = DistilBERTBiLSTM()

model.load_state_dict(
    torch.load(
        "../../Results/distilbert_bilstm.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

preds = []
targets = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch[0].to(device)

        attention_mask = batch[1].to(device)

        labels = batch[2]

        outputs = model(
            input_ids,
            attention_mask
        )

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

plt.title("DistilBERT + BiLSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "../../Results/text_confusion_matrix.png"
)

plt.close()

print("\nAccuracy:", acc)
print("Weighted F1:", f1)

print("\nConfusion Matrix:")
print(cm)

with open(
    "../../Results/distilbert_results.txt",
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