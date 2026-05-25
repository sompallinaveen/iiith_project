import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import FusionDataset
from model import FusionEmotionModel
from collate_fn import fusion_collate_fn


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load test data
test_df = pd.read_csv(
    "../../test.csv"
)

test_dataset = FusionDataset(
    test_df
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=fusion_collate_fn
)

# Load model
model = FusionEmotionModel()

model.load_state_dict(
    torch.load(
        "../../Results/fusion_pipeline/fusion_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

preds = []
targets = []

with torch.no_grad():

    for batch in test_loader:

        audio = batch[0].to(device)

        input_ids = batch[1].to(device)

        attention_mask = batch[2].to(device)

        labels = batch[3]

        outputs = model(
            audio,
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

# Metrics
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


print("\nAccuracy:", acc)
print("Weighted F1:", f1)

print("\nConfusion Matrix:")
print(cm)

# Save metrics
with open(
    "../../Results/fusion_pipeline/fusion_results.txt",
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

# Save confusion matrix image
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Fusion Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "../../Results/fusion_pipeline/fusion_confusion_matrix.png"
)

plt.close()