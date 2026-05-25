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

from dataset import TESSDataset
from model import CNNEmotion

# Load test data
test_df = pd.read_csv("../../test.csv")

test_dataset = TESSDataset(test_df)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNNEmotion().to(device)

model.load_state_dict(
    torch.load(
        r"C:\Users\Naveen kumar\OneDrive\Documents\project\Results\cnn_baseline.pth",
        map_location=device
    )
)

model.eval()

preds = []
targets = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        outputs = model(x)

        pred = torch.argmax(
            outputs,
            dim=1
        )

        preds.extend(pred.cpu().numpy())
        targets.extend(y.numpy())

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

plt.title("MFCC + CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("../../Results/confusion_matrix.png")

print("Confusion matrix saved.")

plt.close()

print("\nAccuracy:", acc)
print("Weighted F1:", f1)

print("\nConfusion Matrix:")
print(cm)

with open("../../Results/baseline_results.txt", "w") as f:

    f.write("Model: MFCC + CNN\n\n")

    f.write(f"Accuracy: {acc}\n")
    f.write(f"Weighted F1: {f1}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))