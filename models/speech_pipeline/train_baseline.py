import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn, optim

from dataset import TESSDataset
from model import CNNEmotion

# Load train data
train_df = pd.read_csv("../../train.csv")

# Dataset
train_dataset = TESSDataset(train_df)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Model
model = CNNEmotion().to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training
epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

# Save model
torch.save(
    model.state_dict(),
    "cnn_baseline.pth"
)

print("Model saved successfully")