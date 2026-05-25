import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

from dataset import TESSTextDataset
from model import DistilBERTBiLSTM


def main():

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    # Load training data
    train_df = pd.read_csv(
        "../../train.csv"
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    # Dataset
    train_dataset = TESSTextDataset(
        train_df
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    # Model
    model = DistilBERTBiLSTM()

    model = model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5
    )

    epochs = 5

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for batch_idx, batch in enumerate(train_loader):

            input_ids = batch[0].to(device)

            attention_mask = batch[1].to(device)

            labels = batch[2].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = (
            running_loss
            / len(train_loader)
        )

        print(
            f"\nEpoch {epoch+1} "
            f"Average Loss: "
            f"{avg_loss:.4f}\n"
        )

    # Save model
    torch.save(
        model.state_dict(),
        "../../Results/text_pipeline/distilbert_bilstm.pth"
    )

    print(
        "\nModel saved successfully"
    )


if __name__ == "__main__":
    main()