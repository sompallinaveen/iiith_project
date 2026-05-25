import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

from dataset import FusionDataset
from model import FusionEmotionModel
from collate_fn import fusion_collate_fn


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    train_df = pd.read_csv(
        "../../train.csv"
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    train_dataset = FusionDataset(
        train_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,      # increased
        shuffle=True,
        collate_fn=fusion_collate_fn
    )

    model = FusionEmotionModel()

    model = model.to(device)

    # Check trainable params
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {trainable_params:,}\n"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4
    )

    epochs = 10

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for batch_idx, batch in enumerate(train_loader):

            audio = batch[0].to(device)

            input_ids = batch[1].to(device)

            attention_mask = batch[2].to(device)

            labels = batch[3].to(device)

            optimizer.zero_grad()

            outputs = model(
                audio,
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

    torch.save(
        model.state_dict(),
        "../../Results/fusion_pipeline/fusion_model.pth"
    )

    print(
        "\nFusion model saved successfully"
    )


if __name__ == "__main__":
    main()