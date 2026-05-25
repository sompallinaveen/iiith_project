import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW

from wav2vec_dataset import Wav2VecDataset
from wav2vec_model import Wav2Vec2BiLSTM
from collate_fn import collate_fn


def main():

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    # -------------------------
    # Load training data
    # -------------------------

    train_df = pd.read_csv("../../train.csv")

    print(
        f"Training samples: {len(train_df)}"
    )

    # Uncomment for quick testing
    # train_df = train_df.sample(
    #     200,
    #     random_state=42
    # )

    # -------------------------
    # Dataset
    # -------------------------

    train_dataset = Wav2VecDataset(
        train_df
    )

    # -------------------------
    # DataLoader
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # -------------------------
    # Model
    # -------------------------

    model = Wav2Vec2BiLSTM()

    model = model.to(device)

    # -------------------------
    # Loss
    # -------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=1e-5
    )

    # -------------------------
    # Training
    # -------------------------

    epochs = 5

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch_idx, (audio, labels) in enumerate(train_loader):

            audio = audio.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(audio)

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

    # -------------------------
    # Save Model
    # -------------------------

    save_path = (
        "../../Results/"
        "wav2vec_bilstm.pth"
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print(
        f"\nModel saved to:\n"
        f"{save_path}"
    )


if __name__ == "__main__":
    main()