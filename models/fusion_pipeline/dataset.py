import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class FusionDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe

        self.tokenizer = (
            DistilBertTokenizer
            .from_pretrained(
                "distilbert-base-uncased"
            )
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Audio
        audio_path = Path("../../") / row["path"]

        audio, sr = librosa.load(
            str(audio_path),
            sr=16000
        )

        audio = torch.tensor(
            audio,
            dtype=torch.float32
        )

        # Text
        text = str(row["word"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=8,
            return_tensors="pt"
        )

        input_ids = (
            encoding["input_ids"]
            .squeeze(0)
        )

        attention_mask = (
            encoding["attention_mask"]
            .squeeze(0)
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        return (
            audio,
            input_ids,
            attention_mask,
            label
        )