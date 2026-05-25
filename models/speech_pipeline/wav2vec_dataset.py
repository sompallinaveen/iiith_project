from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset


class Wav2VecDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        project_root = Path(__file__).resolve().parents[2]

        audio_path = project_root / row["path"]

        audio, sr = librosa.load(
            str(audio_path),
            sr=16000
        )

        audio, _ = librosa.effects.trim(audio)

        audio = torch.tensor(
            audio,
            dtype=torch.float32
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        return audio, label