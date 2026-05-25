from pathlib import Path
import torch
from torch.utils.data import Dataset
import librosa

class TESSDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        project_root = Path(__file__).resolve().parents[2]

        audio_path = project_root / row["path"]
        print(audio_path)

        audio, sr = librosa.load(
            str(audio_path),
            sr=16000
        )

        audio, _ = librosa.effects.trim(audio)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc = librosa.util.fix_length(
            mfcc,
            size=200,
            axis=1
        )

        mfcc = torch.tensor(
            mfcc,
            dtype=torch.float32
        )

        label = torch.tensor(
            row["label"],
            dtype=torch.long
        )

        return mfcc, label