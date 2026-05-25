import pandas as pd

from dataset import FusionDataset

df = pd.read_csv(
    "../../train.csv"
)

dataset = FusionDataset(df)

audio, ids, mask, label = dataset[0]

print(audio.shape)
print(ids.shape)
print(mask.shape)
print(label)