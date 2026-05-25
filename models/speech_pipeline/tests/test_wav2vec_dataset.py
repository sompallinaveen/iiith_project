import pandas as pd

from wav2vec_dataset import Wav2VecDataset

df = pd.read_csv("../../train.csv")

dataset = Wav2VecDataset(df)

audio, label = dataset[0]

print(audio.shape)
print(label)