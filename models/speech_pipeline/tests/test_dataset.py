import pandas as pd
from dataset import TESSDataset

df = pd.read_csv("../../train.csv")

dataset = TESSDataset(df)

x, y = dataset[0]

print(x.shape)
print(y)