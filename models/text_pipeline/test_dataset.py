import pandas as pd

from dataset import TESSTextDataset

df = pd.read_csv("../../train.csv")

dataset = TESSTextDataset(df)

input_ids, attention_mask, label = dataset[0]

print("Input IDs shape:", input_ids.shape)
print("Attention shape:", attention_mask.shape)
print("Label:", label)