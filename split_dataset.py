import pandas as pd
from sklearn.model_selection import train_test_split

# Load metadata
df = pd.read_csv("metadata.csv")

# Stratified split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# Save files
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Train size:", len(train_df))
print("Test size:", len(test_df))

print("\nTrain emotion distribution:")
print(train_df["emotion"].value_counts())

print("\nTest emotion distribution:")
print(test_df["emotion"].value_counts())