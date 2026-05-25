import torch

from model import DistilBERTBiLSTM

model = DistilBERTBiLSTM()

input_ids = torch.randint(
    0,
    1000,
    (2, 8)
)

attention_mask = torch.ones(
    (2, 8)
)

output = model(
    input_ids,
    attention_mask
)

print(output.shape)