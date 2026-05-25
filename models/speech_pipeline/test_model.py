import torch
from model import CNNEmotion

model = CNNEmotion()

x = torch.randn(
    8,
    40,
    200
)

output = model(x)

print(output.shape)