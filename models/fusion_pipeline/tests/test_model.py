import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import torch
from model import FusionEmotionModel

model = FusionEmotionModel()

audio = torch.randn(
    2,
    16000
)

input_ids = torch.randint(
    0,
    1000,
    (2, 8)
)

attention_mask = torch.ones(
    (2, 8)
)

output = model(
    audio,
    input_ids,
    attention_mask
)

print(output.shape)