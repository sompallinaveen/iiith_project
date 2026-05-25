import torch

from wav2vec_model import Wav2Vec2BiLSTM

model = Wav2Vec2BiLSTM()

x = torch.randn(
    2,
    16000
)

out = model(x)

print(out.shape)