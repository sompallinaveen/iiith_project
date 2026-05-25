import torch
import torch.nn as nn

from transformers import Wav2Vec2Model


class Wav2Vec2BiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                7
            )
        )

    def forward(self, input_values):

        outputs = self.wav2vec(
            input_values
        )

        features = outputs.last_hidden_state

        lstm_out, _ = self.lstm(
            features
        )

        pooled = torch.mean(
            lstm_out,
            dim=1
        )

        return self.classifier(
            pooled
        )