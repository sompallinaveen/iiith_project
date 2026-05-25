import torch
import torch.nn as nn

from transformers import DistilBertModel


class DistilBERTBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                7
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = outputs.last_hidden_state

        lstm_out, _ = self.lstm(
            embeddings
        )

        pooled = torch.mean(
            lstm_out,
            dim=1
        )

        return self.classifier(
            pooled
        )