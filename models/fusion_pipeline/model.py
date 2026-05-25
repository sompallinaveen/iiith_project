import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2Model,
    DistilBertModel
)


class FusionEmotionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # =========================
        # Speech Branch
        # =========================

        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # Freeze Wav2Vec2
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        self.speech_lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # =========================
        # Text Branch
        # =========================

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # Freeze DistilBERT
        for param in self.bert.parameters():
            param.requires_grad = False

        self.text_lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # =========================
        # Fusion Classifier
        # =========================

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                7
            )
        )

    def forward(
        self,
        audio,
        input_ids,
        attention_mask
    ):

        # =========================
        # Speech Branch
        # =========================

        with torch.no_grad():

            speech_out = self.wav2vec(
                audio
            ).last_hidden_state

        speech_out, _ = self.speech_lstm(
            speech_out
        )

        speech_embedding = torch.mean(
            speech_out,
            dim=1
        )

        # =========================
        # Text Branch
        # =========================

        with torch.no_grad():

            text_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

        text_out, _ = self.text_lstm(
            text_out
        )

        text_embedding = torch.mean(
            text_out,
            dim=1
        )

        # =========================
        # Fusion
        # =========================

        fused = torch.cat(
            [
                speech_embedding,
                text_embedding
            ],
            dim=1
        )

        output = self.classifier(
            fused
        )

        return output