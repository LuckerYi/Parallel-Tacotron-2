import torch

from .residual_encoder import ResidualEncoder
from .text_encoder import TextEncoder


class Encoder(torch.nn.Module):
    def __init__(self, num_tokens, num_speakers, dim):
        super().__init__()
        self.text_encoder = TextEncoder(num_speakers, dim)
        self.speaker_embed = torch.nn.Embedding(num_speakers, dim)
        self.residual_encoder = ResidualEncoder(dim)

    def forward(self, text, speaker, mel):
        encoded_text = self.text_encoder(text)
        encoded_speaker = self.speaker_embed(speaker)
        residual = self.residual_encoder(mel)

        x = encoded_text + encoded_speaker + residual
