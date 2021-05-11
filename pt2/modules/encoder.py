import torch 
from .text_encoder import TextEncoder
from .residual_encoder import ResidualEncoder


class Encoder(torch.nn.Module):
    def __init__(self, num_tokens, num_speakers, dim):
        super().__init__()
        self.text_encoder = TextEncoder(num_speakers, dim)
        self.speaker_embed = torch.nn.Embedding(num_speakers, dim)
        self.residual_encoder = ResidualEncoder