import pdb
import torch
import math
import einops
from torch.nn.modules.conv import Conv1d

class ConvBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 5, 1, 2)
        self.bn = torch.nn.BatchNorm1d(dim)
        self.dropout = torch.nn.Dropout(0.5, inplace=False)
    
    def forward(self, x):
        return self.dropout(torch.relu(self.bn(self.conv(x))))
        

class PositionalEncoding(torch.nn.Module):
    """Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512):
        """We are using the model from Parllel Tacotron 1 paper, see assets/parallel_tacotron1_model.png.
        """

        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embedding_dim)
        self.convblock1 = ConvBlock(embedding_dim)
        self.convblock2 = ConvBlock(embedding_dim)
        self.convblock3 = ConvBlock(embedding_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, 6)

    def forward(self, x, x_masks):
        x = self.embed(x)
        x = einops.rearrange(x, 'n w c -> n c w')
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = einops.rearrange(x, 'n c w -> w n c')
        x = self.positional_encoding(x)
        # easy to confuse mask and src_key_padding_mask
        x = self.transformer_encoder(x, src_key_padding_mask=x_masks)
        x = einops.rearrange(x, 'w n c -> n c w')
        return x

