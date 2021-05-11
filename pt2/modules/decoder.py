import torch

from .lconv import LConvBlock


class Decoder(torch.nn.Module):
    def __init__(self, dim, num_blocks, num_mels):
        super().__init__()

        self.blocks = torch.nn.ModuleList([
            LConvBlock(dim, 17, 0.1) for _ in range(num_blocks)
        ])
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(dim, num_mels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        out = []
        for f, projection in zip(self.blocks, self.projections):
            x = f(x)
            out.append(projection(x))
        return out
