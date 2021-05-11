import torch

from .lconv import LConvBlock


class DurationPredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.stack = torch.nn.Sequential(
            LConvBlock(dim, 3, 0.1),
            LConvBlock(dim, 3, 0.1),
            LConvBlock(dim, 3, 0.1),
            LConvBlock(dim, 3, 0.1),
        )
        self.projection = torch.nn.Linear(dim, 1)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        ft = self.stack(x)
        x = self.projection(x)
        x = self.softplus(x)
        return x, ft
