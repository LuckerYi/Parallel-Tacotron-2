import torch

from .lconv import LConvBlock


class ResidualEncoder(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stack1 = torch.nn.Sequential(
            LConvBlock(dim, 17, 0.1),
            LConvBlock(dim, 17, 0.1),
            LConvBlock(dim, 17, 0.1),
        )
        self.stack2 = torch.nn.Sequential(
            LConvBlock(dim, 3, 0.1), LConvBlock(dim, 3, 0.1),
            LConvBlock(dim, 3, 0.1), LConvBlock(dim, 3, 0.1),
            LConvBlock(dim, 3, 0.1), LConvBlock(dim, 3, 0.1),
        )

        self.projection = torch.nn.Linear(dim, dim*2)

    def forward(self, x):
        x = self.stack1(x)
        x = self.stack2(x)
        mean, std = torch.split(self.projection(x), 2, dim=-1)
        return std, mean
