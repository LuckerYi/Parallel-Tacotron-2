import torch

from .lconv import LConvBlock


class ResidualEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack1 = torch.nn.Sequential(* [
            LConvBlock(1, 256, 0.5, 17) for i in range(3)
        ])
        self.stack2 = torch.nn.Sequential(* [
            LConvBlock(1, 256, 0.5, 3) for i in range(6)
        ])
        self.projection = torch.nn.Linear(256, 512)

    def forward(self, x):
        x = self.stack1(x)
        x = self.stack2(x)
        mean, std = torch.split(self.projection(x), 2, dim=-1)
        noise = torch.randn(1, 512)
        return noise * std + mean
