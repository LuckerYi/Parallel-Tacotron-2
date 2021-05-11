import torch

from torch import Tensor

class ParallelTacotron2(torch.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        return x