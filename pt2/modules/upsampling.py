import torch


class Upsampling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(dim, dim, 3)
        self.conv2 = torch.nn.Conv1d(dim, dim, 3)

        self.swish_block1 = ...
        self.swish_block2 = ...

    def forward(self, duration, features):
        ...
