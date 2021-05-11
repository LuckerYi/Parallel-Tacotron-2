from pt2.modules.lconv import LConvBlock
import torch


def test_lconv():
    f = LConvBlock(32, 13, 0.1)
    mask = None
    x = torch.zeros((1, 10, 32))
    y = f(x, mask=mask)
    assert y.shape == (1, 10, 32)
