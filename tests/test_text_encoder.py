import torch
from pt2.modules.text_encoder import TextEncoder


def test_text_encoder():
    encoder = TextEncoder(256, 32)
    token = torch.zeros( (1, 10), dtype=torch.int32)
    lengths = torch.zeros( (1,), dtype= torch.int32)
    mask = torch.arange(0, token.shape[1], 1)[None, :] >= lengths[:, None]
    encoded_token = encoder(token, mask)
    assert encoded_token.shape == (1, 32, 10)
