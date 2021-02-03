from itertools import product

import pytest
import torch
from torch import Tensor
from torch_isin import isin

from .utils import dtypes, devices, tensor


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_isin(dtype, device):
    pass


@pytest.mark.parametrize("device", devices)
def test_random_isin(device):
    N = 1024
    for _ in range(5):
        pos = torch.randn((2 * N, 3), device=device)
        batch_1 = torch.zeros(N, dtype=torch.long, device=device)
        batch_2 = torch.ones(N, dtype=torch.long, device=device)
        batch = torch.cat([batch_1, batch_2])
        idx = fps(pos, batch, ratio=0.5)
        assert idx.min() >= 0 and idx.max() < 2 * N
