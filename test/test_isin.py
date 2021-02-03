from itertools import product

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_isin import isin

from .utils import devices, dtypes, tensor, rand_tensor


@pytest.mark.parametrize("device,invert", product(devices, [True, False]))
def test_isin(device, invert):
    elements_np = np.arange(-5, 5, dtype=np.int32)
    # elements_np = np.repeat(np.repeat(elements_np[None], 5, 0)[None], 5, 0)
    test_elements_np = np.arange(-3, 3, dtype=np.int32)
    isin_np = np.isin(elements_np, test_elements_np, invert=invert).astype(np.uint8)

    elements = tensor(elements_np, dtype=torch.int, device=device)
    test_elements = tensor(test_elements_np, dtype=torch.int, device=device)

    isin_torch = isin(elements, test_elements, invert=invert)
    print(isin_torch)
    print(isin_np)
    assert (isin_torch.cpu().numpy() == isin_np).all()


@pytest.mark.parametrize("dtype,device", product(dtypes, devices))
def test_random_isin(dtype, device):
    pass
