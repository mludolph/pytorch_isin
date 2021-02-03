from typing import Optional

import torch


@torch.jit.script
def isin(
    elements: torch.Tensor, test_elements: torch.Tensor, invert: bool = False
) -> torch.Tensor:
    """

    Args:
        elements (torch.Tensor): The elements which should be tested.
        test_elements (torch.Tensor): The elements for which should be tested for.
        invert (bool, optional): Whether to invert the output (i.e. compute 'is not in'). Defaults to False.

    Returns:
        torch.Tensor: A ByteTensor with the same shape as the input tensor and the corresponding 
                      values (True/False) if the element at the position is contained in test_elements.
    """
    elements, test_elements = elements.contiguous(), test_elements.contiguous()

    return torch.ops.torch_isin.isin(elements, test_elements, invert)
