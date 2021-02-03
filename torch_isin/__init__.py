import importlib
import os.path as osp

import torch

__version__ = "1.0.1"

for library in [
    "_version",
    "_isin",
]:
    torch.ops.load_library(
        importlib.machinery.PathFinder()
        .find_spec(library, [osp.dirname(__file__)])
        .origin
    )

if torch.version.cuda is not None:  # pragma: no cover
    cuda_version = torch.ops.torch_isin.cuda_version()

    if cuda_version == -1:
        major = minor = 0
    elif cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split(".")]

    if t_major != major:
        raise RuntimeError(
            f"Detected that PyTorch and torch_isin were compiled with "
            f"different CUDA versions. PyTorch has CUDA version "
            f"{t_major}.{t_minor} and torch_cluster has CUDA version "
            f"{major}.{minor}. Please reinstall the torch_isin that "
            f"matches your PyTorch install."
        )

from .isin import isin

__all__ = [
    "isin",
    "__version__",
]
