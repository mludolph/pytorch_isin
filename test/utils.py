import torch

dtypes = [torch.float, torch.double, torch.int, torch.long]

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices += [torch.device(f"cuda:{torch.cuda.current_device()}")]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)


def rand_tensor(size, dtype, device, low=-1000, high=1000):
    if dtype == torch.float or dtype == torch.double:
        return ((low - high) * torch.rand(size) + high).type(dtype)
    elif dtype == torch.int or dtype == torch.long:
        return torch.randint(low, high, size, device=device).type(dtype)
