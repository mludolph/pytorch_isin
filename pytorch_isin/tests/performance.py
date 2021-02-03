import torch
from torch_isin import isin

arr = torch.zeros((5,5)).int().cuda()
test_elements = torch.tensor([5,3,2,0]).int().cuda()

print(f"Testing {arr} for elements {test_elements}...")
x = isin(arr, test_elements, False)
print(x)
#arr[(arr[..., None] == [some_values]).any(-1)] = 0
