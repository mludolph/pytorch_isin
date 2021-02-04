# PyTorch isin()

PyTorch extension library bringing the isin() function similar to [numpy.isin()](https://numpy.org/doc/stable/reference/generated/numpy.isin.html) to PyTorch tensors, supporting both CPU and CUDA.

## Installation

Currently, you can only install the extension by compiling it from the source.

### From Source

Make sure that torch (tested with version 1.7.0) is installed. If you want CUDA support, make sure you got the CUDA development kit setup with the same version you installed torch with. Make sure that your `$PATH`, and `CUDA_PATH` or `CPATH` are setup correctly.

```bash
$ python -c "import torch; print(torch.__version__)"
> 1.7.0

$ echo $PATH
> /usr/local/cuda/bin:...

$ echo $CUDA_PATH
> /usr/local/cuda/include:...
```

Then run:

```bash
$ python setup.py install
> ...
```

## Examples

`isin(elements: torch.Tensor, test_elements: torch.Tensor, invert: bool = False)`
Returns a boolean array of the same shape as element that is True where an element of element is in test_elements and False otherwise.

**Note**: This function is equivalent to computing the element-wise python keyword `isin(a, b)` for each element in the input.

```python
import torch
from torch_isin import isin
x = torch.tensor([0, 1, 1, 2])
y = torch.tensor([0, 2])

out = isin(x, y)
```

```python
print(out)
tensor([True, False, False, True])
```

## Running test

```bash
$ python setup.py test
> ...
```
