#include <torch/torch.h>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

at::Tensor is_in_cuda(
    at::Tensor elements,
    at::Tensor test_elements,
    bool invert);


#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor is_in(
    at::Tensor elements,
    at::Tensor test_elements,
    bool invert) {
  CHECK_INPUT(elements);
  CHECK_INPUT(test_elements);

  return is_in_cuda(elements, test_elements, invert);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("isin", &is_in, "Numpy-like is in functionality (CUDA)",
  py::arg("elements"), py::arg("test_elements"), py::arg("invert") = false);
}

