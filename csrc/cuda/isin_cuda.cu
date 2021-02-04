#include "isin_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024

namespace
{
  template <typename scalar_t>
  __global__ void isin_cuda_kernel(
      const scalar_t *__restrict__ elements,
      const scalar_t *__restrict__ test_elements,
      const bool invert,
      const int64_t test_elements_size,
      int *output)
  {
    const int element_index = blockIdx.x;
    const int index = threadIdx.x;
    const int stride = blockDim.x;

    const scalar_t element = elements[element_index];

    output += element_index;

    for (int j = index; j < test_elements_size; j += stride)
    {
      int val = ((element == test_elements[j]) != invert) ? 1 : 0;

      if (!invert)
        atomicMax(output, val);
      else
        atomicMin(output, val);
    }
  }
} // namespace

at::Tensor isin_cuda(
    at::Tensor elements,
    at::Tensor test_elements,
    bool invert)
{
  CHECK_CUDA(elements);
  CHECK_CUDA(test_elements);
  cudaSetDevice(elements.get_device());

  const auto N = elements.numel();

  auto output = at::zeros(elements.sizes(), elements.options().dtype(at::kInt)); // elements.type().toScalarType(at::kInt)); // atomicMax doesn't work for byte
  if (invert)
    output.fill_(1);

  dim3 block(THREADS);
  dim3 grid(N);

  AT_DISPATCH_ALL_TYPES(elements.type(), "isin_cuda", ([&] {
                          isin_cuda_kernel<scalar_t><<<grid, block>>>(
                              elements.data_ptr<scalar_t>(),
                              test_elements.data_ptr<scalar_t>(),
                              invert,
                              test_elements.numel(),
                              output.data_ptr<int>());
                        }));

  return output.toType(at::kByte);
}
