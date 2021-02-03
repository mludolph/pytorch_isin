#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <vector>

namespace
{
  template <typename scalar_t>
  __global__ void is_in_cuda_kernel(
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

at::Tensor is_in_cuda(
    at::Tensor elements,
    at::Tensor test_elements,
    bool invert)
{

  const auto N = elements.numel();

  auto output = at::zeros(elements.sizes(), elements.type().toScalarType(at::kInt)); // atomicMax doesn't work for byte
  if (invert)
    output.fill_(1);

  int nthreads = 1024;
  dim3 block(nthreads);
  dim3 grid(N);
  //dim3 block(nthreads, nthreads);
  //dim3 grid((N + nthreads - 1) / nthreads);

  AT_DISPATCH_ALL_TYPES(elements.type(), "is_in_cuda", ([&] {
                          is_in_cuda_kernel<scalar_t><<<grid, block>>>(
                              elements.data<scalar_t>(),
                              test_elements.data<scalar_t>(),
                              invert,
                              test_elements.numel(),
                              output.data<int>());
                        }));

  return output.toType(at::kByte);
}