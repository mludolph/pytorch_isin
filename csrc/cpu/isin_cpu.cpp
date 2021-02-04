#include "isin_cpu.h"

#include <ATen/Parallel.h>

#include "utils.h"

template <typename scalar_t>
void inline isin_cpu_kernel(
    const scalar_t *elements,
    const scalar_t *test_elements,
    const bool invert,
    const int64_t elements_size,
    const int64_t test_elements_size,
    int *output)
{
    at::parallel_for(
        0, elements_size, 1, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; b++)
            {
                auto current_out = output + b;
                auto element = *(elements + b);

                for (int j = 0; j < test_elements_size; j++)
                {
                    int val = ((element == test_elements[j]) != invert) ? 1 : 0;
                    if ((!invert && *current_out < val) || (invert && *current_out > val))
                        *current_out = val;
                }
            }
        });
}

torch::Tensor isin_cpu(torch::Tensor elements, torch::Tensor test_elements, bool invert)
{
    CHECK_CPU(elements);
    CHECK_CPU(test_elements);
    auto output = at::zeros(elements.sizes(), elements.options().dtype(at::kInt));
    if (invert)
        output.fill_(1);

    AT_DISPATCH_ALL_TYPES(elements.scalar_type(), "isin_cpu", [&] {
        isin_cpu_kernel<scalar_t>(
            elements.data_ptr<scalar_t>(),
            test_elements.data_ptr<scalar_t>(),
            invert,
            elements.numel(),
            test_elements.numel(),
            output.data_ptr<int>());
    });

    return output;
}
