#pragma once

#include <torch/extension.h>

at::Tensor isin_cuda(
    at::Tensor elements,
    at::Tensor test_elements,
    bool invert);
