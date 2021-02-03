#pragma once

#include <torch/extension.h>

int64_t cuda_version();

torch::Tensor isin(torch::Tensor src, torch::Tensor test_elements, bool invert);
