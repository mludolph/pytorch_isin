#pragma once

#include <torch/extension.h>

torch::Tensor isin_cpu(torch::Tensor elements, torch::Tensor test_elements, bool invert);