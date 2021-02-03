#include <Python.h>
#include <torch/script.h>

#include "cpu/isin_cpu.h"

#ifdef WITH_CUDA
#include "cuda/isin_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__isin(void) { return NULL; }
#endif

torch::Tensor isin(torch::Tensor elements, torch::Tensor test_elements, bool invert)
{
    if (elements.device().is_cuda())
    {
#ifdef WITH_CUDA
        return isin_cuda(elements, test_elements, invert);
#else
        AT_ERROR("Not compiled with CUDA support");
#endif
    }
    else
    {
        return isin_cpu(elements, test_elements, invert);
    }
}

static auto registry =
    torch::RegisterOperators().op("torch_isin::isin", &isin);