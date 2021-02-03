from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

setup(
    name="torch_isin",
    author="Moritz Ludolph",
    author_email="moritz.ludolph@tu-dortmund.de",
    description="Operation like numpy.isin(element, test_elements) for PyTorch",
    version=__version__,
    install_requires=["torch"],
    setup_requires=["torch"],
    ext_modules=[
        CUDAExtension(
            "torch_isin",
            [
                "cuda/torch_isin.cpp",
                "cuda/torch_isin_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
