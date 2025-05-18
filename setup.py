from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension(
            name='matmul_cuda',
            sources=['opt_runs_pytorch/inefficient_matmul_loop_cuda_implementation_1_cleaned.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
) 