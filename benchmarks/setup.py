<<<<<<< HEAD
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
=======
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
>>>>>>> 835b184 (Added build scripts)

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension(
            name='matmul_cuda',
<<<<<<< HEAD
            sources=['opt_runs_pytorch/inefficient_matmul_loop_cuda_implementation_1_cleaned.cu']
=======
            sources=['matmul_cuda.cu']
>>>>>>> 835b184 (Added build scripts)
        )
    ],
    cmdclass={'build_ext': BuildExtension}
) 