# Matrix Multiplication Benchmark

<<<<<<< HEAD
This benchmark compares the performance of four different matrix multiplication implementations:

1. **PyTorch Inefficient Loop**: The original implementation using a Python loop to perform repeated matrix multiplications with PyTorch.
2. **PyTorch Efficient Batch**: An improved implementation that computes the matrix product once and adds it in a loop.
3. **PyTorch Very Efficient**: A fully vectorized implementation that eliminates loops completely.
4. **TorchFuser CUDA**: A custom CUDA kernel implementation that demonstrates C++/CUDA integration with PyTorch.
=======
This benchmark compares the performance of three different matrix multiplication implementations:

1. **Inefficient Loop**: The original implementation using a Python loop to perform repeated matrix multiplications.
2. **Efficient Batch**: An improved implementation that computes the matrix product once and adds it in a loop.
3. **Very Efficient**: A fully vectorized implementation that eliminates loops completely.
>>>>>>> 835b184 (Added build scripts)

## Prerequisites

- Python 3.6+
- PyTorch 
<<<<<<< HEAD
- CUDA-capable GPU (for running GPU benchmarks and especially for the custom CUDA kernel)
- matplotlib, seaborn, pandas, numpy (for visualization)
- CUDA toolkit and compatible C++ compiler for building the custom CUDA extension

## Setup

1. Install the required dependencies:
=======
- CUDA-capable GPU (optional, will run on CPU if not available)
- matplotlib, seaborn, pandas, numpy (for visualization)

## Setup

Install the required dependencies:
>>>>>>> 835b184 (Added build scripts)
```bash
pip install -r benchmark_requirements.txt
```

<<<<<<< HEAD
2. Build and install the custom CUDA extension:
```bash
python setup.py install
```
This step is required to enable the TorchFuser CUDA benchmarks. If skipped, the benchmark will run only the PyTorch implementations.

=======
>>>>>>> 835b184 (Added build scripts)
## Running the Benchmark

To run the benchmark:
```bash
python benchmark.py
```

This will:
1. Run benchmarks with different matrix sizes (64x64 to 1024x1024)
2. Run benchmarks with different iteration counts (10 to 200)
<<<<<<< HEAD
3. Calculate speedup of all implementations over the inefficient one
=======
3. Calculate speedup of efficient implementations over the inefficient one
>>>>>>> 835b184 (Added build scripts)

## Output

The benchmark will generate three visualization files:
- `size_benchmark.png`: Performance comparison across different matrix sizes
- `iteration_benchmark.png`: Performance comparison across different iteration counts
<<<<<<< HEAD
- `speedup.png`: Speedup chart showing how much faster the efficient implementations are compared to the inefficient one

## Optimization Approaches

1. **PyTorch Inefficient Loop** (original code):
=======
- `speedup.png`: Speedup chart showing how much faster the efficient implementations are

## Optimization Approaches

1. **Inefficient Loop** (original code):
>>>>>>> 835b184 (Added build scripts)
   ```python
   for i in range(iterations):
       c += torch.matmul(a, b)
   ```

<<<<<<< HEAD
2. **PyTorch Efficient Batch**:
=======
2. **Efficient Batch**:
>>>>>>> 835b184 (Added build scripts)
   ```python
   # Pre-compute the product once
   ab = torch.matmul(a, b)
   # Add the pre-computed product multiple times
   for i in range(iterations):
       c += ab
   ```

<<<<<<< HEAD
3. **PyTorch Very Efficient**:
=======
3. **Very Efficient**:
>>>>>>> 835b184 (Added build scripts)
   ```python
   # Pre-compute the product once
   ab = torch.matmul(a, b)
   # Multiply by the number of iterations
   c = ab * iterations
   ```

<<<<<<< HEAD
4. **TorchFuser CUDA** (custom implementation):
   ```cpp
   // C++ CUDA kernel implementation
   // See matmul_cuda.cu for full implementation
   torch::Tensor matmul_cuda(const torch::Tensor &a_tensor, const torch::Tensor &b_tensor, int iterations) {
       // Custom CUDA kernel that performs the multiplication and accumulation
   }
   ```

## Understanding the Results

The benchmark is designed to show:

1. **Performance scaling with matrix size**: How execution time increases with larger matrices
2. **Performance scaling with iterations**: How execution time increases with more iterations
3. **Speedup comparison**: Direct comparison of how much faster each implementation is

### Expected Findings:

- The inefficient loop will be slowest due to repeated matrix multiplications
- The efficient batch will be faster by avoiding redundant multiplications
- The very efficient implementation will be fastest for PyTorch by eliminating loops
- The custom CUDA implementation may show different performance characteristics, particularly for larger matrices

=======
>>>>>>> 835b184 (Added build scripts)
## Notes

- All implementations use GPU if available, fallback to CPU otherwise
- The benchmark includes proper CUDA synchronization to ensure accurate timing
<<<<<<< HEAD
- The custom CUDA kernel demonstrates how to integrate C++/CUDA with PyTorch for potential performance gains
- For small matrices, the overhead of kernel launches might make differences less significant 
=======
- For small matrices, the overhead of tensor operations might make differences less significant 
>>>>>>> 835b184 (Added build scripts)
