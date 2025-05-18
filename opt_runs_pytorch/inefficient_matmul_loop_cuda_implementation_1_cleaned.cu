// matmul_cuda.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16 // Assuming a square block size
#define TILE_SIZE 16

// CUDA error checking macro
#define CUDA_CHECK(status)                                                     \
    {                                                                          \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)           \
                      << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Error");                           \
        }                                                                      \
    }

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matmul_kernel(float *A, float *B, float *C, int size) {
    // Block row and column index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread row and column index within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column index of the output element Cij
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Shared memory for storing sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Partial result accumulator
    float sum = 0.0f;

    // Loop over the sub-matrices of A and B required to compute Cij
    for (int k = 0; k < (size / BLOCK_SIZE); ++k) {
        // Load sub-matrix of A into shared memory
        As[ty][tx] = A[row * size + k * BLOCK_SIZE + tx];
        // Load sub-matrix of B into shared memory
        Bs[ty][tx] = B[(k * BLOCK_SIZE + ty) * size + col];

        // Synchronize threads within the block to ensure all sub-matrices are loaded
        __syncthreads();

        // Multiply sub-matrices
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        // Synchronize threads before loading the next sub-matrices
        __syncthreads();
    }

    // Write the partial result to global memory
    C[row * size + col] = sum;
}

// CUDA kernel for matrix addition (C = C + D)
__global__ void matrix_add_kernel(float *C, float *D, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        C[row * size + col] += D[row * size + col];
    }
}

// Host function to perform batched matrix multiplication and addition
torch::Tensor matmul_cuda(const torch::Tensor &a_tensor, const torch::Tensor &b_tensor, int iterations) {
    int size = a_tensor.size(0); // Assuming square matrices

    // Check if inputs are on CUDA and contiguous
    if (!a_tensor.is_cuda() || !b_tensor.is_cuda() || !a_tensor.is_contiguous() || !b_tensor.is_contiguous()) {
        throw std::runtime_error("Inputs must be CUDA tensors and contiguous.");
    }

    // Allocate memory on the device for the result matrix C
    torch::Tensor c_tensor = torch::zeros_like(a_tensor);
    float *d_a, *d_b, *d_c, *d_temp;

    d_a = (float *)a_tensor.data_ptr();
    d_b = (float *)b_tensor.data_ptr();
    d_c = (float *)c_tensor.data_ptr();

    // Allocate temporary result matrix on GPU
    torch::Tensor temp_tensor = torch::empty_like(a_tensor);
    d_temp = (float *)temp_tensor.data_ptr();


    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE); // Ceiling division

    // Perform matrix multiplications in a loop
    for (int i = 0; i < iterations; ++i) {
        // Launch the matrix multiplication kernel
        matmul_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_temp, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion before addition.

        // Launch the matrix addition kernel
        matrix_add_kernel<<<gridDim, blockDim>>>(d_c, d_temp, size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completion.

    }

    return c_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "CUDA matrix multiplication with loop");
} 