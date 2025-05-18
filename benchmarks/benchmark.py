import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pytorch_optimizer_cli.sample_torch_script import inefficient_matmul_loop
<<<<<<< HEAD
import os
import sys

# Conditionally import CUDA extension if available
cuda_impl_available = False
try:
    import matmul_cuda
    cuda_impl_available = True
    print("CUDA implementation available and loaded successfully")
except ImportError:
    print("CUDA implementation not available, skipping custom CUDA benchmarks")
    print("To enable, run: python setup.py install")
=======
>>>>>>> 835b184 (Added build scripts)

# Define the alternative implementation that uses PyTorch's batched operations
def efficient_matmul_batch(size=256, iterations=50):
    """
    Alternative implementation using PyTorch's batched operations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tensors on the target device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Efficient way: pre-compute the result and add it iterations times
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Compute a*b once
    ab = torch.matmul(a, b)
    
    # Initialize c
    c = torch.zeros(size, size, device=device)
    
    # Add the precomputed product iterations times
    for i in range(iterations):
        c += ab
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Efficient: Finished {iterations} iterations in {end_time - start_time:.4f} seconds.")
    return c.sum(), end_time - start_time

# Define a function that uses PyTorch's batched operations even more efficiently
def very_efficient_matmul(size=256, iterations=50):
    """
    Very efficient implementation using PyTorch's vectorized operations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tensors on the target device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Super efficient way: compute a*b once and then multiply by iterations
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Compute a*b once
    ab = torch.matmul(a, b)
    
    # Multiply by iterations (effectively the same as adding it iterations times)
    c = ab * iterations
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Very Efficient: Finished {iterations} iterations in {end_time - start_time:.4f} seconds.")
    return c.sum(), end_time - start_time

<<<<<<< HEAD
# Custom CUDA implementation
def custom_cuda_matmul(size=256, iterations=50):
    """
    Implementation using custom CUDA kernel
    """
    if not cuda_impl_available:
        print("Custom CUDA implementation not available, skipping")
        return 0, float('inf')
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA device not available, skipping custom CUDA benchmark")
        return 0, float('inf')
    
    print(f"Using device: {device}")
    
    # Create tensors on the target device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Using custom CUDA implementation
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Call our custom CUDA kernel
    c = matmul_cuda.matmul_cuda(a, b, iterations)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Custom CUDA: Finished {iterations} iterations in {end_time - start_time:.4f} seconds.")
    return c.sum(), end_time - start_time

=======
>>>>>>> 835b184 (Added build scripts)
def pytorch_inefficient_matmul_loop(size=256, iterations=50):
    """
    Original implementation with inefficient loop
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create tensors on the target device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.zeros(size, size, device=device)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Inefficient: Performing matmul repeatedly inside a loop
    for i in range(iterations):
        c += torch.matmul(a, b)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Inefficient: Finished {iterations} iterations in {elapsed_time:.4f} seconds.")
    return c.sum(), elapsed_time

def run_size_benchmarks():
    """
    Run benchmarks with different matrix sizes
    """
    iterations = 50
    sizes = [64, 128, 256, 512, 1024]
    
    inefficient_times = []
    efficient_times = []
    very_efficient_times = []
<<<<<<< HEAD
    cuda_times = []
=======
>>>>>>> 835b184 (Added build scripts)
    
    for size in sizes:
        print(f"\nBenchmarking with size {size}x{size}")
        
        # Run inefficient implementation
        _, time_inefficient = pytorch_inefficient_matmul_loop(size, iterations)
        inefficient_times.append(time_inefficient)
        
        # Run efficient implementation
        _, time_efficient = efficient_matmul_batch(size, iterations)
        efficient_times.append(time_efficient)
        
        # Run very efficient implementation
        _, time_very_efficient = very_efficient_matmul(size, iterations)
        very_efficient_times.append(time_very_efficient)
<<<<<<< HEAD
        
        # Run custom CUDA implementation
        _, time_cuda = custom_cuda_matmul(size, iterations)
        cuda_times.append(time_cuda)
=======
>>>>>>> 835b184 (Added build scripts)
    
    # Create a dataframe for plotting
    data = []
    for i, size in enumerate(sizes):
        data.append({
            'Size': f"{size}x{size}",
<<<<<<< HEAD
            'Implementation': 'PyTorch Inefficient Loop',
=======
            'Implementation': 'Inefficient Loop',
>>>>>>> 835b184 (Added build scripts)
            'Time (s)': inefficient_times[i]
        })
        data.append({
            'Size': f"{size}x{size}",
<<<<<<< HEAD
            'Implementation': 'PyTorch Efficient Batch',
=======
            'Implementation': 'Efficient Batch',
>>>>>>> 835b184 (Added build scripts)
            'Time (s)': efficient_times[i]
        })
        data.append({
            'Size': f"{size}x{size}",
<<<<<<< HEAD
            'Implementation': 'PyTorch Very Efficient',
            'Time (s)': very_efficient_times[i]
        })
        if cuda_impl_available:
            data.append({
                'Size': f"{size}x{size}",
                'Implementation': 'TorchFuser CUDA',
                'Time (s)': cuda_times[i]
            })
=======
            'Implementation': 'Very Efficient',
            'Time (s)': very_efficient_times[i]
        })
>>>>>>> 835b184 (Added build scripts)
    
    df = pd.DataFrame(data)
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='Size', y='Time (s)', hue='Implementation', data=df)
    plt.title('Execution Time by Matrix Size (Lower is Better)', fontsize=16)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add values on bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
<<<<<<< HEAD
        if height > 0.001 and height < float('inf'):  # Only add text if the bar is visible and meaningful
=======
        if height > 0.001:  # Only add text if the bar is visible
>>>>>>> 835b184 (Added build scripts)
            ax.annotate(f'{height:.4f}s', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('size_benchmark.png', dpi=300)
    print("Size benchmark saved to 'size_benchmark.png'")

def run_iteration_benchmarks():
    """
    Run benchmarks with different iteration counts
    """
    size = 256
    iterations_list = [10, 25, 50, 100, 200]
    
    inefficient_times = []
    efficient_times = []
    very_efficient_times = []
<<<<<<< HEAD
    cuda_times = []
=======
>>>>>>> 835b184 (Added build scripts)
    
    for iterations in iterations_list:
        print(f"\nBenchmarking with {iterations} iterations")
        
        # Run inefficient implementation
        _, time_inefficient = pytorch_inefficient_matmul_loop(size, iterations)
        inefficient_times.append(time_inefficient)
        
        # Run efficient implementation
        _, time_efficient = efficient_matmul_batch(size, iterations)
        efficient_times.append(time_efficient)
        
        # Run very efficient implementation
        _, time_very_efficient = very_efficient_matmul(size, iterations)
        very_efficient_times.append(time_very_efficient)
<<<<<<< HEAD
        
        # Run custom CUDA implementation
        _, time_cuda = custom_cuda_matmul(size, iterations)
        cuda_times.append(time_cuda)
=======
>>>>>>> 835b184 (Added build scripts)
    
    # Create a dataframe for plotting
    data = []
    for i, iters in enumerate(iterations_list):
        data.append({
            'Iterations': iters,
<<<<<<< HEAD
            'Implementation': 'PyTorch Inefficient Loop',
=======
            'Implementation': 'Inefficient Loop',
>>>>>>> 835b184 (Added build scripts)
            'Time (s)': inefficient_times[i]
        })
        data.append({
            'Iterations': iters,
<<<<<<< HEAD
            'Implementation': 'PyTorch Efficient Batch',
=======
            'Implementation': 'Efficient Batch',
>>>>>>> 835b184 (Added build scripts)
            'Time (s)': efficient_times[i]
        })
        data.append({
            'Iterations': iters,
<<<<<<< HEAD
            'Implementation': 'PyTorch Very Efficient',
            'Time (s)': very_efficient_times[i]
        })
        if cuda_impl_available:
            data.append({
                'Iterations': iters,
                'Implementation': 'TorchFuser CUDA',
                'Time (s)': cuda_times[i] if cuda_times[i] < float('inf') else None
            })
=======
            'Implementation': 'Very Efficient',
            'Time (s)': very_efficient_times[i]
        })
>>>>>>> 835b184 (Added build scripts)
    
    df = pd.DataFrame(data)
    
    # Create line plot
    plt.figure(figsize=(15, 8))
    sns.lineplot(x='Iterations', y='Time (s)', hue='Implementation', 
                 style='Implementation', markers=True, data=df, linewidth=3, markersize=10)
    plt.title('Execution Time by Iteration Count (Lower is Better)', fontsize=16)
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('iteration_benchmark.png', dpi=300)
    print("Iteration benchmark saved to 'iteration_benchmark.png'")

def calculate_speedup():
    """
    Calculate speedup of efficient implementations over inefficient
    """
    iterations = 50
    sizes = [64, 128, 256, 512, 1024]
    
    efficient_speedups = []
    very_efficient_speedups = []
<<<<<<< HEAD
    cuda_speedups = []
=======
>>>>>>> 835b184 (Added build scripts)
    
    for size in sizes:
        print(f"\nCalculating speedup with size {size}x{size}")
        
        # Run all implementations
        _, time_inefficient = pytorch_inefficient_matmul_loop(size, iterations)
        _, time_efficient = efficient_matmul_batch(size, iterations)
        _, time_very_efficient = very_efficient_matmul(size, iterations)
<<<<<<< HEAD
        _, time_cuda = custom_cuda_matmul(size, iterations)
=======
>>>>>>> 835b184 (Added build scripts)
        
        # Calculate speedups
        efficient_speedup = time_inefficient / time_efficient
        very_efficient_speedup = time_inefficient / time_very_efficient
<<<<<<< HEAD
        cuda_speedup = time_inefficient / time_cuda if time_cuda > 0 and time_cuda < float('inf') else 0
        
        efficient_speedups.append(efficient_speedup)
        very_efficient_speedups.append(very_efficient_speedup)
        cuda_speedups.append(cuda_speedup)
        
        print(f"PyTorch Efficient Speedup: {efficient_speedup:.2f}x")
        print(f"PyTorch Very Efficient Speedup: {very_efficient_speedup:.2f}x")
        if cuda_impl_available:
            print(f"TorchFuser CUDA Speedup: {cuda_speedup:.2f}x")
=======
        
        efficient_speedups.append(efficient_speedup)
        very_efficient_speedups.append(very_efficient_speedup)
        
        print(f"Efficient Speedup: {efficient_speedup:.2f}x")
        print(f"Very Efficient Speedup: {very_efficient_speedup:.2f}x")
>>>>>>> 835b184 (Added build scripts)
    
    # Create dataframe for plotting
    data = []
    for i, size in enumerate(sizes):
        data.append({
            'Size': f"{size}x{size}",
<<<<<<< HEAD
            'Implementation': 'PyTorch Efficient Batch',
=======
            'Implementation': 'Efficient Batch',
>>>>>>> 835b184 (Added build scripts)
            'Speedup': efficient_speedups[i]
        })
        data.append({
            'Size': f"{size}x{size}",
<<<<<<< HEAD
            'Implementation': 'PyTorch Very Efficient',
            'Speedup': very_efficient_speedups[i]
        })
        if cuda_impl_available:
            data.append({
                'Size': f"{size}x{size}",
                'Implementation': 'TorchFuser CUDA',
                'Speedup': cuda_speedups[i]
            })
=======
            'Implementation': 'Very Efficient',
            'Speedup': very_efficient_speedups[i]
        })
>>>>>>> 835b184 (Added build scripts)
    
    df = pd.DataFrame(data)
    
    # Create bar plot for speedup
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(x='Size', y='Speedup', hue='Implementation', data=df)
<<<<<<< HEAD
    plt.title('Speedup Over PyTorch Inefficient Implementation', fontsize=16)
=======
    plt.title('Speedup Over Inefficient Implementation', fontsize=16)
>>>>>>> 835b184 (Added build scripts)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Speedup (higher is better)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend(fontsize=12)
    
    # Add values on bars
    for i, p in enumerate(ax.patches):
<<<<<<< HEAD
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}x', 
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.1), 
                        ha='center', fontsize=10)
=======
        ax.annotate(f'{p.get_height():.2f}x', 
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.1), 
                    ha='center', fontsize=10)
>>>>>>> 835b184 (Added build scripts)
    
    plt.tight_layout()
    plt.savefig('speedup.png', dpi=300)
    print("Speedup chart saved to 'speedup.png'")

if __name__ == "__main__":
    print("Starting benchmark...")
    
    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Benchmarks will run on CPU.")
    
<<<<<<< HEAD
    # Ensure CUDA extension is built before running benchmarks
    if cuda_impl_available:
        print("Using custom CUDA implementation for benchmarks")
    else:
        print("Custom CUDA implementation not available.")
        print("Run 'python setup.py install' to build and install the CUDA extension.")
    
=======
>>>>>>> 835b184 (Added build scripts)
    # Run different benchmarks
    run_size_benchmarks()
    run_iteration_benchmarks()
    calculate_speedup()
    
    print("\nBenchmark completed!") 