import torch
import time
import numpy as np
import subprocess
import os
import sys

# Add the pytorch_optimizer_cli directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch_optimizer_cli'))
from pytorch_optimizer_cli.sample_torch_script import inefficient_matmul_loop

def build_cuda_extension():
    """Build the CUDA extension"""
    print("Building CUDA extension...")
    try:
        # Run the setup.py to install the CUDA extension
        subprocess.check_call([sys.executable, "setup.py", "install", "--user"])
        print("CUDA extension built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build CUDA extension: {e}")
        return False

def run_original_pytorch():
    """Run the original PyTorch implementation"""
    print("\n--- Running Original PyTorch Implementation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    start_time = time.time()
    result = inefficient_matmul_loop()
    end_time = time.time()
    
    print(f"Original PyTorch completed in {end_time - start_time:.4f} seconds")
    return end_time - start_time, result

def run_torch_compile():
    """Run the same function but optimized with torch.compile"""
    print("\n--- Running torch.compile Implementation ---")
    
    # Import the function and compile it
    compiled_function = torch.compile(inefficient_matmul_loop)
    
    # Run the compiled function
    start_time = time.time()
    result = compiled_function()
    end_time = time.time()
    
    print(f"torch.compile completed in {end_time - start_time:.4f} seconds")
    return end_time - start_time, result

def run_cuda_implementation():
    """Run the CUDA implementation using the PyTorch extension"""
    print("\n--- Running CUDA Implementation ---")
    
    try:
        # Import the CUDA module (this will fail if not built)
        import matmul_cuda
        
        # Setup the data similar to the original function
        device = torch.device("cuda")
        size = 256
        iterations = 50
        
        # Create tensors on the target device (same as in the original function)
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Run the CUDA implementation
        start_time = time.time()
        c = matmul_cuda.matmul_cuda(a, b, iterations)
        end_time = time.time()
        
        print(f"CUDA implementation completed in {end_time - start_time:.4f} seconds")
        return end_time - start_time, c.sum()
    
    except ImportError as e:
        print(f"CUDA extension not built or not available: {e}")
        print("Run this script with --build first.")
        return None, None

def benchmark(num_runs=5):
    """Run benchmarks for all implementations multiple times"""
    results = {
        "pytorch_original": [],
        "torch_compile": [],
        "cuda": []
    }
    
    print(f"Running each implementation {num_runs} times...")
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}:")
        
        # Original PyTorch
        time_original, result_original = run_original_pytorch()
        results["pytorch_original"].append(time_original)
        
        # torch.compile version
        time_compile, result_compile = run_torch_compile()
        results["torch_compile"].append(time_compile)
        
        # CUDA implementation
        time_cuda, result_cuda = run_cuda_implementation()
        if time_cuda is not None:
            results["cuda"].append(time_cuda)
        
        # Print intermediate results
        if result_original is not None and result_cuda is not None:
            rel_diff = abs((result_original - result_cuda) / result_original) * 100
            print(f"Result comparison - PyTorch: {result_original.item():.2f}, CUDA: {result_cuda.item():.2f}")
            print(f"Relative difference: {rel_diff:.2f}%")
    
    # Print summary
    print("\n--- Benchmark Results ---")
    if results["pytorch_original"]:
        print(f"Original PyTorch: {np.mean(results['pytorch_original']):.4f}s ± {np.std(results['pytorch_original']):.4f}s")
    
    if results["torch_compile"]:
        print(f"torch.compile:    {np.mean(results['torch_compile']):.4f}s ± {np.std(results['torch_compile']):.4f}s")
    
    if results["cuda"]:
        print(f"CUDA Extension:   {np.mean(results['cuda']):.4f}s ± {np.std(results['cuda']):.4f}s")
    else:
        print("CUDA implementation not available or failed")
    
    # Calculate speedups
    if results["pytorch_original"]:
        baseline = np.mean(results["pytorch_original"])
        
        if results["torch_compile"]:
            compile_speedup = baseline / np.mean(results["torch_compile"])
            print(f"\nSpeedup with torch.compile: {compile_speedup:.2f}x")
        
        if results["cuda"]:
            cuda_speedup = baseline / np.mean(results["cuda"])
            print(f"Speedup with CUDA: {cuda_speedup:.2f}x")
        
        if results["torch_compile"] and results["cuda"]:
            cuda_vs_compile = np.mean(results["torch_compile"]) / np.mean(results["cuda"])
            print(f"CUDA vs torch.compile: {cuda_vs_compile:.2f}x")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark different implementations of inefficient_matmul_loop")
    parser.add_argument("--build", action="store_true", help="Build the CUDA extension before benchmarking")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Build the CUDA extension if requested
    if args.build:
        success = build_cuda_extension()
        if not success:
            print("Failed to build CUDA extension. Exiting.")
            sys.exit(1)
    
    print("Starting benchmarks...")
    benchmark(num_runs=args.runs)
    print("Benchmarking complete!") 