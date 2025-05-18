# pytorch_optimizer_cli/sample_torch_script.py
import torch
import time

# Define a simple function to profile
def inefficient_matmul_loop():
    """
    Performs repeated matrix multiplications using a Python loop.
    This is intentionally inefficient for demonstration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    size = 256
    iterations = 50 # Keep iterations relatively low to avoid excessive runtime

    # Create tensors on the target device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.zeros(size, size, device=device)

    # Simulate some work with potential bottlenecks
    start_time = time.time()
    for i in range(iterations):
        # Inefficient: Performing matmul repeatedly inside a loop
        # A better approach might involve batching or different ops
        c += torch.matmul(a, b) 
        # Add a small sleep to simulate potential non-compute work or ensure profiler captures distinct steps
        # time.sleep(0.001) 

        # Example of potential CPU-GPU sync point (though less impactful here)
        # if i % 10 == 0:
        #     print(f"Iteration {i}, Current norm: {torch.linalg.norm(c).item()}") # Syncs CPU/GPU

    end_time = time.time()
    print(f"Finished {iterations} iterations in {end_time - start_time:.4f} seconds.")
    # Return something small to ensure computation isn't optimized away
    return c.sum()

# You could add more functions here to test targeting different ones

if __name__ == "__main__":
    print("Running sample script directly...")
    result = inefficient_matmul_loop()
    print(f"Final result sum: {result.item()}")
    print("Sample script finished.") 