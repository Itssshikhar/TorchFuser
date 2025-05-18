#!/bin/bash

# Run CUDA Benchmarking script
# This script builds and runs the CUDA benchmark comparing
# PyTorch, torch.compile, and the CUDA implementation

# Set default number of runs
RUNS=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --help)
      echo "Usage: ./run_cuda_benchmark.sh [options]"
      echo ""
      echo "Options:"
      echo "  --runs N        Run the benchmark N times (default: 5)"
      echo "  --build-only    Only build the CUDA extension, don't run the benchmark"
      echo "  --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Make sure we're in the TorchFuser directory
cd "$(dirname "$0")"

# Check if PyTorch is installed
if ! python -c "import torch" &>/dev/null; then
    echo "PyTorch not found. Installing required packages..."
    pip install torch numpy
fi

# Check if CUDA is available
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "Warning: CUDA is not available in PyTorch. The benchmark may not work correctly."
    echo "Make sure you have a CUDA-compatible GPU and PyTorch with CUDA support."
fi

# Build the CUDA extension
echo "Building CUDA extension..."
python pytorch_cuda_benchmark.py --build

# Exit if only building
if [ -n "$BUILD_ONLY" ]; then
    echo "Build completed. Skipping benchmark as requested."
    exit 0
fi

# Run the benchmark
echo "Running benchmark with $RUNS runs..."
python pytorch_cuda_benchmark.py --runs $RUNS

echo "Benchmark completed!" 