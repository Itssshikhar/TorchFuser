#!/bin/bash

# Color formatting
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TorchFuser Benchmark Suite ===${NC}"
echo

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}CUDA compiler (nvcc) not found.${NC}"
    echo -e "${YELLOW}Some benchmarks may still run on CPU but the CUDA extension will not be available.${NC}"
    echo
    echo -e "If you want to run the CUDA benchmarks, please install CUDA toolkit."
    echo
else
    echo -e "${GREEN}CUDA compiler detected.${NC}"
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "Using CUDA version: ${CYAN}$CUDA_VERSION${NC}"
    echo
fi

# Check if PyTorch is available
if python -c "import torch" &> /dev/null; then
    echo -e "${GREEN}PyTorch detected.${NC}"
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo -e "PyTorch version: ${CYAN}$TORCH_VERSION${NC}"
    
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo -e "CUDA available: ${GREEN}Yes${NC}"
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo -e "GPU: ${CYAN}$GPU_NAME${NC}"
    else
        echo -e "CUDA available: ${RED}No${NC}"
        echo -e "${YELLOW}Benchmarks will run on CPU only.${NC}"
    fi
    echo
else
    echo -e "${RED}PyTorch not found. Please install PyTorch first.${NC}"
    echo -e "Install with: pip install torch"
    exit 1
fi

# Install dependencies
echo -e "${GREEN}Installing benchmark dependencies...${NC}"
pip install -r benchmark_requirements.txt
echo

# Build and install CUDA extension
echo -e "${GREEN}Building and installing CUDA extension...${NC}"
python setup.py install
echo

# Run benchmark
echo -e "${GREEN}Running benchmarks...${NC}"
python benchmark.py
echo

# Show results
echo -e "${GREEN}Benchmark completed.${NC}"
echo -e "The following visualization files were generated:"
echo -e "  ${CYAN}size_benchmark.png${NC} - Performance comparison across different matrix sizes"
echo -e "  ${CYAN}iteration_benchmark.png${NC} - Performance comparison across different iteration counts"
echo -e "  ${CYAN}speedup.png${NC} - Speedup chart showing how much faster the efficient implementations are"
echo
echo -e "To view the results, you can use an image viewer or copy these files to your local machine."
echo

# If running in a headless environment, give instructions
if [ -z "$DISPLAY" ]; then
    echo -e "${YELLOW}You are running in a headless environment.${NC}"
    echo -e "To view the results, copy the PNG files to your local machine using scp or another file transfer method."
    echo
fi

echo -e "${GREEN}Thank you for using TorchFuser Benchmark Suite!${NC}" 