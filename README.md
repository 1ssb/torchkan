# torchkan

**Under Development**


# TorchKAN: KAN Model Evaluation with PyTorch and CUDA

This project demonstrates the training, validation, and quantization of the simplified KAN model using PyTorch, with CUDA acceleration for improved performance. The project builds the `torchkan` library to create and evaluate KAN models on the MNIST dataset, as a preliminary test.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python (version 3.6 or higher)
- CUDA Toolkit (corresponding to the CUDA version required by your PyTorch installation)
- cuDNN (compatible with your installed CUDA Toolkit)

## Installation (Tested on MacOS and Linux)

### 1. Clone the Repository

Start by cloning the repository containing the `torchkan` library and navigating into the project directory:

```bash
git clone https://github.com/1ssb/torchkan.git
cd torchkan
pip install -r requirements.txt
```

If not already installed:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

To run the MNIST training, make sure you configure wandb and run the script:

```python
python mnist.py
```

## Contact

For questions contact me at Subhransu.Bhattacharjee@anu.edu.au

## Contributions

Contributions are welcome, please raise issues as required after commit Fin. This repository is still under development and testing. The code is licensed under the MIT License.