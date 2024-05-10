# TorchKAN: A Simplified KAN Model (Version: 0)
[![PyPI version](https://badge.fury.io/py/TorchKAN.svg)](https://pypi.org/project/TorchKAN/)

This project demonstrates the training, validation, and quantization of the KAN model using PyTorch with CUDA acceleration. The `torchkan` model evaluates performance on the MNIST dataset.

## Project Status: Under Development

The KAN model has shown promising results amidst various GAMs since the 1980s. This implementation, inspired by various sources, achieves over 97% accuracy with an eval time of 0.6 seconds and the quantised model achieves under 0.55 seconds on the MNIST dataset in 8 epochs on a Ubuntu 22.04 OS with a single Nvidia RTX4090. 

As this model is still under study, further exploration into its full capabilities is ongoing.

## Prerequisites

Ensure you have the following installed on your system:

- Python (version 3.9 or higher)
- CUDA Toolkit (corresponding to your PyTorch installation's CUDA version)
- cuDNN (compatible with your installed CUDA Toolkit)

## Installation

Tested on MacOS and Linux.

### 1. Clone the Repository

Clone the `torchkan` repository and set up the project environment:

```bash
git clone https://github.com/1ssb/torchkan.git
cd torchkan
pip install -r requirements.txt
```

Alternately PyPi install as:

```bash
pip install torchKAN
```

### 2. Configure CUDA environment variables if they are not already set:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Configure Weights & Biases (wandb)

To track experiments and model performance with wandb:

1. **Set Up wandb Account:**

- Sign up or log in at [Weights & Biases](https://wandb.ai).
- Find your API key in your account settings.

2. **Initialize wandb in Your Project:**

Before running the training script, initialize wandb:

```python
wandb login
```

When prompted, enter your API key. This will link your script executions to your wandb account.

3. **Make sure to change the Entity name in the `mnist.py` file to your username instead of `1ssb`**

This script will train the model, validate it, and log performance metrics using wandb.

## Contact

For any inquiries or support, contact: Subhransu.Bhattacharjee@anu.edu.au

## Cite this Project

If you use this project in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```bibtex
@misc{torchkan,
  author = {Subhransu S. Bhattacharjee},
  title = {TorchKAN},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/1ssb/torchkan/}}
}
```

## Contributions

Contributions are welcome. Please raise issues as necessary after commit "Fin.", scheduled end-June, 2024. The code is licensed under the MIT License.

## References

- [0] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [1] https://github.com/KindXiaoming/pykan
- [2] https://github.com/Blealtan/efficient-kan
