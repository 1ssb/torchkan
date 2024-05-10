# TorchKAN: A Simplified KAN Model (Version: 0)
[![PyPI version](https://badge.fury.io/py/TorchKAN.svg)](https://pypi.org/project/TorchKAN/)

This project demonstrates the training, validation, and quantization of the KAN model using PyTorch with CUDA acceleration. The `torchkan` model evaluates performance on the MNIST dataset.

## Project Status: Under Development

The KAN model has shown promising results amidst various GAMs since the 1980s. This implementation, inspired by various sources, achieves over 97% accuracy with an eval time of 0.6 seconds and the quantised model achieves under 0.55 seconds on the MNIST dataset in 8 epochs on a Ubuntu 22.04 OS with a single Nvidia RTX4090. 

As this model is still under study, further exploration into its full capabilities is ongoing.

---

## Introducing KAL_Net

The `KAL_Net` class represents a GAM network architecture called the **Kolmogorov Arnold Legendre Network (KAL-Net)**. This network leverages the mathematical properties of Legendre polynomials to enhance learning and generalization capabilities over traditional polynomial approximations like splines used in KANs.

### Key Features

- **Polynomial Order:** It utilizes Legendre polynomials up to a specified order for each input normalization. This method captures nonlinear relationships more efficiently compared to using simple linear or lower-order polynomial approximations.
- **Efficient Computations:** By caching Legendre polynomial calculations using `functools.lru_cache`, the network avoids redundant computations, significantly speeding up the forward pass.
- **Activation Function:** The network uses the SiLU (Sigmoid Linear Unit) as the base activation function, which is known for better performance in deeper networks due to its non-monotonic behavior.
- **Layer Normalization:** Each layer's output is stabilized using layer normalization, enhancing the training stability and convergence speed.

### Design and Initialization

1. **Weight Initialization:** Weights are initialized using the Kaiming uniform distribution, optimized for the linear nonlinearity, ensuring a robust start for training.
2. **Dynamic Weight and Normalization Management:** Each layer has its weights for both base transformations and polynomial expansions, managed dynamically to scale with the input features and polynomial order.

## Advantages Over Splines (Needs to be rigorously emperically tested)

- **Flexibility in High-Dimensional Spaces:** Legendre polynomials provide a more systematic approach to capturing interactions in high-dimensional data compared to splines, which often require manual knot placement and suffer from dimensionality issues.
- **Analytical Efficiency:** The caching and recurrence relation used in Legendre polynomial computations avoid the computational overhead associated with spline evaluations, especially in high dimensions.
- **Generalization:** Due to their orthogonal properties, Legendre polynomials often lead to better generalization in machine learning model fitting, avoiding overfitting issues common with higher-degree splines.

## Performance Metrics

- **Accuracy:** `KAL_Net` achieved an impressive **97.8% accuracy on the MNIST dataset**, demonstrating its capability to handle complex patterns in image data.
- **Efficiency:** The average forward pass takes only **500 microseconds**, illustrating the computational efficiency brought by caching Legendre polynomials and optimizing tensor operations in PyTorch.

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

Alternately PyPI install only the model as:

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

## Usage

```python
python mnist.py
```
This script will train the model, validate it, quantise and log performance metrics using wandb.

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

Contributions are welcome. Please raise issues as necessary. All issues, as they come up, will be definitely solved to the best of my abilities, after commit "Fin.", scheduled end-June, 2024. Till then if you send a merge request, describe the problem, the fix and why it works.

## References

- [0] Ziming Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024, arXiv. https://arxiv.org/abs/2404.19756
- [1] https://github.com/KindXiaoming/pykan
- [2] https://github.com/Blealtan/efficient-kan
