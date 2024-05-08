# Quantized Adaptive Networks for Efficient Deep Learning

This repository contains an implementation of Quantized Adaptive Networks (QAN) for efficient deep learning tasks. QAN introduces a novel approach to adaptively quantize deep neural networks, particularly suited for scenarios where memory and computational resources are limited.

## Overview

QAN leverages B-spline transformations to adaptively adjust the grid points for quantization, allowing for fine-grained control over the quantization process. This enables the model to achieve high accuracy while significantly reducing memory footprint and computational overhead.

## Key Features

- **Adaptive Quantization**: QAN dynamically adjusts the quantization grid based on input data, optimizing the trade-off between model accuracy and resource efficiency.
- **Efficient Training and Inference**: QAN supports efficient training and inference pipelines, with minimal overhead compared to traditional deep learning models.

## Requirements

- Python 3.11
- PyTorch
- torchvision
- wandb (Weights & Biases)
- tqdm
