# PyTorch Basics

An introduction to PyTorch tensors, custom tensor functions and moving
computation between CPU and GPU.

**Notebook:** [`pytorch_basics.ipynb`](pytorch_basics.ipynb)
(code comments in Russian; run in Google Colab on a T4 GPU)

## What I did

1. **Tensor operations**
   - Created a random 10×10 matrix from the standard normal distribution.
   - Implemented the matrix trace without `torch.trace`, using
     `torch.diagonal` and `torch.sum`.
   - Implemented element-wise $\sin(X)$ through its Taylor series
     $x - \frac{x^3}{3!} + \frac{x^5}{5!} - \dots$ and checked it against
     `torch.sin`.
2. **CPU vs GPU for tensor arithmetic** — timed addition and element-wise
   multiplication of three 10,000×10,000 matrices on each device.
3. **CPU vs GPU for a neural network** — built a small fully connected network
   (`1024 → 512 → 64 → 2` with ReLU and softmax) and ran inference on
   1,000,000 × 1024 inputs on both devices.
4. Answered theory questions on the trade-offs between GPU and CPU.

## Results

| Operation | CPU | GPU (T4) | Speed-up |
| --- | --- | --- | --- |
| $X + Y + Z$ (10k × 10k) | 926 ms | 47 ms | ~20× |
| $X \cdot Y \cdot Z$ element-wise (10k × 10k) | 1.47 s | 32 ms | ~46× |
| MLP inference on 1M samples | 20.8 s | 2.2 s | ~9× |

GPUs have thousands of cores and are much faster at highly parallel work such
as matrix operations; CPUs remain the better choice for small or sequential
workloads.

## Tech

`PyTorch` · Google Colab
