# CNN: CIFAR-10 Image Classification

A convolutional classifier with a ResNet-style residual block, trained on
CIFAR-10, followed by experiments with learning-rate schedules, data
augmentation and experiment tracking in Weights & Biases.

**Notebook:** [`cnn_cifar10.ipynb`](cnn_cifar10.ipynb)
(text and comments in Russian; HSE "Introduction to Deep Learning", homework 2)

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 colour
images in 10 classes. It is downloaded automatically by `torchvision`.

## Model

One residual block, as in ResNet:

```
input ─┬─ Conv3×3(32) → BN → ReLU → Conv3×3(32) → BN ─┐
       └─────────────── Conv1×1(32) ──────────────────(+) → ReLU → AvgPool8×8 → Linear(512 → 10)
```

## What I did

- Implemented the network, the training loop and the evaluation loop, and ran
  training on Apple Silicon (MPS).
- Plotted loss and accuracy per optimizer step, with per-epoch validation
  points.
- Compared learning-rate schedules from `torch.optim.lr_scheduler`.
- Compared augmentations from `torchvision.transforms`.
- Logged a training run (metrics, gradients and weights) to Weights & Biases.

## Results (test accuracy)

**Learning-rate schedules** (SGD + momentum, 20 epochs):

| Schedule | Test accuracy |
| --- | --- |
| Constant `lr = 0.1` | 0.7053 |
| Constant `lr = 0.001` | 0.6711 |
| MultiStepLR (milestones 10, 15) | 0.7163 |
| ExponentialLR (γ = 0.95) | 0.7015 |
| **CosineAnnealingLR** | **0.7199** |
| CyclicLR | 0.6786 |

**Augmentations** (with MultiStepLR):

| Augmentation | Test accuracy |
| --- | --- |
| **Random horizontal flip** | **0.7208** |
| Colour jitter | 0.7121 |
| Random rotation (15°) | 0.7102 |
| Random crop (padding 5) | 0.6953 |
| All of the above combined | 0.7113 |

## Takeaways

- Decaying the learning rate helps: a constant small rate converges too slowly
  and a constant large one plateaus early. Cosine annealing worked best.
- Augmentations should match the data. Horizontal flips and colour changes
  produce realistic CIFAR images and help; crops and rotations produce images
  unlike the test set and do not help here. Stacking every augmentation
  distorts the images too much.

## Tech

`PyTorch` · `torchvision` · `wandb` · `matplotlib`
