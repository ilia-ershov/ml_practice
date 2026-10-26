# Handwritten Chinese Character Recognition

**Creative project.** A CNN trained from scratch that classifies handwritten
Chinese characters into 1,000 classes. Starting from a simple baseline, I
improved the Kaggle score from **0.815 to 0.945** through a series of
controlled experiments with the architecture, learning-rate schedule and
augmentations.

| File | Description |
| --- | --- |
| [`01_eda.ipynb`](01_eda.ipynb) | Dataset exploration, image-size analysis, preprocessing checks |
| [`02_final_model.ipynb`](02_final_model.ipynb) | Final model: data pipeline, training, evaluation, Kaggle submission |
| [`report_ru.pdf`](report_ru.pdf) | Full experiment log with plots and code (in Russian) |

This README is an English summary of the report.

## Dataset

- 332,987 grayscale images of handwritten characters, 1,000 classes.
- Classes are roughly balanced (about 280–400 images each).
- Image sizes vary widely: from 8×13 up to 207×164 pixels, with a median of
  about 88×77. The same character also looks very different from writer to
  writer: different stroke thickness, slant and proportions.

Split used for local evaluation: 70% train, 15% validation, 15% test
(stratified by class). The Kaggle test set is separate.

## Preprocessing

All images are resized to **64×64** — a compromise between keeping strokes
legible and keeping training time reasonable — and normalized to $[-1, 1]$.

## Experiment log

Kaggle scores are given as the two leaderboard values reported in the report.

| # | Change | Kaggle score |
| --- | --- | --- |
| — | First baseline: one conv layer → pool → linear. 0.92 validation accuracy, but **0.001 on test** — exactly the accuracy of predicting a constant class | — |
| 0 | ResNet-style block (2 convs + 1×1 shortcut), pool size 4 | 0.815 / 0.818 |
| 1 | MultiStepLR scheduler | no change |
| 2 | **CosineAnnealingLR** scheduler | 0.834 / 0.839 |
| 3 | Pool size 4 → 8 (fewer, more general spatial features) | 0.876 / 0.877 |
| 4 | Pool size 8 → 16 | 0.895 / 0.895 |
| 5 | 4 convolutional layers | 0.924 / 0.924 |
| 6 | 5 conv layers + rotation, affine and blur augmentations | 0.927 / 0.930 |
| 7 | Keep only rotation (±5°) | 0.930 / 0.929 |
| 8 | Rotation + Gaussian blur | 0.927 / 0.929 |
| 9 | Rotation + affine, no blur | 0.923 / 0.923 |
| 10 | Rotation only + a 1×1 shortcut around every conv block and a global shortcut from the input (~114 min of training) | 0.931 / 0.930 |
| 11 | **Kernel size 3 → 5** — a larger receptive field fits the large 64×64 strokes | **0.945 / 0.942** |

Augmentations I ruled out on purpose: horizontal and vertical flips (a
mirrored character is a different character or not a character at all) and
random resized crops (they cut off strokes that define the character).

## Final model

```
input 1×64×64
 ├─ global 1×1 shortcut ─────────────────────────────────────────────┐
 └─ [Conv5×5 → BN (+ 1×1 shortcut) → ReLU] × 4 → Conv5×5 → BN ──────(+) → ReLU
    → AvgPool 16×16 → Linear(32·4·4 → 1000)
```

- 32 channels per layer, SGD (`lr = 0.1`, momentum 0.9), CosineAnnealingLR,
  10 epochs, batch size 64, trained on Apple Silicon (MPS).
- Local results: **97.6% test accuracy** (97.7% validation).

## What I would try next

- Dilated convolutions, to widen the receptive field further.
- Dropout, to reduce overfitting (training accuracy reaches ~99.97%).
- Richer augmentations and other schedulers.

## Data

The competition data is not included. `02_final_model.ipynb` expects
`data/train-{1..4}.npy` and `data/test.npy` next to the notebook (arrays of
`(image, label)` pairs) and writes `data/answers.csv` for submission.

## Tech

`PyTorch` · `torchvision` · `scikit-learn` · `numpy` · `pandas` · `matplotlib`
