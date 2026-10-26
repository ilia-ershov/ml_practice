# MLP: Song Release Year Prediction

Predicting the release year of a song from 90 audio features with a fully
connected neural network, and step by step making the network beat a linear
baseline.

**Notebook:** [`mlp_song_year_prediction.ipynb`](mlp_song_year_prediction.ipynb)
(code comments in Russian; HSE "Introduction to Deep Learning", homework 1)

## Dataset

[YearPredictionMSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
(UCI) — 515,345 songs from 1922–2011, 90 timbre features each. Fixed split:
first 75% for training, the rest for testing; 25% of the training part is
held out for validation.

Download it next to the notebook as `data.txt.zip`:

```bash
wget -O data.txt.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
```

## What I did

1. **Baselines:** best constant prediction and Ridge regression.
2. **Wrote a reusable training loop** (`train_and_validate`) with live loss and
   MSE plots.
3. **Classification vs regression:** trained the network as an 89-class
   classifier first. It lost to the baseline because classification ignores
   that 1998 is closer to 1999 than to 1960.
4. **Made regression work:** raw MSE regression diverged (NaN / MSE in the
   millions). Min-max scaling of the target with a sigmoid output, and then
   standardizing the input features, fixed training.
5. **Optimizers:** SGD with momentum vs Adam over 5 runs each.
6. **Capacity and regularization:** a wider and deeper network overfit;
   adding dropout brought validation and test MSE back in line with training.
7. **Hyperparameter search** over learning rate ($10^{-4}$ to $10^{2}$) and
   weight decay ($10^{-6}$ to $10^{-2}$).
8. **Final model** — `90 → 512 → 256 → 128 → 1` with ReLU, dropout 0.2 and a
   sigmoid output, trained with SGD + momentum, `lr = 1e-2`,
   `weight_decay = 1e-4`.

## Results (test MSE, lower is better)

| Model | Test MSE |
| --- | --- |
| Best constant | 117.63 |
| Ridge regression | 89.76 |
| MLP as a classifier | ~195 (val) |
| MLP regression, normalized target and inputs | ~85.6 (val) |
| **Final MLP with dropout** | **74.83** |

## Tech

`PyTorch` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`
