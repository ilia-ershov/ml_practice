# Ensembles: Income Prediction

Predicting whether a person earns more than $50K a year from census data, and
comparing a single decision tree with bagging and stacking ensembles.

**Notebook:** [`ensembles_adult_income.ipynb`](ensembles_adult_income.ipynb)
(text and comments in Russian)

## Dataset

[UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) — 48,842 people with
demographic and employment features. It is downloaded inside the notebook via
`ucimlrepo`, so no manual download is needed.

## What I did

- **Target cleaning:** merged the inconsistent labels (`>50K` / `>50K.`,
  `<=50K` / `<=50K.`) into a binary target (≈24% positive class).
- **Two preprocessing variants:**
  1. one-hot encoding of the raw categorical columns;
  2. missing values filled with `"Other"` and rare categories (< 0.5% of rows)
     grouped together before one-hot encoding.
- **Decision tree** trained on both variants to compare preprocessing.
- **Bagging** of decision trees (`BaggingClassifier`), plus a ROC-AUC grid
  search over `n_estimators`, `max_features`, `max_samples` and tree depth
  (the code is kept but commented out because it took about 30 minutes to run).
- **Base models:** logistic regression, SVM and a decision tree.
- **Stacking** (`StackingClassifier`) with a logistic-regression meta-model,
  then with bagging added as a fourth base model, then with an SVM meta-model.
- Compared all models on precision, recall, F1 and ROC-AUC with bar charts.

## Results (test set)

| Model | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Decision tree (raw one-hot) | 0.657 | 0.524 | 0.583 | 0.822 |
| Decision tree (cleaned features) | 0.660 | 0.521 | 0.583 | 0.824 |
| Bagging (trees) | 0.661 | 0.531 | 0.589 | 0.846 |
| Logistic regression | 0.699 | 0.516 | 0.594 | **0.873** |
| SVM | 0.702 | 0.518 | 0.596 | 0.839 |
| Stacking (LR meta-model) | 0.686 | 0.501 | 0.579 | 0.872 |
| Stacking + bagging | 0.684 | 0.506 | 0.582 | 0.873 |
| Stacking (SVM meta-model) | 0.706 | 0.517 | **0.597** | 0.794 |

## Takeaways

- Bagging clearly improves the ROC-AUC of a single tree (0.82 → 0.85) because
  averaging many trees produces smoother probability estimates.
- Plain logistic regression turned out to be the strongest model by ROC-AUC;
  stacking did not beat it on this dataset.
- An SVM as the meta-model slightly increases F1 but hurts ROC-AUC
  considerably, since SVM probability estimates are poorly calibrated.

## Tech

`scikit-learn` · `pandas` · `numpy` · `matplotlib` · `ucimlrepo`
