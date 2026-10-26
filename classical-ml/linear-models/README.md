# Logistic Regression and SVM

Binary classification with linear models in scikit-learn: training logistic
regression and a linear SVM, visualizing decision boundaries and support
vectors, and comparing classification metrics.

**Notebook:** [`logistic_regression_svm.ipynb`](logistic_regression_svm.ipynb)
(text and comments in Russian)

## What I did

### Logistic regression on synthetic data

- Generated a 2D binary dataset with `make_classification`.
- Split it into train/test sets and applied min-max scaling.
- Trained `LogisticRegression` and plotted the predicted probability
  $p(y = +1 \mid x)$ over a grid, including the $p = 0.5$ decision boundary.
- Wrote a helper that reports accuracy, precision, recall and F1.

### Linear SVM

- Trained `SVC(kernel="linear")` on a second synthetic dataset.
- Extracted and highlighted the support vectors and the weight vector.
- Used the support vectors to reason about linear separability: many points
  lie inside the margin, so the classes are not linearly separable.

## Results

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.980 | 0.990 | 0.970 | 0.980 |
| Linear SVM | 0.960 | 0.980 | 0.942 | 0.961 |

## Status

The final part of the assignment — a logistic-regression pipeline on the
[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
dataset with ROC analysis and profit-based threshold selection — is
**not completed yet**; those cells are still the assignment template.

## Tech

`scikit-learn` · `numpy` · `pandas` · `matplotlib` · `seaborn`
