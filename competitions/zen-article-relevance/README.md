# Article Pair Relevance

Predicting how relevant two articles are to each other on a four-level scale
from their titles and texts (mostly Russian). Built on Kaggle under tight
memory limits.

**Notebook:** [`zen_article_relevance.ipynb`](zen_article_relevance.ipynb)
(Kaggle notebook; comments in Russian)

## Task

The `path-to-zen` competition dataset: an `items` table with each article's
title and content, and a list of article pairs labelled `no_relevant`,
`relevant_minus`, `relevant` or `relevant_plus`. Pairs labelled `doubles` or
`unavailable` are removed. The metric is weighted F1.

## Approach

1. **Text:** title + the first 500 characters of the content for each article.
2. **Symmetric augmentation:** every training pair is also added with left and
   right swapped. This improved the score.
3. **Embeddings** of both articles with the multilingual sentence transformer
   `paraphrase-multilingual-MiniLM-L12-v2`, chosen because heavier models were
   too slow.
4. **Pair features:** `[u, v, |u − v|, u ⊙ v]`, the standard combination for
   sentence-pair classification.
5. **Classifier:** multiclass CatBoost on GPU with balanced class weights.
   Hyperparameters were picked with a small grid search over depth and
   learning rate (depth 8, `lr = 0.1`).
6. **Test-time augmentation:** predict each test pair in both orders and
   average the probabilities.
7. **Ensemble:** a second model built the same way on embeddings from the much
   heavier `DeepPavlov/rubert-base-cased-sentence` (first 750 characters of
   content; computing these embeddings took about two hours). The final
   prediction is a weighted average: 0.6 × MiniLM model + 0.4 × ruBERT model.

## Results (validation weighted F1)

| Model | Weighted F1 |
| --- | --- |
| MiniLM embeddings + symmetric augmentation | 0.720* |
| ruBERT embeddings (no augmentation) | 0.445 |

\* This score is optimistic. After symmetric augmentation, a pair and its
swapped copy can land in different splits, so information leaks from train
to validation. The ruBERT model gets a lower weight in the ensemble because
it scored worse on its own.

## Engineering notes

The merged pair table took more than 5 GB in memory, and the notebook kept
crashing. I added explicit memory cleanup (`del` + `gc.collect()`), saved
embeddings to `.npy` / `.parquet` and reloaded them between steps.

## What I would do differently

- Split by pair (or by article) **before** augmenting, so the validation score
  is reliable.
- Fine-tune a cross-encoder on the pairs instead of using frozen embeddings.

## Tech

`sentence-transformers` · `CatBoost` · `PyTorch` · `scikit-learn` · `pandas` · `numpy`
