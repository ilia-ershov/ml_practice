# Text Embeddings: Disaster Tweet Classification

Classifying whether a tweet is about a real disaster, comparing text
representations from bag-of-words to a fine-tuned BERT. Each representation
is fed to the same CatBoost classifier head, so the comparison is about the
embeddings alone; the last step fine-tunes BERT end to end.

**Notebook:** [`text_embeddings_disaster_tweets.ipynb`](text_embeddings_disaster_tweets.ipynb)
(code comments in Russian)

## Dataset

The Kaggle competition
[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started):
7,613 labelled tweets for training and 3,263 for testing. The metric is F1.

## What I did

0. **Warm-up:** trained the CatBoost head on precomputed noisy embeddings and
   compared `argmax` predictions with a fixed 0.65 probability threshold.
1. **Bag-of-words** with `CountVectorizer`.
2. **TF-IDF** with `TfidfVectorizer`.
3. **Word2Vec:** trained my own model with `gensim` on the training tweets,
   then switched to pretrained Google News vectors (300-d). Each tweet is the
   mean of its word vectors.
4. **GloVe:** pretrained `glove.6B.100d` vectors with NLTK tokenization.
5. **BERT embeddings:** `bert-base-uncased` used as a frozen encoder.
6. **BERT fine-tuning**, implemented twice: with the Hugging Face `Trainer`
   and with a custom PyTorch training loop (classification head on the
   `[CLS]` token, 3 epochs).

## Results (Kaggle F1)

| Representation | Score |
| --- | --- |
| Word2Vec trained on the tweets | 0.633 |
| Word2Vec, pretrained Google News | 0.775 |
| TF-IDF | 0.774 |
| Bag-of-words | 0.792 |
| GloVe | 0.794 |
| Frozen BERT embeddings | 0.806 |
| **Fine-tuned BERT** (PyTorch loop) | **0.820** |

## Takeaways

- A Word2Vec model trained on only ~6,000 short tweets is too weak; the same
  method with vectors pretrained on a large corpus is far better.
- On short, noisy tweets simple sparse features (bag-of-words) remain a
  strong baseline, on par with averaged pretrained word vectors.
- Contextual embeddings from BERT beat every static representation, and
  fine-tuning the whole model gives the best result.
- With the warm-up embeddings, `argmax` (0.788) slightly beat the fixed 0.65
  threshold (0.787). A threshold is only worth using once it is tuned on
  validation data, for example from the ROC or precision–recall curve.

## Data and models

Not included in the repository. To rerun, place `train.csv`, `test.csv` and
`sample_submission.csv` from the competition next to the notebook, together
with `GoogleNews-vectors-negative300.bin.gz` and `glove.6B.100d.txt`. The
`FAKE_train.csv` / `FAKE_test.csv` files of the warm-up step came with the
assignment.

## Tech

`transformers` · `PyTorch` · `CatBoost` · `gensim` · `NLTK` · `scikit-learn` · `pandas`
