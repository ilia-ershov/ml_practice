# Shopee: Product Matching

Finding listings of the same product in an e-commerce catalogue from their
photos and titles. This is a multimodal retrieval task: for every listing,
predict all other listings that sell the same item.

**Notebook:** [`shopee_product_matching.ipynb`](shopee_product_matching.ipynb)
(Kaggle notebook)

## Competition

[Shopee – Price Match Guarantee](https://www.kaggle.com/competitions/shopee-product-matching)
on Kaggle. Each listing has an image, a title and a perceptual image hash
(pHash); listings with the same `label_group` are the same product. The metric
is the mean F1 score of the predicted match lists.

The competition runs without internet access, so the pretrained weights
(ResNet-50 and MiniLM) were uploaded as Kaggle datasets. The first cell of
the notebook shows how they were exported.

## Approach

1. **Embeddings**
   - Images: ImageNet-pretrained **ResNet-50** with the classification layer
     removed (2048-d), L2-normalized. Embeddings are cached on disk.
   - Titles: lower-cased and cleaned, then encoded with the
     **all-MiniLM-L6-v2** sentence transformer.
   - pHash: converted into a 64-bit binary vector.
2. **Pair features** for two listings: cosine similarity, L2 distance and dot
   product for both image and text embeddings, the share of matching pHash
   bits, and the difference in title length.
3. **Pair classifier:** CatBoost trained on 20,000 positive pairs (same
   `label_group`) and 20,000 random negative pairs, with early stopping on a
   validation split.
4. **Inference:** for each test listing, find its nearest neighbours by image
   cosine similarity, score each candidate pair with the classifier and return
   the ranked matches, always including the listing itself.

An earlier version (kept in commented-out cells) simply concatenated the image
and text embeddings and matched everything above a fixed cosine-similarity
threshold of 0.9. The learned pair classifier replaced it.

## Results

- Pair-classifier F1 on the validation pairs: **0.973**.
- The leaderboard score is not recorded in the notebook.

## Ideas for improvement

- Threshold the classifier probability instead of returning a fixed number of
  candidates, and retrieve more candidates (currently the top 3 neighbours).
- Retrieve candidates by text similarity as well as by image similarity.
- Fine-tune the encoders with a metric-learning loss (e.g. ArcFace) on
  `label_group`.

## Tech

`PyTorch` · `torchvision` · `sentence-transformers` · `CatBoost` · `scikit-learn` · `pandas`
