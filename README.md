# ML Practice

My machine learning and data analysis portfolio: classical ML, deep learning,
NLP and Kaggle competitions, plus written solutions to the underlying math.
Each project folder has its own README with the task, approach and results.

## Highlights

- **[Handwritten Chinese character recognition](deep-learning/handwritten-chinese-characters/)**:
  a CNN built from scratch for 1,000 classes. A series of controlled
  experiments raised the Kaggle score from 0.815 to **0.945**.
- **[Disaster tweet classification](nlp/disaster-tweets-embeddings/)**:
  seven text representations compared, from bag-of-words to fine-tuned BERT
  (**0.820** F1).
- **[Shopee product matching](competitions/shopee-product-matching/)**:
  multimodal retrieval that combines ResNet-50 image embeddings, MiniLM title
  embeddings and a CatBoost pair classifier.

## Projects

### Classical ML

| Project | Task | Methods | Result |
| --- | --- | --- | --- |
| [Linear models](classical-ml/linear-models/) | Binary classification | Logistic regression, linear SVM, decision boundaries, support vectors | 0.98 accuracy |
| [Ensembles](classical-ml/ensembles/) | Income > $50K (UCI Adult) | Decision trees, bagging, stacking, GridSearchCV | 0.873 ROC-AUC |
| [Clustering](classical-ml/clustering/) | Customer and location segmentation | K-Means from scratch, elbow method, Folium map | Interactive map |

### Deep learning

| Project | Task | Methods | Result |
| --- | --- | --- | --- |
| [PyTorch basics](deep-learning/pytorch-basics/) | Tensor operations, CPU vs GPU | Custom trace and Taylor-series sin, GPU benchmarking | Up to ~46× GPU speed-up |
| [MLP: song year](deep-learning/mlp-song-year-prediction/) | Regression (YearPredictionMSD) | MLP, target and input normalization, dropout, LR / weight-decay search | 74.8 MSE (Ridge: 89.8) |
| [CNN: CIFAR-10](deep-learning/cnn-cifar10/) | Image classification | ResNet block, LR schedulers, augmentations, W&B | 72.1% accuracy |
| [Chinese characters](deep-learning/handwritten-chinese-characters/) ⭐ | 1,000-class image classification | Residual CNN, skip connections, augmentation study | 0.945 Kaggle, 97.6% local test |

### NLP

| Project | Task | Methods | Result |
| --- | --- | --- | --- |
| [Disaster tweets](nlp/disaster-tweets-embeddings/) | Binary text classification | BoW, TF-IDF, Word2Vec, GloVe, BERT embeddings, BERT fine-tuning | 0.820 F1 (Kaggle) |

### Competitions

| Project | Task | Methods | Result |
| --- | --- | --- | --- |
| [Shopee product matching](competitions/shopee-product-matching/) | Multimodal duplicate search | ResNet-50 + MiniLM embeddings, pHash, CatBoost pair classifier, kNN | 0.973 pair F1 (val) |
| [Article pair relevance](competitions/zen-article-relevance/) | 4-class text-pair relevance | Sentence transformers, symmetric augmentation, TTA, CatBoost ensemble | 0.72 weighted F1 (val) |

### Theory

| Topic | Contents |
| --- | --- |
| [Linear algebra](theory/linear-algebra/) | Vectors, matrices, linear spaces, operators, SVD and tensors (5 assignments, LaTeX, with an English translation) |
| [Calculus](theory/calculus/) | Limits and derivatives |
| [ML basics](theory/ml-basics/) | Feature types and formulating ML tasks |

## Repository structure

```text
.
├── classical-ml/
│   ├── linear-models/
│   ├── ensembles/
│   └── clustering/
├── deep-learning/
│   ├── pytorch-basics/
│   ├── mlp-song-year-prediction/
│   ├── cnn-cifar10/
│   └── handwritten-chinese-characters/
├── nlp/
│   └── disaster-tweets-embeddings/
├── competitions/
│   ├── shopee-product-matching/
│   └── zen-article-relevance/
└── theory/
    ├── linear-algebra/
    ├── calculus/
    └── ml-basics/
```

## Getting started

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

Datasets are not stored in the repository. Each project README says where to
get its data and where to put it. The competition notebooks were written for
Kaggle and read data from `/kaggle/input/...`. Some deep-learning notebooks
pick the Apple Silicon GPU (`mps`) or CUDA automatically when one is
available.

## Language note

The notebooks were written as coursework and their markdown, comments and
answers are mostly in **Russian**. Rerunning them would take many hours of
GPU time, so the saved outputs are kept as they are. The project READMEs give
full English descriptions of each task, approach and result.

## Tech stack

Python · NumPy · pandas · scikit-learn · PyTorch · torchvision ·
Hugging Face Transformers · sentence-transformers · CatBoost · gensim · NLTK ·
Matplotlib · seaborn · Folium · Weights & Biases · LaTeX
