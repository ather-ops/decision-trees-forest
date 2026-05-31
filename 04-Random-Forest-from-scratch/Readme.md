# 04 — Random Forest from Scratch

A week-by-week build of the Random Forest algorithm in pure Python and NumPy — no sklearn, no shortcuts. Each day adds one component until the full ensemble is running and evaluated on the drug200 dataset.

This folder follows directly from `02-Decision-tree-from-scratch`. The decision tree built there becomes the base learner here. If you have not worked through folder 02, the bootstrap sampling and majority vote logic in this folder will make less sense because the tree underneath it is a black box.

---

## Folder Structure

```
04-Random-Forest-from-scratch/
├── Projects/
│   └── 01-Drug-Classifier.py       # Full Random Forest pipeline on drug200
├── Day1-scratch.ipynb
├── Day2-scratch.ipynb
├── Day3-scratch.ipynb
├── Day4-scratch.ipynb
├── Day5-scratch.ipynb
├── Day6-scratch.ipynb
└── Day7-scratch.ipynb
```

---

## Daily Breakdown

| Day | Topic |
|-----|-------|
| 1 | Why a single decision tree overfits — the variance problem that forests solve |
| 2 | Bootstrap sampling — creating diverse training subsets with `np.random.choice(..., replace=True)` |
| 3 | Feature randomization — drawing a random subset of features at each split |
| 4 | Training multiple trees — building a collection of `DecisionTree` instances independently |
| 5 | Majority vote aggregation — combining predictions from all trees at inference time |
| 6 | Out-of-bag evaluation — using the samples left out of each bootstrap as a free validation set |
| 7 | Full assembly — complete `RandomForest` class with `fit()`, `predict()`, and OOB score |

---

## What Gets Built

A `RandomForest` class in pure Python with the following components:

**Bootstrap sampling.** Each tree in the forest is trained on a different random subset of the training data, sampled with replacement. This means some rows appear multiple times and some not at all. Trees trained on different subsets make different errors, and averaging those errors is what reduces variance.

**Feature randomization.** At every split inside every tree, only a random subset of features is considered — typically `sqrt(n_features)` for classification. This forces the trees to be different from each other even when they see similar data. Correlated trees do not help the ensemble.

**Majority vote.** At prediction time, every tree in the forest independently classifies the input. The class that receives the most votes becomes the final prediction. For regression it is the mean of all outputs.

**Out-of-bag score.** Because each tree is trained on a bootstrap sample, roughly one third of the data is never seen by any given tree. Those unused samples become a validation set for that tree. Averaging OOB accuracy across all trees gives a reliable generalization estimate without a held-out split.

---

## The Project — Drug Classifier

**Dataset:** drug200.csv — 200 patients, five drug classes.

**What this project adds over the decision tree project in folder 02:**

- Trains 100 trees on bootstrap samples of the training data
- Aggregates predictions via majority vote
- Reports OOB accuracy alongside test accuracy
- Shows that the ensemble consistently outperforms a single tree, especially when individual trees are allowed to grow deep

---

## Core Implementation Pattern

```python
import numpy as np
from decision_tree import DecisionTree   # built in folder 02

class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees         = n_trees
        self.max_depth       = max_depth
        self.min_samples_split = min_samples_split
        self.n_features      = n_features
        self.trees           = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap(self, X, y):
        n = X.shape[0]
        indices = np.random.choice(n, n, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # majority vote across all trees
        return [np.bincount(predictions[:, i]).argmax() for i in range(X.shape[0])]
```

---

## How to Run

```bash
jupyter notebook Day1-scratch.ipynb
python Projects/01-Drug-Classifier.py
```

---

## Dependencies

```bash
pip install numpy pandas matplotlib
```

No scikit-learn. The `DecisionTree` class imported in the core pattern above is the one built in folder 02.

---

## Related Folders

| Folder | Content |
|--------|---------|
| 01-Data | drug200.csv dataset |
| 02-Decision-tree-from-scratch | The base learner used inside this forest |
| 03-Decision-tree-with-sklearn | Sklearn decision tree for comparison |
| 04-Random-Forest-from-scratch | This folder |
| 05-Random-Forest-with-sklearn | Same algorithm via sklearn — faster, with tuning tools |
