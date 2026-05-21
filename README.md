# Decision Trees & Random Forest

A structured, two-week study project that builds both algorithms from the ground up — then validates every concept against scikit-learn. The central application is a drug-response classifier trained on real patient data.

---

## What This Repository Covers

This is not a collection of copy-pasted tutorials. Every folder moves from raw theory to working code, starting with pure Python implementations that expose the internals, then cross-referencing them with production-grade sklearn equivalents so you know exactly what the library is doing under the hood.

The project runs on the **drug200** dataset: 200 patients, five possible drug prescriptions (drugA, drugB, drugC, drugX, drugY), and four clinical features — Age, Sex, Blood Pressure, Cholesterol, and Na-to-K ratio. A clean, real problem that fits comfortably in a notebook and gives immediate, interpretable results.

---

## Repository Structure

```
decision-trees-forest/
├── 01-Data/
│   └── drug200.csv                   Patient dataset (200 rows, 5 features)
│
├── 02-Decision-tree-from-scratch/    Pure Python implementation
│   └── Covers: node structure, splitting logic, entropy, information gain,
│              recursive build, and feature importance
│
├── 03-Decision-tree-with-sklearn/    Scikit-learn equivalent
│   └── Covers: DecisionTreeClassifier, visualization, cross-validation,
│              pruning via max_depth / min_samples_split
│
├── 04-Random-Forest-from-scratch/    Ensemble built manually
│   └── Covers: bootstrap sampling, feature randomization, majority voting,
│              out-of-bag error estimation
│
└── 05-Random-Forest-with-sklearn/    Scikit-learn RandomForestClassifier
    └── Covers: hyperparameter tuning, feature importance, OOB score
```

---

## Learning Path

The material is structured as a 14-day curriculum. Each day has a focused concept so nothing compounds before it has been properly introduced.

### Week 1 — Decision Trees

| Day | Topic |
|-----|-------|
| 1 | Tree structure and impurity measures (Gini, Entropy) |
| 2 | Split conditions: axis-aligned vs oblique, binary vs multi-way |
| 3 | Growing the tree — greedy divide-and-conquer strategy |
| 4 | Training algorithm — recursive build with base cases |
| 5 | Exact splitter — entropy calculation and information gain |
| 6 | Overfitting, pruning strategies, regularization parameters |
| 7 | Feature importance — how splits contribute to impurity reduction |

### Week 2 — Random Forests

| Day | Topic |
|-----|-------|
| 8 | Ensemble learning — why single trees overfit, what bagging buys you |
| 9 | Bootstrap sampling — constructing diverse training subsets |
| 10 | Feature randomization — reducing tree correlation |
| 11 | Aggregation — majority voting for classification, averaging for regression |
| 12 | Out-of-bag evaluation — free validation without a held-out set |
| 13 | Hyperparameter tuning — n_estimators, max_features, max_depth |
| 14 | Complete project — end-to-end pipeline on drug200 |

---

## The Drug Classifier Problem

Given a patient's clinical profile, predict which of five drugs they are most likely to respond to.

**Features:** Age, Sex, Blood Pressure (LOW/NORMAL/HIGH), Cholesterol (NORMAL/HIGH), Na_to_K ratio

**Target classes:** drugA, drugB, drugC, drugX, drugY

**Baseline accuracy (simple decision tree):** ~70%

The dataset is clean, small enough to iterate on quickly, and meaningful enough that predictions carry real-world intuition. Na_to_K ratio, for instance, tends to be the dominant split at the root of most trees grown on this data — a fact that emerges naturally when you implement the splitter yourself.

---

## Key Concepts Implemented From Scratch

**Decision Tree**
- `Node` class holding feature index, threshold, left/right children, and leaf value
- Entropy and information gain as the split criterion
- `_best_split()` scanning all features and thresholds for the purest partition
- Recursive `_grow_tree()` with configurable `max_depth` and `min_samples_split`
- `predict()` traversing the tree from root to leaf for each sample

**Random Forest**
- Bootstrap sampling with `np.random.choice(..., replace=True)`
- Random feature subsets drawn at each split (typically `sqrt(n_features)`)
- Collection of `DecisionTree` instances trained independently
- Majority vote aggregation across all trees at inference time
- Out-of-bag samples tracked per tree for unbiased error estimation

---

## Technologies

| Tool | Role |
|------|------|
| Python 3 | Core implementation language |
| NumPy | Vectorized array operations for splitting and sampling |
| Pandas | Data loading, preprocessing, and exploration |
| Matplotlib | Tree visualization and feature importance plots |
| Scikit-Learn | Reference implementations for validation and benchmarking |

All notebooks run in a standard Jupyter environment. No GPU required. No exotic dependencies.

---

## Getting Started

```bash
git clone https://github.com/ather-ops/decision-trees-forest.git
cd decision-trees-forest
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

Open the folders in order. Start with `02-Decision-tree-from-scratch` before touching the sklearn version — the from-scratch implementation makes every sklearn parameter meaningful.

---

## Project Layout Rationale

The parallel structure (from-scratch alongside sklearn) is intentional. Working through a from-scratch implementation first makes the sklearn API transparent rather than magical. When you set `max_depth=5` in `DecisionTreeClassifier`, you already know which line of your recursive builder that controls. When `feature_importances_` returns a vector, you understand exactly what impurity sum it is aggregating.

---

## License

MIT. Use freely.

---

## Author

[ather-ops](https://github.com/ather-ops)
