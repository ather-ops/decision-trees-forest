# 05 — Random Forest with Scikit-Learn

The sklearn companion to `04-Random-Forest-from-scratch`. The algorithm is identical — the implementation switches to `RandomForestClassifier` and `RandomForestRegressor` from sklearn, which are optimized in C and expose the full set of production tuning parameters.

Work through folder 04 first. When you set `n_estimators=100` here, you should already know that means 100 bootstrap samples, 100 separately trained trees, and 100 sets of predictions being aggregated. When `feature_importances_` returns a vector, you should know it is summing impurity reductions across all splits in all trees. The from-scratch folder makes those connections obvious.

---

## Folder Structure

```
05-Random-Forest-with-sklearn/
├── Projects/
│   └── 01-Drug-Classifier.py       # Full sklearn Random Forest pipeline on drug200
├── Day1-sklearn.ipynb
├── Day2-sklearn.ipynb
├── Day3-sklearn.ipynb
├── Day4-sklearn.ipynb
├── Day5-sklearn.ipynb
├── Day6-sklearn.ipynb
└── Day7-sklearn.ipynb
```

---

## Daily Breakdown

| Day | Topic |
|-----|-------|
| 1 | `RandomForestClassifier` — fitting, predicting, basic evaluation |
| 2 | `feature_importances_` — which features drive the splits across all trees |
| 3 | OOB score — `oob_score=True` gives free validation without a held-out set |
| 4 | Cross-validation with `cross_val_score` — confirming the OOB estimate |
| 5 | Hyperparameter tuning — `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| 6 | `GridSearchCV` — systematic search over the parameter space |
| 7 | `RandomForestRegressor` — the same ensemble applied to continuous targets |

---

## The Project — Drug Classifier

**Dataset:** drug200.csv — 200 patients, five drug classes.

**What this project adds over folder 04:**

- Trains with `RandomForestClassifier` — significantly faster than the pure Python version
- Extracts and plots `feature_importances_` across all 100 trees
- Compares OOB score with 5-fold cross-validation score — they should be close
- Runs `GridSearchCV` to find the best `n_estimators` and `max_depth` combination
- Reports final accuracy, classification report, and confusion matrix

---

## Key Parameters

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,        # number of trees — more trees reduce variance but increase compute
    criterion='entropy',     # split quality measure — same as decision tree
    max_depth=None,          # maximum tree depth — None grows trees fully
    max_features='sqrt',     # features considered per split — sqrt(n_features) for classification
    min_samples_split=2,     # minimum samples required to split a node
    min_samples_leaf=1,      # minimum samples required at a leaf
    oob_score=True,          # compute out-of-bag accuracy for free
    n_jobs=-1,               # use all CPU cores
    random_state=42
)
```

The single most important parameter to tune is `max_depth`. Start with `None` to see peak training accuracy, then reduce until the validation score stabilizes. The point where OOB score stops improving is your optimal depth.

---

## From-Scratch vs Sklearn — What Changes

| Aspect | Folder 04 (from scratch) | Folder 05 (sklearn) |
|--------|--------------------------|----------------------|
| Implementation | Manual bootstrap, tree list, vote loop | `RandomForestClassifier()` |
| Speed | Slow — pure Python loops | Fast — parallel C implementation |
| Feature importance | Not implemented | `feature_importances_` built in |
| OOB score | Manual tracking per tree | `oob_score=True` parameter |
| Hyperparameter tuning | Manual loops | `GridSearchCV` |
| Visualization | Print outputs | `plot_tree()` on individual estimators |

Everything in the sklearn version is behavior you already implemented in folder 04. The abstraction is thinner than it looks.

---

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':       [50, 100, 200],
    'max_depth':          [None, 5, 10, 20],
    'max_features':       ['sqrt', 'log2'],
    'min_samples_leaf':   [1, 2, 4]
}

grid = GridSearchCV(
    RandomForestClassifier(oob_score=True, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

---

## Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Mean impurity decrease across all trees')
plt.show()
```

A feature's importance score is the average reduction in impurity it causes, weighted by the number of samples it splits, summed across all trees. Features near zero can often be dropped without hurting accuracy — and dropping them reduces noise.

---

## How to Run

```bash
jupyter notebook Day1-sklearn.ipynb
python Projects/01-Drug-Classifier.py
```

---

## Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib
```

---

## Related Folders

| Folder | Content |
|--------|---------|
| 01-Data | drug200.csv dataset |
| 02-Decision-tree-from-scratch | Base learner — the tree inside the forest |
| 03-Decision-tree-with-sklearn | Sklearn decision tree |
| 04-Random-Forest-from-scratch | Same algorithm in pure Python — work through this first |
| 05-Random-Forest-with-sklearn | This folder |
