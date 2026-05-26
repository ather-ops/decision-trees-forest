# 03 — Decision Tree with Scikit-Learn

This folder is the sklearn companion to `02-Decision-tree-from-scratch`. The algorithm is the same — the implementation is now done through scikit-learn rather than pure Python.

Working through the from-scratch folder first is strongly recommended. Once you know what `max_depth` is actually doing inside the recursive builder, every sklearn parameter becomes transparent rather than a knob you tune blindly.

---

## Folder Structure

```
03-Decision-tree-with-sklearn/
├── Projects/
│   └── 01-Drug-Classifier.py       # End-to-end project on drug200 dataset
├── Decision-tree-sklearn-day1.ipynb
├── Decision-tree-sklearn-day2.ipynb
├── Decision-tree-sklearn-day3.ipynb
├── Decision-tree-sklearn-day4.ipynb
├── Decision-tree-sklearn-day5.ipynb
├── Decision-tree-sklearn-day6.ipynb
└── Decision-tree-sklearn-day7.ipynb
```

---

## Daily Breakdown

| Day | Topic |
|-----|-------|
| 1 | DecisionTreeClassifier — fitting, predicting, basic evaluation |
| 2 | Tree visualization with `plot_tree` and `export_text` |
| 3 | Controlling overfitting — `max_depth`, `min_samples_split`, `min_samples_leaf` |
| 4 | Feature importance — `feature_importances_` and what it measures |
| 5 | Cross-validation with `cross_val_score` — why a single train/test split is not enough |
| 6 | Hyperparameter tuning with `GridSearchCV` |
| 7 | DecisionTreeRegressor — applying the same concepts to continuous targets |

---

## The Project — Drug Classifier

**Dataset:** drug200.csv — 200 patients, five possible drug prescriptions.

**Features:** Age, Sex, Blood Pressure, Cholesterol, Na-to-K ratio.

**Target:** drugA, drugB, drugC, drugX, or drugY.

**What the project covers:**

- Loading and preprocessing the dataset
- Encoding categorical features with `LabelEncoder`
- Training `DecisionTreeClassifier`
- Evaluating with accuracy score and classification report
- Visualizing the full tree
- Tuning depth and comparing performance

Na-to-K ratio tends to dominate the first split in almost every tree grown on this dataset. That is not a coincidence — it emerges from the entropy calculation, and you understand exactly why if you came from folder 02.

---

## From-Scratch vs Sklearn — What Changes

| Aspect | Folder 02 (from scratch) | Folder 03 (sklearn) |
|--------|--------------------------|----------------------|
| Implementation | Manual Node class, recursive builder | `DecisionTreeClassifier()` |
| Splitting logic | Entropy + information gain written by hand | Same math, optimized C under the hood |
| Tree visualization | Print statements | `plot_tree()` with full graphical output |
| Pruning | Manual `max_depth` check in recursion | Parameters passed directly to constructor |
| Speed | Slow on large datasets | Fast — compiled internally |

The behavior is identical. The abstraction level is different.

---

## Key Parameters

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='entropy',       # or 'gini' — the split quality measure
    max_depth=5,               # maximum tree depth — primary overfitting control
    min_samples_split=10,      # minimum samples required to split a node
    min_samples_leaf=4,        # minimum samples required at a leaf node
    random_state=42
)
```

Start with `max_depth=None` (unlimited) to see the fully grown tree, then reduce it until validation accuracy peaks. That trade-off is the bias-variance balance in practice.

---

## How to Run

```bash
# Individual notebooks
jupyter notebook Decision-tree-sklearn-day1.ipynb

# Project
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
| 02-Decision-tree-from-scratch | Same algorithm in pure Python — start here |
| 03-Decision-tree-with-sklearn | This folder |
| 04-Random-Forest-from-scratch | Ensemble of decision trees built manually |
