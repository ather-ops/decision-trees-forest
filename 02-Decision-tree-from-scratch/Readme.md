# Decision Tree from Scratch

This folder builds a complete Decision Tree classifier using only Python and NumPy — no scikit-learn, no abstractions. The work is spread across seven notebooks, one concept per day, followed by a real project that applies everything.

---

## Folder Structure

```
02-Decision-tree-from-scratch/
├── Decision-tree-day1.ipynb
├── Decision-tree-day2.ipynb
├── Decision-tree-day3.ipynb
├── Decision-tree-day4.ipynb
├── Decision-tree-day5.ipynb
├── Decision-tree-day6.ipynb
├── Decision-tree-day7.ipynb
└── Projects/
    └── 01-Drug-Classifier.py
```

---

## Daily Breakdown

**Day 1 — Tree Structure and Impurity Measures**
Introduction to how a decision tree is structured as a graph of nodes and edges. Covers the two main impurity measures: Gini impurity and Entropy. Understanding these is the prerequisite for everything that follows.

**Day 2 — Split Conditions**
How the tree decides where to draw a boundary. Covers axis-aligned splits, binary vs multi-way splits, and why greedy splitting at each node is the standard approach despite being locally optimal rather than globally optimal.

**Day 3 — Growing the Tree**
The recursive divide-and-conquer strategy that builds the tree top-down. Covers how base cases — maximum depth reached, minimum samples per node, pure leaf — stop the recursion.

**Day 4 — Training Algorithm**
Full implementation of the `fit()` method. The `Node` class is defined here, holding feature index, threshold, left and right children, and leaf value. The `_grow_tree()` function is implemented recursively.

**Day 5 — The Exact Splitter**
The core of the algorithm — `_best_split()`. For every feature and every possible threshold, compute the information gain of the proposed split and keep the one that produces the purest partition. This is the most computationally expensive step and the most important one to understand.

**Day 6 — Overfitting and Pruning**
A decision tree grown without constraints memorizes the training data completely. Covers `max_depth`, `min_samples_split`, and `min_samples_leaf` as regularization parameters. Shows the effect of each on training vs validation accuracy.

**Day 7 — Feature Importance**
How to measure each feature's contribution to the total impurity reduction across all splits. Features that appear near the root and split large node populations score highest. This is how you identify which variables actually drive the prediction.

---

## Project — Drug Classifier

**File:** `Projects/01-Drug-Classifier.py`

**Dataset:** drug200.csv — 200 patients, five possible drug prescriptions, four clinical features.

**Features:** Age, Sex, Blood Pressure, Cholesterol, Na-to-K ratio

**Target classes:** drugA, drugB, drugC, drugX, drugY

The Na-to-K ratio consistently appears as the dominant split at the root of the tree. This is not a coincidence — it is the most informative single feature in the dataset, and the entropy calculation in Day 5 surfaces it automatically.

This project runs the full pipeline: data loading, encoding categorical features, train-test split, fitting the from-scratch tree, and evaluating accuracy against the sklearn equivalent to verify correctness.

---

## What You Need

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Open notebooks in order from Day 1 through Day 7. Do not skip to the project without completing Day 5 — the splitter implementation is what makes the project meaningful rather than mechanical.

---

## Part of

[decision-trees-forest](https://github.com/ather-ops/decision-trees-forest) — a two-week study project covering Decision Trees and Random Forest, from scratch and with sklearn, side by side.
