<p align="center">
  <img src="https://github.com/ather-ops/decision-trees-forest/blob/main/Screenshots/ip.png" alt="Decision Trees and Random Forest banner" width="100%">
</p>

<h1 align="center">Decision Trees &amp; Random Forest</h1>

<p align="center">
  A complete, end-to-end repository covering Decision Trees and ensemble methods (Random Forest) —
  built from scratch and re-implemented with scikit-learn.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-workflows-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/NumPy-from--scratch-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/status-complete-22c55e?style=flat-square" alt="Status: Complete">
  <img src="https://img.shields.io/badge/license-MIT-9a4fd0?style=flat-square" alt="MIT License">
</p>

---

## About

This repository is a structured, end-to-end deep dive into **Decision Trees**, **Random Forests**, and **Gradient Boosting** — foundational tree-based algorithms in machine learning. Core concepts are implemented twice: once **from scratch** using only NumPy and core Python, and once using **scikit-learn**, so the underlying mechanics and the production-ready workflow can be compared side by side. Gradient boosting is included as a scikit-learn-only extension, benchmarked against the random forest results.

The project is **complete**. All notebooks, implementations, and the final drug classification project are finished and will not receive further changes.

---

## Repository structure

| Folder | Contents |
|---|---|
| `01-Data` | `drug200.csv` — the dataset used throughout the project |
| `02-Decision-tree-from-scratch` | Full decision tree implementation built from first principles |
| `03-Decision-tree-with-sklearn` | Equivalent decision tree workflow using scikit-learn |
| `04-Random-Forest-from-scratch` | Random forest (bagging ensemble) implementation built from first principles |
| `05-Random-Forest-with-sklearn` | Equivalent random forest workflow using scikit-learn |
| `06-Gradient-Boosting-with-sklearn` | Gradient boosting workflow using scikit-learn |
| `Screenshots` | Banner and reference images used in this README |

---

## Learning path

### Week 1 — Decision Tree

| Day | Topic |
|---|---|
| 1 | Tree structure and impurity measures |
| 2 | Split conditions — axis-aligned vs. oblique, binary vs. non-binary |
| 3 | Growing a tree via greedy divide-and-conquer |
| 4 | Training algorithm and recursive tree building |
| 5 | Exact splitter — entropy and information gain |
| 6 | Overfitting, pruning, and regularization |
| 7 | Feature importance |

### Week 2 — Random Forest

| Day | Topic |
|---|---|
| 8 | Ensemble learning and bagging |
| 9 | Bootstrap sampling |
| 10 | Feature randomization |
| 11 | Aggregation — majority voting and averaging |
| 12 | Out-of-bag (OOB) evaluation |
| 13 | Hyperparameter tuning |
| 14 | Final project, end to end |

**Bonus: Gradient Boosting**

- Boosting vs. bagging — sequential error correction instead of parallel averaging
- Gradient boosting workflow with scikit-learn, benchmarked against the random forest results

---

## Project: drug classifier

A classification model that predicts the drug type a patient should be prescribed, based on patient attributes.

**Input features**
- Age
- Sex
- Blood pressure (BP)
- Cholesterol
- Na-to-K ratio

**Output classes**
- `drugY`, `drugX`, `drugC`, `drugA`, `drugB`

**Result:** ~70% accuracy using a simple, single decision tree — establishing a baseline that the random forest implementations build on.

---

## Technologies

Python · NumPy · Pandas · Matplotlib · scikit-learn

---

## Getting started

```bash
git clone https://github.com/ather-ops/decision-trees-forest.git
cd decision-trees-forest
pip install numpy pandas matplotlib scikit-learn jupyter
```

Open any folder in order (`01` through `06`) and step through the notebooks to follow the path from raw data to a complete random forest and gradient boosting classifier.

---

## Author

**Ather-ops**

## License

Released under the [MIT License](LICENSE).
