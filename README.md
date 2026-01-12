# COMP 551 - Assignment 2

This folder contains our Assignment 2 work for COMP 551 (Winter 2025). The main notebook `assignment2_group-59.ipynb` implements linear regression and classification models and evaluates them on two tabular datasets: Breast Cancer Wisconsin (diagnostic) and Wine Recognition (UCI).

## Contents
- `assignment2_group-59.ipynb`: Full assignment workflow and results (primary deliverable).
- `BreastCancer.ipynb`, `WineRecognition.ipynb`: Dataset-specific notebooks and experiments.
- `src/`: Core implementations (gradient descent, simple regression, multi-class linear/logistic regression).
- `figures/`: Saved plots from the notebooks.
- `reports/`: Assignment PDF and submitted report.
- `requirements.txt`: Python dependencies.

## What is covered
- Data preprocessing and feature scaling for both datasets.
- Simple regression for feature importance (Breast Cancer and Wine).
- Custom implementations:
  - Gradient Descent
  - Multi-Class Linear Regression
  - Multi-Class Logistic Regression
- Baselines/evaluation using scikit-learn (e.g., KNN, Decision Trees) and cross-validation.

## Requirements
Python 3 with the following libraries:
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `ucimlrepo`

Install with:
```bash
pip install -r requirements.txt
```

## How to run
1) Create/activate a Python environment.
2) Install dependencies.
3) Open the notebook with Jupyter:
```bash
jupyter lab
```
4) Run `assignment2_group-59.ipynb` top to bottom.

## Notes
- Both datasets are fetched programmatically using `ucimlrepo`.
- Plots are saved under `figures/`.
