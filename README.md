---
# House Price Prediction — Regularised Regression

## Author: K. Karthik Kumar Reddy

## Program: SAP labs COHORT3


Prepared for submission


This repository contains a reproducible analysis for the house-price assignment using Ridge and Lasso regression. The pipeline includes EDA, preprocessing, hyperparameter selection (5‑fold CV), an alpha-doubling experiment, and a fallback Lasso retraining after removing top predictors. Key outputs (figures, `model_summary.json`, `report.pdf`, and a styled submission PDF) are included for grading.

## Files of interest

- `run_house_model.ipynb` — Notebook with EDA, preprocessing, model selection, experiments and plots. Use `Kernel -> Restart & Run All` to reproduce interactively.
- `run_house_model.py` — Script to run the pipeline headless and produce `model_summary.json` and figures.
- `model_summary.json` — Machine-readable summary of numeric results (best alphas, CV & test RMSEs, top features, retrained top features).
- `generate_report.py` / `report.pdf` — Script and generated report with CV curves and coefficient plots.
- `Subjective_Answers.md` / `subjective.pdf` — answers and the converted PDF for submission.
- `data.txt` — Data dictionary describing dataset columns and values (included for grader reference).
- 'train.csv' - Data set
- `requirements.txt` — Python dependencies required to reproduce results.

## Quick reproduction

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2a. Reproduce interactively (recommended):

```bash
jupyter lab
open run_house_model.ipynb -> Kernel -> Restart & Run All
```

2b. Reproduce headless (script):

```bash
python run_house_model.py
```


## Mapping to evaluation rubric

- ** Data understanding & preparation **: Notebook includes missingness checks, imputation, dummy-variable creation, and a derived `TotalSF` feature where applicable.
- ** Model building & evaluation **: Ridge and Lasso hyperparameter tuning via 5-fold CV, hold-out test evaluation, alpha doubling experiment, and Lasso retraining after removing top predictors are implemented and results are saved to `model_summary.json` and `report.pdf`.
- ** Coding guidelines **: Code comments, `requirements.txt`, and both script and notebook are provided for reproducibility.
- ** Subjective answers **: `Subjective_Answers.md` and `subjective.pdf` contain answers referencing numeric outputs from the analysis.

## Reproducibility notes

- Deterministic seed: `random_state=42` used where applicable.
- CV folds: 5-fold cross-validation used for hyperparameter selection.

---
