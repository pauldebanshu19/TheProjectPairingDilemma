# The Project Pairing Dilemma: Team vs Solo Preference Predictor

Predict whether a student prefers working Solo (0) or in a Team (1) from a small survey. The notebook auto-detects messy CSV headers, cleans inputs, and compares multiple ML models with cross-validated hyperparameter search and threshold tuning.

## Project highlights
- Robust column detection for noisy/messy CSV headers (whitespace/newlines/symbols).
- Features: introversion_extraversion, risk_taking, club_top1, weekly_hobby_hours.
- Target: teamwork_preference (Likert 1–5). Binarized as Solo (1–2) vs Team (4–5). Neutral (3) rows dropped.
- Models compared: Logistic Regression, Random Forest, Linear SVM, KNN, HistGradientBoosting.
- Speed-ups: cached preprocessing, randomized search, fewer CV folds, fast estimators.
- Evaluation: accuracy, balanced accuracy, ROC-AUC, confusion matrices, ROC overlay, CV-based threshold tuning.

## Repository structure
- project.ipynb — main notebook with data cleaning, modeling, tuning, and evaluation
- data.csv — input dataset (expected in repo root)
- requirements.txt — Python dependencies
- .gitignore — ignores cache/checkpoints and IDE clutter

## Model Pipeline

![alt text](project-1.jpg)

## Quick start
Prereqs: Python 3.9+ and pip.

1) Install dependencies
   pip install -r requirements.txt

2) Launch the notebook
   jupyter notebook project.ipynb

3) Run all cells. A cache directory sklearn_cache/ will be created to speed up repeated runs.

Notes
- The notebook will try to auto-locate required columns in data.csv by fuzzy/keyword matching; see the console prints for found mappings.
- If a required column cannot be found, the run stops early and prints available columns.

## Data expectations
The notebook searches for columns that match these concepts (robust to messy headers):
- introversion_extraversion
- risk_taking
- weekly_hobby_hours
- club_top1
- teamwork_preference

Target binarization
- Solo = 1–2, Team = 4–5; Neutral (3) dropped for a clean binary task.

## Reproduce results
- Split: stratified train/test (25% test).
- Tuning: RandomizedSearchCV with 3-fold CV, scoring=balanced_accuracy.
- Best model is selected by test balanced accuracy; if probabilistic, a CV-based threshold sweep further tunes the decision threshold.

Outputs
- Printed metrics per model (Accuracy, Balanced Acc, ROC-AUC) and classification reports.
- Confusion matrices and an ROC comparison plot.
- Final summary line with best model and decision threshold (if applicable).

## Custom prediction
The notebook exposes a helper function:
- predict_preference(introversion_extraversion, risk_taking, club_top1, weekly_hobby_hours) -> dict with prob_team, pred_label, pred_text

## Troubleshooting
- Column not found: check data.csv headers; adjust tokens in the notebook’s name_map if needed.
- Class imbalance: balanced_accuracy and class_weight are used; you can also tweak threshold search ranges.
- Slow runs: reduce n_iter in RandomizedSearchCV, lower n_estimators for forests, or disable some models.

## License
No license specified. Add one if you plan to share or reuse.

