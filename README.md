# Heart Disease Prediction Using Support Vector Machine (SVM)

A leakage-safe machine learning pipeline for **heart disease pre-screening**, comparing a tuned SVM against Logistic Regression and Random Forest on 918 patient records.

> **Scope:** this is an academic project and a decision-support prototype, not a clinical tool. All results are from internal validation only.

## Highlights

| Model (tuned) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **SVM (RBF, C=10, gamma=0.01)** | 0.891 | 0.880 | **0.931** | 0.905 | **0.9445** |
| Logistic Regression (C=1) | 0.902 | 0.904 | 0.922 | 0.913 | 0.9327 |
| Random Forest (400 trees, depth 10) | 0.880 | 0.870 | 0.922 | 0.895 | 0.9269 |

- All models beat the 0.85 ROC-AUC target on the held-out test set (184 patients).
- SVM was chosen because its recall and ROC-AUC best match a screening goal, where a missed case costs more than a false alarm.
- McNemar's test found **no statistically significant difference** between models (p ≈ 0.73 for SVM vs Logistic Regression), so the choice is driven by the screening objective, not by claimed superiority.
- SVM showed a near-zero train–test ROC-AUC gap; Random Forest showed the largest (about 0.05), consistent with overfitting.

## What's in the pipeline

1. **Data cleaning:** found disguised missing values (172 Cholesterol and 1 RestingBP recorded as 0).
2. **Leakage control:** train/test split (80/20, stratified, `random_state=42`) *before* imputation; scaling and encoding live inside the sklearn `Pipeline`.
3. **Imputation benchmark:** six methods compared (global median, group median, KNN, linear regression, random forest, MICE). Sex × AgeGroup median won, confirmed over 30 repeated splits.
4. **Feature engineering:** `AgeGroup`, `Cholesterol_Missing`, `HR_Achievement_Ratio`, `Is_ASY`, validated by an ablation study (CV ROC-AUC 0.908 → 0.918).
5. **EDA:** point-biserial correlation, Cramér's V, and disease-rate breakdowns.
6. **Modelling:** baseline and `GridSearchCV` tuning (5-fold stratified, ROC-AUC) for all three models.
7. **Evaluation:** confusion matrices, ROC and precision–recall curves, bias–variance check, McNemar's test, calibration (Brier score 0.086).
8. **Interpretability & decision-making:** SHAP explanations and a threshold/cost analysis (e.g. at a 3:1 FN:FP cost ratio, recall rises to 0.961 while false positives rise from 7 to 20).

## Key limitations

- Small dataset (918 rows) and a single train/test split.
- 79% male, with only 6 female-positive cases in the test set, so subgroup performance can't be reliably assessed.
- About 18.7% of Cholesterol values were imputed, and `Cholesterol_Missing` may partly reflect data provenance rather than physiology.
- No external validation.

## Tech stack

Python, pandas, NumPy, scikit-learn, SciPy, statsmodels, SHAP, Matplotlib, Seaborn.
