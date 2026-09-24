# Heart Disease Prediction — A Systematic Study of Class Imbalance and Data Signal Quality

## Overview

This project applies a rigorous, end-to-end machine learning pipeline to predict heart disease
risk from the Kaggle dataset [`oktayrdeki/heart-disease`](https://www.kaggle.com/datasets/oktayrdeki/heart-disease).
Rather than reporting a single "best accuracy" number, the project is designed as a **systematic
experiment**: 8 model architectures × 2 train/test splits × 9 class-imbalance handling techniques
(144+ configurations), evaluated with Accuracy, Precision, Recall, F1, and ROC-AUC.

**Key finding:** across every configuration tested, model performance never meaningfully exceeds
random guessing (mean ROC-AUC = 0.498, best single result = 0.550). This is traced — through
correlation, mutual information, and feature-importance analysis — to a fundamental lack of
predictive signal in the dataset's features relative to the target, rather than any weakness in
modeling approach. The project's value lies in the rigor of *diagnosing* this, not in a headline
accuracy figure.

## Dataset

- **Source:** Kaggle, `oktayrdeki/heart-disease` (synthetic health-risk dataset)
- **Size:** ~10,000 records, 21 features (demographic, lifestyle, and clinical measurements)
- **Target:** `Heart Disease Status` (binary: Yes / No)
- **Class balance:** ~80% No / ~20% Yes — meaningful class imbalance

## Methodology

| Stage | What was done |
|---|---|
| **1. Data Acquisition** | Loaded via `kagglehub`; explored shape, dtypes, distributions, and class balance |
| **2. Preprocessing** | Imputed missing values (mean for numeric, mode for categorical); encoded categorical features; split into **two parallel configurations (80:20 and 90:10)** to test whether split ratio affects outcome; scaled features with `StandardScaler` (fit on train only) |
| **3. Feature Selection** | Recursive Feature Elimination (RFE) compared across 4 estimators (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) via `RepeatedStratifiedKFold`; top-10 features selected separately by LR (`rfe_lr_features.pkl`) and RF (`rfe_rf_features.pkl`) for cross-comparison |
| **4. Model Training** | 8 architectures × 2 splits × 3 feature sets (all features / FS_LR / FS_RF): Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP, DNN, ResNet, TabNet |
| **5. Imbalance Handling** | Each model tested against 10 techniques: no sampling (baseline), SMOTE, ADASYN, RandomUnderSampler, NearMiss, ClusterCentroids, SMOTETomek, SMOTEENN, ADASYN+Tomek, BorderlineSMOTE+Tomek — using `StratifiedKFold` (5-fold), with the best fold selected for final test-set evaluation |
| **6. Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, and full confusion matrix for every model × split × technique combination |
| **7. Diagnostic Analysis** | Correlation and mutual information between every feature and the target, to test whether poor results were a modeling problem or a data problem |

## Results Summary

### Baseline performance (no imbalance handling) — headline accuracy is misleading

| Model | Split | Test Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80:20 | 0.800 | 0.469 |
| Random Forest | 90:10 | 0.800 | 0.519 |
| XGBoost | 80:20 | 0.800 | 0.472 |
| DNN | 90:10 | 0.800 | 0.516 |

Every baseline model lands at ~80% accuracy — but this is achieved by **predicting "No Disease"
for every single case**, i.e., simply matching the majority class. Recall for the positive class
is 0.000 across the board at baseline. Accuracy alone is not a trustworthy metric here.

### Best result per model (by Recall, after imbalance handling)

| Model | Split | Technique | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 80:20 | SMOTEENN | 0.200 | 0.200 | 1.000 | 0.333 | 0.471 |
| Random Forest | 80:20 | ClusterCentroids | 0.200 | 0.200 | 0.998 | 0.333 | 0.487 |
| XGBoost | 90:10 | NearMiss | 0.401 | 0.210 | 0.725 | 0.326 | **0.549** |
| Decision Tree | 80:20 | ClusterCentroids | 0.277 | 0.204 | 0.900 | 0.332 | 0.522 |

Aggressive undersampling techniques can push Recall toward 1.0, but only by pushing Precision and
Accuracy down toward ~0.20 (the base rate of the positive class) — i.e., the model degenerates
toward guessing "Disease" for nearly everyone. This is the imbalance-handling trade-off working
exactly as expected; it does **not** indicate the model has learned a meaningful pattern.

### The diagnostic result that matters most

> **Across all 158 model × split × technique combinations tested, mean ROC-AUC = 0.498
> (std ≈ 0.03), and the single best result across every configuration is 0.550.**
> An ROC-AUC of 0.50 is equivalent to random guessing.

This was cross-validated three ways:
- **Correlation** of every feature with the target: all values fall in **-0.02 to +0.02**
- **Mutual Information** (captures non-linear relationships too): similarly near-zero for every feature
- **Feature importance** (Random Forest): no feature stands out; importance is spread near-uniformly

All three point to the same conclusion: **the dataset's features carry almost no predictive signal
for the target**, most likely because it is a synthetically generated dataset rather than real
clinical data with genuine physiological relationships.

Full results for all configurations: [`master_results.csv`](05_results/master_results.csv) · feature-selection results: [`fs_results_combined.csv`](05_results/fs_results_combined.csv)

## What This Project Demonstrates

The headline accuracy numbers are not the point of this project — the **methodology and diagnostic
reasoning** are:

- **Systematic experimentation:** 8 architectures × 2 split ratios × 9 resampling techniques,
  evaluated consistently rather than cherry-picking a single favorable run
- **Correct handling of imbalanced data:** recognizing that 80% baseline accuracy was misleading,
  and testing multiple resampling strategies rather than accepting the first result
- **Root-cause diagnosis over metric-chasing:** rather than continuing to tune models against a
  ceiling that resampling techniques could not break, the project pivoted to testing *why* — using
  correlation, mutual information, and feature importance as three independent lines of evidence
- **Honest reporting:** presenting a null/negative result clearly and explaining it, rather than
  overstating what the models achieved

## Repository Structure

```
heart-disease-imbalance-study/
├── README.md                          ← this file
├── requirements.txt
├── 01_data_acquisition/
│   └── Step1_Data_Acquisition.ipynb
├── 02_preprocessing/
│   └── Step2_Data_Preprocessing.ipynb  (80:20 and 90:10 splits)
├── 03_feature_selection/
│   ├── Step3_RFE.ipynb                 (RFE with Random Forest only)
│   └── step3_FS_LR.ipynb               (RFE comparison: LR, CART, RF, GBM → saves rfe_lr_features.pkl & rfe_rf_features.pkl)
├── 04_model_training/
│   ├── Step4_Train_Log_{80,90}.ipynb
│   ├── Step4_Train_DECISION_Tree_{80,90}.ipynb
│   ├── Step4_Train_Random_{80,90}.ipynb
│   ├── Step4_Train_XGBoost_{80,90}.ipynb
│   ├── Step4_Train_MLP_{80,90}.ipynb
│   ├── Step4_Train_DNN_{80,90}.ipynb
│   ├── Step4_Train_ResNet_{80,90}.ipynb
│   ├── Step4_Train_TabNet_{80,90}.ipynb
│   ├── Step4_Train_Log_90_FS_{LR,RF}.ipynb          ← feature-selected variants
│   ├── Step4_Train_DECISION_Tree_90_FS_{LR,RF}.ipynb
│   ├── Step4_Train_Random_90_FS_{LR,RF}.ipynb
│   ├── Step4_Train_XGBoost_90_FS_{LR,RF}.ipynb
│   ├── Step4_Train_MLP_90_FS_{LR,RF}.ipynb
│   ├── Step4_Train_DNN_90_FS_{LR,RF}.ipynb
│   ├── Step4_Train_ResNet_90_FS_{LR,RF}.ipynb
│   └── Step4_Train_TabNet_90_FS_{LR,RF}.ipynb
└── 05_results/
    ├── master_results.csv              ← all experiment results (80:20 and 90:10, no FS)
    ├── fs_results_combined.csv         ← feature selection results (FS_LR and FS_RF, 160 rows)
    ├── Full_Model_Comparison.csv       ← combined comparison across all splits
    ├── Sorted_Model_Comparison.csv
    ├── best_by_recall.csv
    ├── best_by_f1.csv
    ├── best_by_auc.csv
    └── summay.ipynb                    ← aggregation and comparison notebook
```

## Reproducing This Work

1. Download the dataset via `kagglehub.dataset_download("oktayrdeki/heart-disease")`
2. Run notebooks in numeric order (01 → 05)
3. Run `03_feature_selection/step3_FS_LR.ipynb` to generate feature selection pkl files, then copy `models/` into `04_model_training/`
4. Each `Step4_Train_*` notebook is self-contained per model; run all 8 (×2 splits) for baseline results, plus 16 FS variants (`_FS_LR` / `_FS_RF`) for feature-selected results
5. Run `05_results/summay.ipynb` to aggregate all results

## Limitations & Honest Caveats

- This is a **synthetic** dataset; conclusions about feature-target relationships apply to this
  dataset only and should not be read as claims about real-world heart disease risk factors
- The weak signal found here means the trained models should **not** be used for any real
  screening or diagnostic purpose
- Deep learning models (DNN, ResNet, TabNet) were included to test whether non-linear
  architectures could capture patterns invisible to linear/tree models — they could not, which is
  itself informative given the mutual information result

## Author's Note

This project intentionally documents a negative result. In real-world data science, correctly
diagnosing that a modeling ceiling is caused by data quality — rather than continuing to tune
hyperparameters indefinitely — is a core skill, and one this project was built to demonstrate.
