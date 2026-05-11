# Model Documentation
## Energy Demand Prediction — Training, Optimization & Results

**Target variable:** `en2025_enegy_demand_present_m2` — present-state energy demand in kWh/m²/year  
**Train/test split:** 80 % train / 20 % test, `random_state=42`, stratified by row order

---

## Table of Contents

1. [Target Transformation](#1-target-transformation)
2. [Train/Test Split & Leakage Prevention](#2-trainttest-split--leakage-prevention)
3. [Models Overview](#3-models-overview)
4. [Random Forest](#4-random-forest-rf)
5. [XGBoost](#5-xgboost-xgb)
6. [MLP (Neural Network)](#6-mlp-neural-network)
7. [TabNet](#7-tabnet)
8. [TabPFN](#8-tabpfn)
9. [LightGBM](#9-lightgbm-lgb)
10. [CatBoost](#10-catboost-cb)
11. [Hyperparameter Search Strategy](#11-hyperparameter-search-strategy)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Results Summary](#13-results-summary)
14. [Feature Analysis & PCA](#14-feature-analysis--pca)

---

## 1. Target Transformation

Raw energy demand values span a very wide range with a heavy right tail. Before training, the target is log-transformed:

```
target_transformed = log(1 + raw_target)   [np.log1p]
```

Outliers are removed using IQR capping (values beyond Q1 − 3×IQR or Q3 + 3×IQR are dropped) before log transformation. Predictions are reversed with `np.expm1` when computing MAE against real kWh/m² values.

This transformation:
- Makes residuals more normally distributed (benefits MLP, TabNet)
- Reduces the influence of extreme values on squared-error loss
- Keeps all models optimising on the same scale

---

## 2. Train/Test Split & Leakage Prevention

All models use a single 80/20 split with `random_state=42`. The split is applied **once** and reused across all model/feature-set combinations, so results are directly comparable.

### Enhanced_3 Leakage Prevention

The 4th feature set (Enhanced_3) includes **target-encoded** geographic features (municipality, oblast, city, building category). These are computed by the `_target_encode` function, which could leak test-set information if applied to the full dataset before splitting.

**Fix implemented:** Before building Enhanced_3 features, `run_all_feature_sets_comparison` pre-computes a boolean `train_mask` using the same split parameters:

```python
_idx_tr, _idx_te = train_test_split(np.arange(n), test_size=0.2, random_state=42)
train_mask = pd.Series(False, index=df_clean.index)
train_mask.iloc[_idx_tr] = True

feat_sets = {
    ...
    'Enhanced_3': create_enhanced_3_features(df_clean, target_column, train_mask=train_mask),
}
```

Inside `_target_encode`, group means are computed only from `df_tmp[train_mask.values]`. Unseen groups receive the global training-set mean. The actual model `train_test_split` call later produces the identical split because it uses the same data size and `random_state`.

---

## 3. Models Overview

| Model | Type | Handles Categoricals Natively | Feature Sets |
|---|---|---|---|
| Random Forest (RF) | Ensemble (bagging) | No — label-encoded | All 4 |
| XGBoost (XGB) | Ensemble (boosting) | No — label-encoded | All 4 |
| MLP | Neural network | No — label-encoded + scaled | Baseline, Enhanced, Enhanced_2 |
| TabNet | Neural network (attention) | No — StandardScaler applied | All 4 |
| TabPFN | Transformer (in-context) | No — StandardScaler applied | All 4 |
| LightGBM (LGB) | Ensemble (boosting) | Yes (native) | Enhanced_3 only |
| CatBoost (CB) | Ensemble (boosting) | Yes (native cat indices) | Enhanced_3 only |

---

## 4. Random Forest (RF)

**Library:** `sklearn.ensemble.RandomForestRegressor`

### Architecture
An ensemble of decision trees, each trained on a bootstrap sample of the training data. Predictions are the mean across all trees. Trees grow without pruning; variance is reduced by averaging and feature subsampling at each split.

### Hyperparameter Search

Optimised with `RandomizedSearchCV` (3-fold CV, 15 iterations, scoring = R²):

| Parameter | Search Space |
|---|---|
| `n_estimators` | 300, 500 |
| `max_depth` | 15, 20, None (unlimited) |
| `min_samples_split` | 2, 5 |
| `max_features` | `"sqrt"`, `"log2"` |

### Input Preparation
Categorical columns are label-encoded. Inf/NaN replaced with column median. No scaling needed (tree-based).

### Best Results (Enhanced_3)
| Metric | Value |
|---|---|
| R² | 0.7124 |
| MAE | 20.99 kWh/m² |

---

## 5. XGBoost (XGB)

**Library:** `xgboost.XGBRegressor`

### Architecture
Gradient-boosted decision trees. Each tree fits the residuals of the previous ensemble using second-order gradient information. Regularisation (L1/L2) and subsampling control overfitting.

### Hyperparameter Search

Optimised with `RandomizedSearchCV` (3-fold CV, 30 iterations, scoring = R²):

| Parameter | Search Space |
|---|---|
| `n_estimators` | 300, 500, 800 |
| `max_depth` | 4, 6, 8 |
| `learning_rate` | 0.03, 0.05, 0.1 |
| `subsample` | 0.8, 0.9, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |

### Input Preparation
Same as RF: label-encoded categoricals, median imputation. No scaling.

### Best Results (Enhanced_3) — Best overall model
| Metric | Value |
|---|---|
| R² | **0.7594** |
| MAE | **15.94 kWh/m²** |

---

## 6. MLP (Neural Network)

**Library:** `sklearn.neural_network.MLPRegressor`

### Architecture
A fully-connected feedforward network. The hidden layer configuration is tuned as a hyperparameter (single or double hidden layer). Uses ReLU-like activations via the `relu` activation parameter, Adam optimiser, and early stopping against a validation fraction.

### Hyperparameter Search

Optimised with `RandomizedSearchCV` (3-fold CV, 20 iterations, scoring = R²):

| Parameter | Search Space |
|---|---|
| `hidden_layer_sizes` | (100,), (200,), (100, 50), (200, 100) |
| `activation` | `"relu"`, `"tanh"` |
| `alpha` (L2 reg) | 0.0001, 0.001, 0.01 |
| `learning_rate_init` | 0.001, 0.01 |

### Input Preparation
Features are scaled with `StandardScaler` (fit on training set, transform applied to test). Categorical columns label-encoded first.

### Feature Sets
Runs on Baseline, Enhanced, Enhanced_2 only. Enhanced_3's high-cardinality target-encoded features do not provide additional benefit to MLP compared to tree-based models.

---

## 7. TabNet

**Library:** `pytorch-tabnet` (`TabNetRegressor`)

### Architecture
A neural network specifically designed for tabular data. Uses a sequential attention mechanism to select which features to use at each step of a multi-step decision process — conceptually similar to how gradient-boosted trees perform feature selection at each split, but in a differentiable, end-to-end trainable form.

Key components:
- **Feature transformer**: shared and step-specific fully-connected layers
- **Attentive transformer**: sparse attention (via sparsemax) selects a subset of features per step
- **Steps**: `n_steps=3` sequential processing steps

### Fixed Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `n_d`, `n_a` | 32 | Embedding width |
| `n_steps` | 3 | Processing steps |
| `gamma` | 1.3 | Feature reuse penalty |
| `n_independent`, `n_shared` | 2, 2 | GLU layer counts |
| `max_epochs` | 100 | With early stopping |
| `patience` | 10 | Early stopping patience |
| `batch_size` | 256 | Mini-batch size |
| `seed` | 42 | Reproducibility |

### Input Preparation
All features scaled with `StandardScaler` (fit on training set). Features cast to `float32`. Targets reshaped to `(-1, 1)`.

### Feature Sets
Runs on all 4 feature sets (Baseline, Enhanced, Enhanced_2, Enhanced_3). Enhanced_3 uses a leakage-free `train_mask` for target-encoded features before the final split is applied.

---

## 8. TabPFN

**Library:** `tabpfn` (`TabPFNRegressor`)

### Architecture
A pre-trained transformer model trained on thousands of synthetic tabular datasets using meta-learning (prior-data fitted networks). It performs in-context learning: the entire training set is passed as context to the model at inference time, with no gradient update. Particularly strong on small datasets (< 10 000 rows).

TabPFN is a **zero-shot** model — no hyperparameter tuning is performed. The pre-trained prior encodes general inductive biases for tabular regression.

### Input Preparation
Same as TabNet: `StandardScaler`, `float32` cast.

### Limitations
- Not tuneable (weights are frozen)
- Performance degrades on very large datasets or high-dimensional features
- Requires the `tabpfn` package to be installed

### Feature Sets
Runs on all 4 feature sets (Baseline, Enhanced, Enhanced_2, Enhanced_3).

---

## 9. LightGBM (LGB)

**Library:** `lightgbm.LGBMRegressor`

### Architecture
Gradient-boosted decision trees using histogram-based leaf-wise (best-first) tree growth instead of the level-wise approach used by XGBoost. This typically gives faster training and better accuracy on tabular data with many features.

Key differences from XGBoost:
- **Leaf-wise growth**: splits the leaf with the largest loss reduction first — more accurate but needs `min_child_samples` to prevent overfitting
- **Histogram binning**: groups continuous values into bins for faster split finding
- **Native categorical support**: directly handles string/integer category columns

### Hyperparameter Search

Optimised with `RandomizedSearchCV` (3-fold CV, 30 iterations, scoring = R²):

| Parameter | Search Space |
|---|---|
| `n_estimators` | 300, 500, 800 |
| `max_depth` | 5, 7, -1 (unlimited) |
| `learning_rate` | 0.03, 0.05, 0.1 |
| `num_leaves` | 31, 63, 127 |
| `subsample` | 0.8, 0.9, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `min_child_samples` | 5, 10, 20 |

### Input Preparation
Label-encoded categoricals (numeric representation). NaN/inf replaced with column median. LightGBM can also accept the raw categorical indices if provided.

### Feature Sets
Enhanced_3 only (the only feature set where LGB is run).

### Best Results (Enhanced_3)
| Metric | Value |
|---|---|
| R² | 0.7528 |
| MAE | 17.41 kWh/m² |

---

## 10. CatBoost (CB)

**Library:** `catboost.CatBoostRegressor`

### Architecture
Gradient-boosted decision trees with a patented approach to handling categorical features called **ordered target statistics** — a form of target encoding that avoids data leakage by using only preceding rows (in a random permutation) to compute statistics, similar to k-fold target encoding.

Key features:
- **Native categoricals**: no pre-encoding needed; pass column indices directly
- **Symmetric trees**: each level uses the same split condition (oblivious trees), making the model faster to evaluate and less prone to overfitting
- **Ordered boosting**: reduces prediction shift (a form of leakage inherent to standard gradient boosting)

### Fixed Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| `iterations` | 500 | Max trees |
| `learning_rate` | 0.05 | Conservative step size |
| `depth` | 6 | Tree depth (symmetric) |
| `l2_leaf_reg` | 3 | L2 regularisation |
| `early_stopping_rounds` | 30 | Uses `eval_set` (test split) |
| `random_seed` | 42 | Reproducibility |

### Input Preparation
Raw feature matrix including string columns. Categorical feature column indices are passed via `cat_features`. Numeric NaN/inf replaced with column median.

### Feature Sets
Enhanced_3 only.

### Best Results (Enhanced_3)
| Metric | Value |
|---|---|
| R² | 0.7583 |
| MAE | 18.29 kWh/m² |

---

## 11. Hyperparameter Search Strategy

Three distinct search strategies are used across models:

| Strategy | Models | Library | CV Folds | Iterations |
|---|---|---|---|---|
| `RandomizedSearchCV` | RF, XGB, MLP, LightGBM | scikit-learn | 3 | 15–30 |
| Fixed hyperparameters | CatBoost, TabPFN | — | — | — |
| Fixed hyperparameters | TabNet | pytorch-tabnet | — | early stopping only |

`RandomizedSearchCV` samples from the parameter grid at random (not exhaustively), which allows a large search space to be covered efficiently. With 3-fold CV and 30 iterations, each run trains 90 models; scoring is R² on the held-out fold.

**Note:** All hyperparameter search is performed on **training data only** (the 80% split). The 20% test set is held out until final evaluation.

---

## 12. Evaluation Metrics

All models are evaluated on the same held-out 20% test set.

| Metric | Formula | Interpretation |
|---|---|---|
| R² (coefficient of determination) | `1 - SS_res / SS_tot` | Fraction of variance explained. 1.0 = perfect; 0.0 = predicts the mean. |
| MAE (mean absolute error) | `mean(|y_true - y_pred|)` in kWh/m² | Average absolute prediction error in real units, after reversing log transformation. |

When the log-transformed target is used, predictions are back-transformed with `np.expm1` before computing MAE to give physically interpretable error magnitudes.

---

## 13. Results Summary

Results are from the most recent run on `overlapping_gdf_dataset` (the spatial join of EPC records and GIS building footprints).

### Enhanced_3 Feature Set — All Models

| Rank | Model | R² | MAE (kWh/m²) |
|---|---|---|---|
| 1 | XGBoost | **0.7594** | **15.94** |
| 2 | CatBoost | 0.7583 | 18.29 |
| 3 | LightGBM | 0.7528 | 17.41 |
| 4 | Random Forest | 0.7124 | 20.99 |

### Best per Model Across All Feature Sets

| Model | Best Feature Set | R² | MAE (kWh/m²) |
|---|---|---|---|
| XGBoost | Enhanced_3 | 0.7594 | 15.94 |
| CatBoost | Enhanced_3 | 0.7583 | 18.29 |
| LightGBM | Enhanced_3 | 0.7528 | 17.41 |
| RF | Enhanced_3 | 0.7124 | 20.99 |
| XGBoost | Enhanced_2 | 0.4953 | — |

Feature engineering has a much larger impact than model choice: moving from Enhanced_2 to Enhanced_3 improves XGB R² from 0.4953 to 0.7594 — a **+26.4 pp** gain driven primarily by the location target-encoding features.

---

## 14. Feature Analysis & PCA

A dedicated analysis section at the end of the notebook provides three views of the feature space:

### Feature Importance (LightGBM)
`plot_feature_importance()` trains LightGBM on Enhanced_3 features using the training split only and plots the top-N split-count importances. This reveals which features the model relies on most — typically the location target-encoding features (`mun_mean_energy`, `oblast_mean_energy`) rank highly.

### PCA Decomposition
`plot_pca_feature_space()` standardises Enhanced_3 features and fits a PCA. Two panels are shown:
- **Explained variance**: bar + cumulative line per principal component
- **PC1 vs PC2 scatter**: each point is a building, coloured by log energy demand — shows how well the first two components separate high vs low energy buildings

### Building Subtype Clustering
`plot_building_subtypes()` projects buildings into the first 5 PCA components and applies KMeans (default: 5 clusters). Outputs:
- Scatter of clusters in PC1/PC2 space
- Boxplot of energy demand per cluster (sorted by median)

All three analyses use the same leakage-free `train_mask` for any target-encoding calls inside `create_enhanced_3_features`.
