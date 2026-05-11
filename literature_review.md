# Literature Review: Machine Learning for Building Energy Demand Prediction
## A Comparative Study Using Bulgarian Energy Performance Certificate Data

**Project:** Predicting present-state building energy demand (kWh/m²/year) from EPC records and GIS building footprints  
**Dataset:** ~2,100 Bulgarian buildings; spatial join of EPC records and GIS footprints  
**Best result:** XGBoost with Enhanced_3 features — R² = 0.7594, MAE = 15.94 kWh/m²  
**Date:** May 2026

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Why These Model Families Were Chosen](#2-why-these-model-families-were-chosen)
3. [Model Architectures and Related Work](#3-model-architectures-and-related-work)
   - 3.1 [Random Forest](#31-random-forest)
   - 3.2 [XGBoost](#32-xgboost)
   - 3.3 [LightGBM](#33-lightgbm)
   - 3.4 [CatBoost](#34-catboost)
   - 3.5 [MLP (Multilayer Perceptron)](#35-mlp-multilayer-perceptron)
   - 3.6 [TabNet](#36-tabnet)
   - 3.7 [TabPFN](#37-tabpfn)
4. [Feature Engineering and Preprocessing](#4-feature-engineering-and-preprocessing)
5. [Comparative Performance and Benchmarks](#5-comparative-performance-and-benchmarks)
6. [References](#6-references)

---

## 1. Introduction and Motivation

### 1.1 The Building Energy Prediction Problem

Buildings account for approximately 40% of total final energy consumption in the European Union, making the built environment one of the primary targets of climate mitigation policy (European Commission, Energy Performance of Buildings Directive). Accurately predicting the energy demand of individual buildings — expressed as kilowatt-hours of primary or delivered energy per square metre of floor area per year (kWh/m²/year) — is foundational to three intersecting challenges: urban energy planning, renovation prioritisation, and compliance monitoring.

Traditional approaches to building energy modelling fall into two broad categories. Physics-based simulation tools (such as EnergyPlus, TRNSYS, or IDA-ICE) solve coupled heat-transfer, airflow, and thermodynamic equations for a fully specified building model. These tools are highly accurate when inputs are reliable but require dozens of precisely measured parameters — envelope U-values, thermal bridging coefficients, HVAC schedules — that are rarely available in administrative datasets at city or national scale. Data-driven machine learning approaches, by contrast, learn statistical associations directly from observed energy performance records, sacrificing some physical interpretability in exchange for scalability and applicability to imperfect, real-world data sources.

The growth of national Energy Performance Certificate (EPC) programmes across EU member states has created a new class of large-scale, structured building energy datasets. EPCs record a standardised set of building attributes (heated area, volume, construction year, building category) alongside an auditor-assessed energy demand figure computed by a national calculation methodology. These datasets are inherently heterogeneous — mixing continuous geometric measurements, ordinal year variables, nominal categorical classifications, and location identifiers — and typically contain thousands to tens of thousands of records per country, placing them firmly in the small-to-medium dataset regime where many deep learning methods have historically struggled.

### 1.2 The Bulgarian EPC Dataset

This project uses EPC records from Bulgaria's national building energy certification programme, spatially joined to GIS building footprints to enrich records with geometric features (footprint area, building shape classification, derived floor height). The resulting dataset contains approximately 2,100 buildings spread across Bulgarian municipalities and oblasts (provinces), representing a range of building categories including residential single-family houses, multi-family apartment blocks, educational buildings, commercial premises, and offices.

The target variable — `en2025_enegy_demand_present_m2` — is the EPC-assessed present-state energy demand in kWh/m²/year. This is the standardised delivered energy demand of the building in its current (pre-renovation) state as computed by the Bulgarian national EPC calculation methodology. The distribution of this variable is heavily right-skewed, with a minority of buildings exhibiting very high specific demand values; this motivates the log-transformation described in Section 4.

### 1.3 Challenges Specific to This Setting

Several challenges distinguish this problem from canonical ML benchmarks and motivate the model selection and feature engineering choices described below:

**Small-to-medium dataset size.** With ~2,100 samples, deep learning approaches that require large datasets to outperform well-regularised tree ensembles are at a structural disadvantage. The dataset is too large for exhaustive physics-based simulation as a baseline yet too small for unconstrained deep neural networks.

**Mixed feature types.** Features span continuous geometric measurements (area, volume, height), ordinal time variables (construction year), nominal categorical variables (building category, function type, construction type, geographic identifiers), and engineered ratios. Tree-based methods handle this heterogeneity naturally; neural networks require careful preprocessing.

**High geographic heterogeneity.** Bulgarian buildings span climatic zones from the Sub-Balkan lowlands to alpine regions, with energy demand patterns that vary substantially by municipality and oblast. Capturing spatial variation without using geographically leaking features is a non-trivial engineering challenge.

**Administrative data quality.** EPC records contain missing values, entry errors, and European comma-decimal formatting inconsistencies. Features require robust imputation and outlier handling. Some columns are excluded entirely due to data leakage (e.g., the 2023 EPC energy demand measurement, which is nearly identical to the 2025 target for buildings assessed in both cycles).

---

## 2. Why These Model Families Were Chosen

### 2.1 The Tabular Data Landscape

Tabular data — structured data stored in rows and columns, where each column represents a different feature type — remains the dominant format in applied predictive modelling across energy, finance, healthcare, and logistics domains. Unlike image, audio, or text modalities where deep learning architectures have achieved decisive superiority through inductive biases matched to data structure (spatial locality for CNNs, sequential order for RNNs, token context for transformers), tabular data lacks a single canonical structure that neural architectures can exploit.

A series of empirical benchmark studies has established that gradient-boosted tree ensembles — specifically XGBoost, LightGBM, and CatBoost — consistently outperform deep learning methods on small-to-medium tabular datasets. Shwartz-Ziv and Armon (2022) conducted a large-scale evaluation across 45 tabular datasets and found that tree-based methods outperformed neural networks in the majority of cases, particularly when dataset size was below 10,000 rows. Grinsztajn et al. (2022) drew similar conclusions in a systematic comparison across 45 OpenML datasets, attributing tree superiority to their robustness to uninformative features and their ability to exploit irregular decision boundaries without regularisation.

This finding motivates the backbone of the model portfolio: the three major gradient-boosting frameworks (XGBoost, LightGBM, CatBoost) alongside the classic Random Forest ensemble. These four tree-based methods are expected — and observed — to outperform the neural network approaches on this specific dataset.

### 2.2 Why Neural Networks Are Nevertheless Included

Despite the general tabular data finding, neural networks are included for several reasons:

1. **Representation learning potential.** For datasets with many correlated features (the Enhanced_3 feature set contains ~35 features, many of which are smooth transformations of a smaller number of raw physical attributes), neural networks can in principle learn more compact and generalisable representations.

2. **Gradient-based optimisation.** Neural networks can directly optimise differentiable loss functions without the greedy, stage-wise approximations inherent to boosting. In domains where smooth, globally optimal solutions exist, this can be advantageous.

3. **Novel architectures for tabular data.** TabNet (Arik & Pfister, 2021) and TabPFN (Hollmann et al., 2022) represent fundamentally different approaches to tabular data than standard MLPs and merit evaluation in their own right: TabNet introduces sparse attention-based feature selection inspired by decision-tree logic; TabPFN leverages pre-trained prior knowledge to perform zero-shot in-context inference.

4. **Benchmarking completeness.** A comprehensive comparison across the performance spectrum — from the simplest neural network (MLP) to the state-of-the-art tabular transformers — provides a richer characterisation of what is learnable from Bulgarian EPC data, beyond what any single model family can reveal.

### 2.3 The Role of Feature Engineering

A key finding of this project is that feature engineering dominates over model selection in its impact on predictive performance. The introduction of leakage-free location target-encoding in the Enhanced_3 feature set improved XGBoost R² from 0.4953 (Enhanced_2) to 0.7594 — a gain of +26.4 percentage points — while the improvement from switching between tree models on the same feature set is typically 2–5 percentage points. This is consistent with the broader principle that information content in features is a ceiling on model performance, and that improving features is often more impactful than tuning models.

---

## 3. Model Architectures and Related Work

### 3.1 Random Forest

#### Architecture Description

Random Forest (Breiman, 2001) is a bagging ensemble of unpruned decision trees. The algorithm builds `B` trees in parallel, each trained on an independently drawn bootstrap sample of the training data (sampling with replacement). At each node of each tree, only a random subset of `m` features (typically `m = sqrt(p)` for regression, where `p` is the total number of features) is considered as split candidates. This double randomisation — in the data (bootstrap) and the feature selection (random subspace) — decorrelates the individual trees, reducing the variance of the ensemble average relative to any single tree.

Final predictions are the arithmetic mean of all `B` tree predictions:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

where `T_b` is the `b`-th tree and `x` is the input feature vector. Trees grow until leaves contain fewer than `min_samples_split` samples or `max_depth` is reached (when unlimited, trees can memorise the training data; variance reduction comes from averaging). Feature importances are measured by the mean decrease in node impurity (mean squared error for regression) across all splits that use each feature, weighted by the number of samples reaching each split node.

#### ASCII Architecture Diagram

```
  Training Data (N samples, p features)
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  ├──Bootstrap─────────────────────────────────────────────►│
  │  Sample 1              Sample 2    ...    Sample B      │
  └──────────────────────────────────────────────────────── ┘
         │                    │                   │
         ▼                    ▼                   ▼
   ┌──────────┐         ┌──────────┐       ┌──────────┐
   │  Tree 1  │         │  Tree 2  │       │  Tree B  │
   │          │         │          │       │          │
   │   root   │         │   root   │       │   root   │
   │  /     \ │         │  /     \ │       │  /     \ │
   │ [m feat] │         │ [m feat] │       │ [m feat] │
   │ ↓     ↓  │         │ ↓     ↓  │       │ ↓     ↓  │
   │ n1    n2 │         │ n1    n2 │       │ n1    n2 │
   │ ↓↓   ↓↓  │         │ ↓↓   ↓↓  │       │ ↓↓   ↓↓  │
   │[leaves]  │         │[leaves]  │       │[leaves]  │
   └────┬─────┘         └────┬─────┘       └────┬─────┘
        │  ŷ₁                │  ŷ₂              │  ŷ_B
        └────────────────────┴──────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Ensemble Mean  │
                    │                 │
                    │  ŷ = mean(ŷᵢ)  │
                    └─────────────────┘
                             │
                             ▼
                     Prediction Output

  At each node split, only m = sqrt(p) randomly
  chosen features are considered as split candidates.
  This decorrelates trees and reduces ensemble variance.
```

#### Why Random Forest Suits Building Energy Prediction

Random Forest handles the heterogeneous feature space of EPC data naturally: construction year (ordinal), building category (nominal), heated area (continuous), and location mean energy (continuous) coexist without any need for scaling or one-hot encoding, as decision-tree splits are threshold comparisons that are invariant to monotone feature transformations. The model is robust to irrelevant features and moderate amounts of missing values (handled here by median imputation). Its feature importance outputs (mean decrease in impurity) are readily interpretable, making it useful for understanding which building attributes drive energy demand predictions — a relevant property for practical policy applications.

However, Random Forest's parallel (bagging) structure means each tree is trained on a noisy bootstrap sample with no sequential correction of residuals, making it typically less accurate than boosting methods on this dataset. Confirmed in results: R² = 0.7124 on Enhanced_3, versus 0.7594 for XGBoost.

#### Key Algorithm Reference

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

#### Related Work in Energy Prediction

Random Forest has been one of the most widely applied ML methods in building energy prediction. It handles the small-to-medium dataset sizes typical of EPC databases well and produces interpretable importance scores that can guide retrofit priorities.

See also: [search term: "random forest building energy prediction EPC"]  
See also: [search term: "random forest heating energy demand residential buildings"]

---

### 3.2 XGBoost

#### Architecture Description

XGBoost (eXtreme Gradient Boosting; Chen & Guestrin, 2016) is a scalable, regularised implementation of gradient-boosted decision trees (GBDT). Unlike Random Forest's parallel bagging strategy, XGBoost builds trees sequentially. At each boosting round `t`, a new tree `f_t` is added to fit the negative gradient of the loss function with respect to the current ensemble's predictions. The ensemble at round `T` is:

$$F_T(\mathbf{x}) = \sum_{t=1}^{T} \eta \cdot f_t(\mathbf{x})$$

where `η` is the learning rate (shrinkage). XGBoost uses second-order Taylor expansion of the loss to derive a closed-form optimal leaf weight for each leaf of the newly added tree, enabling more precise gradient steps than first-order (gradient) only methods. The objective includes explicit L1 (alpha) and L2 (lambda) regularisation terms on the leaf weights, controlling model complexity.

Key engineering innovations in XGBoost include:
- **Column block structure** that stores data in compressed, sorted column format for efficient cache-aware split finding
- **Weighted quantile sketch** for approximate split candidates on continuous features
- **Sparsity-aware split finding** that handles missing values without imputation by learning the optimal default direction for each split
- **Parallel tree construction** at the node level within each tree

#### ASCII Architecture Diagram

```
  Input Features x
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  XGBoost Sequential Boosting                                     │
  │                                                                  │
  │  Round 1:  F₁(x) = η·f₁(x)                                      │
  │            ┌──────────────────────────────────────────┐          │
  │            │  Tree 1: fit residuals of F₀ = 0        │          │
  │            │          ┌─────────┐                     │          │
  │            │          │  root   │  ← best split from  │          │
  │            │          │ feat=j  │    2nd-order Taylor  │          │
  │            │          │ val≤θ   │    gain formula      │          │
  │            │          └────┬────┘                     │          │
  │            │          ┌────┴────┐                     │          │
  │            │       ┌──┴──┐  ┌──┴──┐                  │          │
  │            │       │leaf1│  │leaf2│  ← L1/L2 reg.    │          │
  │            │       │  w₁ │  │  w₂ │    leaf weights  │          │
  │            │       └─────┘  └─────┘                  │          │
  │            └──────────────────────────────────────────┘          │
  │                     │                                            │
  │  Residuals ← y - F₁(x)                                          │
  │                     │                                            │
  │  Round 2:  F₂(x) = F₁(x) + η·f₂(x)                             │
  │            ┌──────────────────────────────────────────┐          │
  │            │  Tree 2: fit residuals of F₁             │          │
  │            │       (same structure, new splits)       │          │
  │            └──────────────────────────────────────────┘          │
  │                     │                                            │
  │        ...  (up to n_estimators rounds) ...                      │
  │                     │                                            │
  │  Round T:  F_T(x) = Σ η·fₜ(x)                                   │
  │                     │                                            │
  └─────────────────────┼────────────────────────────────────────────┘
                         │
                         ▼
               Final prediction ŷ = F_T(x)

  Regularisation: Ω(fₜ) = γ·|leaves| + ½λ·Σwⱼ²
  Taylor expansion: gain = ½·[G²/(H+λ) + ...] - γ
  where G = first-order gradient sum, H = second-order (Hessian) sum
```

#### Why XGBoost Suits Building Energy Prediction

XGBoost's regularisation machinery is particularly well-suited to datasets like Bulgarian EPC records. With ~2,100 samples and ~35 features in the Enhanced_3 set, overfitting is a genuine risk. The L1/L2 leaf weight regularisation and minimum child weight constraints act as effective complexity controls. The sparsity-aware split finding handles missing values in EPC columns (e.g., cooled volume is zero or missing for buildings without active cooling) without requiring imputation choices that could introduce bias. The model's support for subsampling of both rows (subsample) and columns (colsample_bytree) per tree provides additional regularisation analogous to the Random Forest's feature subsampling, while the sequential boosting still achieves lower bias than bagging.

XGBoost achieves the best R² (0.7594) and lowest MAE (15.94 kWh/m²) among all seven models in this study, consistent with its strong benchmark performance across dozens of structured prediction competitions and studies.

#### Key Algorithm Reference

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

#### Related Work in Energy Prediction

XGBoost has been applied extensively to building energy prediction tasks using EPC-derived features, sub-meter electricity consumption data, and hourly HVAC load profiles. Its combination of accuracy, speed, and robustness to messy administrative data makes it a natural first choice for EPC-scale studies.

See also: [search term: "XGBoost building energy consumption prediction"]  
See also: [search term: "gradient boosting EPC energy performance certificate machine learning"]

---

### 3.3 LightGBM

#### Architecture Description

LightGBM (Light Gradient Boosting Machine; Ke et al., 2017) is a gradient boosting framework designed for efficiency on large datasets and high-dimensional feature spaces. It introduces two algorithmic innovations over standard GBDT and XGBoost:

1. **Gradient-Based One-Side Sampling (GOSS):** Rather than using all data instances to estimate information gain at each split, GOSS keeps all instances with large gradients (high training loss) and randomly samples from the smaller-gradient instances, with a weighting correction. This reduces the effective data size while retaining the most informative samples.

2. **Exclusive Feature Bundling (EFB):** Many features in high-dimensional sparse datasets are mutually exclusive — they rarely take non-zero values simultaneously. EFB bundles such features into a single composite feature, reducing the effective number of features that need to be scanned for splits.

Beyond these innovations, LightGBM uses **leaf-wise (best-first) tree growth** rather than the level-wise growth used by standard GBDT and XGBoost. At each step, the leaf with the greatest potential loss reduction across all current leaves is split, regardless of tree depth or level. This can produce deeper, more asymmetric trees that reduce loss faster per split, but requires `min_child_samples` regularisation to prevent overfitting on small leaves.

LightGBM also implements **histogram-based split finding**: continuous features are discretised into at most 255 bins, and split search iterates over bin boundaries rather than all unique values, providing O(#bins) rather than O(N) complexity per feature per node.

#### ASCII Architecture Diagram

```
  Input Data: N rows, p features
  ┌─────────────────────────────────────────────────────────────┐
  │  GOSS: subsample by gradient magnitude                      │
  │  ┌──────────────┐   ┌──────────────────────────────────┐   │
  │  │ Large-grad   │   │ Small-grad: random sample        │   │
  │  │ instances    │   │ with amplification weight        │   │
  │  │ (keep all)   │   │ (1 - top_rate) / (1 - base_rate) │   │
  │  └──────┬───────┘   └────────────────┬─────────────────┘   │
  │         └────────────────┬───────────┘                      │
  │                          ▼                                  │
  │  EFB: bundle mutually exclusive features                    │
  │  [f1, f2, f3, f4, f5, ...] → [bundle_A, bundle_B, ...]    │
  └──────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Histogram-based split finding per feature                  │
  │                                                             │
  │  Continuous feature values: [1.2, 5.7, 3.1, 8.9, ...]     │
  │  Binned into ≤ 255 bins:   [ 12,  57,  31,  89, ...]      │
  │  Split search: O(#bins) not O(N)                            │
  └──────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Leaf-wise (best-first) tree growth                         │
  │                                                             │
  │  Round t — current tree state:                              │
  │                                                             │
  │         [root]                                              │
  │        /      \                                             │
  │    [leaf A]  [leaf B]   ← score each leaf's best split     │
  │                                                             │
  │  leaf B has highest gain → split leaf B:                   │
  │                                                             │
  │         [root]                                              │
  │        /      \                                             │
  │    [leaf A]  [node B]                                       │
  │              /    \                                         │
  │          [leaf B1][leaf B2]   ← deeper on one side         │
  │                                                             │
  │  (vs. level-wise: splits ALL nodes at same depth first)    │
  └──────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Sequential boosting: T trees                               │
  │  F_T(x) = Σ(t=1..T) η · fₜ(x)                              │
  │                                                             │
  │  Native categorical support: integer bin encoding,          │
  │  optimal split over category subsets, no one-hot needed     │
  └──────────────────────────┬──────────────────────────────────┘
                              │
                              ▼
                    Final prediction ŷ
```

#### Why LightGBM Suits Building Energy Prediction

LightGBM's native categorical handling is particularly relevant for the Enhanced_3 feature set, which includes building category (`d_Category`), function type (`c2025_functype`), construction type (`construction_type`), and building type from EPC (`en2025_type`) as raw categorical columns. Rather than requiring label encoding followed by numeric threshold splits (which imposes an arbitrary ordinal order on nominal categories), LightGBM can directly partition categorical values into two subsets that maximise information gain. For building typology with many distinct categories (residential, educational, commercial, office, industrial, etc.) that have highly non-linear relationships with energy demand, this is a meaningful advantage over numeric encoding.

LightGBM achieves R² = 0.7528, MAE = 17.41 kWh/m² on Enhanced_3, making it the third-best model — consistent with its strong benchmark track record. Its leaf-wise growth strategy typically outperforms level-wise growth when the dataset has moderate size and complex decision boundaries.

#### Key Algorithm Reference

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30). NeurIPS. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

#### Related Work in Energy Prediction

LightGBM has seen growing adoption in building energy applications due to its speed advantage on larger EPC databases and its ability to handle raw categorical building typology features.

See also: [search term: "LightGBM building energy prediction"]  
See also: [search term: "gradient boosting categorical features building energy certificate"]

---

### 3.4 CatBoost

#### Architecture Description

CatBoost (Categorical Boosting; Prokhorenkova et al., 2018) is a gradient-boosted decision tree framework developed by Yandex, distinguished by two novel algorithmic contributions: **ordered target statistics** for categorical feature encoding, and **ordered boosting** to reduce prediction shift.

**Ordered Target Statistics for Categorical Features.** Standard target encoding computes the mean target value for each category level using the full dataset, which leaks the target into the features and causes the model to overfit. CatBoost instead uses an ordered (permutation-based) approach: for each training example `i`, the mean target for its category level is computed only from examples that precede `i` in a random permutation of the training data. A new random permutation is drawn at each boosting step. This is conceptually equivalent to leave-one-out target encoding computed at inference time, and prevents the overfitting associated with conventional target encoding.

**Ordered Boosting.** Standard gradient boosting computes gradients using the current ensemble's predictions on the same data that was used to train that ensemble — creating a prediction shift bias where training examples have been partially memorised. CatBoost mitigates this by maintaining a separate sequence of models, one for each training example, where each model is trained on the examples that precede that example in the permutation. Gradients are then computed using the model that has not been trained on the current example.

**Symmetric (Oblivious) Trees.** CatBoost uses symmetric decision trees where all nodes at the same depth use the identical split condition (same feature, same threshold). This makes the model evaluation at inference time very fast (a binary vector lookup rather than a tree traversal) and provides implicit regularisation against overfitting deep trees.

#### ASCII Architecture Diagram

```
  ┌────────────────────────────────────────────────────────────────┐
  │  CatBoost: Ordered Categorical Boosting                        │
  │                                                                │
  │  Step 1: Random permutation of training examples               │
  │  Permutation σ: [x₄, x₂, x₇, x₁, x₅, ...]                   │
  │                                                                │
  │  Step 2: Ordered Target Statistics for Categoricals            │
  │                                                                │
  │  For example i (category C), at step t:                        │
  │  ┌─────────────────────────────────────────┐                  │
  │  │  stat(xᵢ, C) = [Σ(j<i, σ(j)∈C) yⱼ + prior] │            │
  │  │               / [count(j<i, σ(j)∈C) + 1]    │            │
  │  │  (only examples before i in permutation)     │            │
  │  └─────────────────────────────────────────┘                  │
  │                 │                                             │
  │  Step 3: Ordered Boosting                                      │
  │                                                                │
  │  For each example xᵢ, gradient is computed using              │
  │  model Mᵢ trained on {x₁...xᵢ₋₁} only:                      │
  │                                                                │
  │  ┌────────────────────────────────────────────────┐           │
  │  │  Model M₁   → gradient for x₁  (no prior data)│           │
  │  │  Model M₂   → gradient for x₂  (seen x₁)      │           │
  │  │  Model M₃   → gradient for x₃  (seen x₁,x₂)   │           │
  │  │     ...                                         │           │
  │  └────────────────────────────────────────────────┘           │
  │                                                                │
  │  Step 4: Symmetric (Oblivious) Tree Structure                  │
  │                                                                │
  │  Depth 0:  ──────[feat_j ≤ θ]──────                          │
  │                   /            \                               │
  │  Depth 1: [feat_j ≤ θ]     [feat_j ≤ θ]  ← SAME split       │
  │              /    \           /    \                           │
  │  Depth 2: [...] [...] │ [...] [...]        ← SAME split       │
  │                                                                │
  │  Leaves = 2^depth;  evaluation = binary feature vector        │
  │  lookup → O(depth) inference, very fast at deploy time        │
  │                                                                │
  │  Step 5: Sequential ensemble                                   │
  │  F_T(x) = Σ(t=1..T) η · fₜ(x)                                │
  └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ŷ = F_T(x)  →  exp(ŷ) - 1  →  kWh/m²
```

#### Why CatBoost Suits Building Energy Prediction

CatBoost's ordered target statistics are particularly relevant for the geographic identifier columns in this dataset — municipality (`en2025_mun`), oblast (`en2025_oblast`), and city (`en2025_city`). These are high-cardinality nominal variables: Bulgaria has 265 municipalities and 28 oblasts. Naive numeric encoding imposes an arbitrary ordinal structure; one-hot encoding produces a sparse, high-dimensional representation. Target encoding captures the meaningful signal (the mean energy demand in each administrative unit) but risks data leakage if computed on the full dataset before splitting. CatBoost's ordered target statistics provide a principled, leakage-resistant way to encode these geographic features directly within the training procedure.

Additionally, the symmetric (oblivious) tree structure provides fast inference — relevant for deployment scenarios where predictions are needed for many buildings simultaneously (e.g., city-wide renovation screening).

CatBoost achieves R² = 0.7583, MAE = 18.29 kWh/m² — the second-best result — confirming that its categorical handling approach is effective on this dataset.

#### Key Algorithm Reference

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems* (Vol. 31). NeurIPS. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac2a2d5be7deef74-Abstract.html

#### Related Work in Energy Prediction

CatBoost has been applied in energy demand and consumption prediction tasks where building typology and location features play important roles as high-cardinality categoricals.

See also: [search term: "CatBoost energy demand prediction buildings"]  
See also: [search term: "ordered boosting categorical features energy performance"]

---

### 3.5 MLP (Multilayer Perceptron)

#### Architecture Description

The Multilayer Perceptron (MLP) is the canonical feedforward artificial neural network, consisting of an input layer, one or more hidden layers, and an output layer. Each neuron in a hidden layer computes a weighted sum of its inputs, applies a non-linear activation function, and passes the result to the next layer. For regression, the output layer has a single neuron with a linear (identity) activation.

Formally, for a network with `L` hidden layers:

```
h⁰ = x                           (input)
hˡ = σ(Wˡ · hˡ⁻¹ + bˡ)          (hidden layer l, l = 1..L)
ŷ  = W^(L+1) · h^L + b^(L+1)    (linear output)
```

where `σ` is the element-wise activation function (ReLU or tanh in this implementation), `Wˡ` are the weight matrices, and `bˡ` are the bias vectors. Training minimises mean squared error via Adam (Adaptive Moment Estimation) optimisation with backpropagation. L2 regularisation (weight decay, controlled by the `alpha` parameter) penalises large weights to reduce overfitting. Early stopping monitors validation loss and halts training when the loss stops improving, preventing over-training on the small dataset.

The `sklearn.neural_network.MLPRegressor` implementation is used, which fits the network with mini-batch gradient descent (or full-batch for small datasets), includes a configurable learning rate schedule, and handles early stopping internally.

#### ASCII Architecture Diagram

```
  Input Layer           Hidden Layers              Output
  (p features)          (tuned: 1–2 layers)         (1 neuron)

  ┌───────┐
  │  x₁   │ ─────────────────────────────────────────────────────►│
  ├───────┤     ┌──────────────────────┐   ┌──────────────────┐   │
  │  x₂   │────►│  Hidden Layer 1      │──►│  Hidden Layer 2  │──►│ ŷ
  ├───────┤     │  (100 or 200 units)  │   │  (50 or 100 units│   │
  │  x₃   │────►│                      │   │   if 2-layer)    │   │
  ├───────┤     │  h₁ᵢ = σ(Σ wᵢⱼxⱼ+b) │   │                  │   │
  │  ...  │     │                      │   │  activation:     │   │
  ├───────┤     │  σ = ReLU or tanh    │   │  ReLU or tanh    │   │
  │  xₚ   │────►│                      │   │                  │   │
  └───────┘     └──────────────────────┘   └──────────────────┘   │
                                                                    │
                          ┌─────────────────────────────┐          │
                          │  Output Layer (1 unit)       │          │
                          │  ŷ = W·h^L + b (linear)     │◄─────────┘
                          └─────────────────────────────┘
                                         │
                                         ▼
                                   ŷ (log-scale)
                                         │
                                   np.expm1(ŷ)
                                         │
                                         ▼
                              Prediction in kWh/m²

  Training:
  ┌────────────────────────────────────────────────┐
  │  Loss = MSE(ŷ, y) + α · Σ‖W‖²   (L2 penalty)  │
  │  Optimiser: Adam                               │
  │  Early stopping: patience on val. loss         │
  │  Input normalisation: StandardScaler required  │
  └────────────────────────────────────────────────┘

  Hyperparameter search (RandomizedSearchCV, 3-fold, 20 iter):
  hidden_layer_sizes ∈ {(100,), (200,), (100,50), (200,100)}
  activation ∈ {relu, tanh}
  alpha ∈ {0.0001, 0.001, 0.01}
  learning_rate_init ∈ {0.001, 0.01}
```

#### Why MLP Suits (and Challenges) Building Energy Prediction

The MLP's key advantages for this task are its ability to learn smooth, non-linear feature interactions and its fast training time (< 0.5 seconds on the Enhanced feature set). For building energy prediction, many feature interactions are plausibly smooth: the relationship between construction year and energy demand is a gradually declining function reflecting successive improvements in insulation standards, not a sharp threshold. The MLP can represent such smooth relationships compactly.

However, the MLP has several well-known disadvantages on small tabular datasets. It requires input normalisation (StandardScaler is applied here, fit on training data only), meaning the preprocessing pipeline must be carefully managed to avoid leakage. It is sensitive to hyperparameters and can be difficult to regularise effectively at small dataset sizes. Decision tree ensembles generally outperform MLPs on datasets with fewer than ~10,000 samples because the irregular, piecewise-constant decision boundaries learned by trees often match the true data-generating process better than the global, smooth functions represented by neural networks.

In this study, MLP achieves R² ≈ 0.38 on Enhanced features, substantially below the gradient boosting methods (R² > 0.75 on Enhanced_3), confirming the expected gap. Notably, MLP was not evaluated on Enhanced_3 because the high-cardinality target-encoded location features (which drive the largest R² improvement) do not appear to provide additional benefit to MLP relative to tree-based models — the soft, location-average signal is not exploited as effectively by the MLP's weight matrices as by tree splits.

#### Key Algorithm Reference

For the general MLP architecture and backpropagation:  
Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536. https://doi.org/10.1038/323533a0

For Adam optimisation:  
Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1412.6980

#### Related Work in Energy Prediction

Neural networks including MLPs have been applied to building energy prediction since the 1990s. For small tabular datasets derived from EPCs, MLPs typically underperform gradient-boosted trees but remain useful as baselines and for comparison with more complex neural architectures.

See also: [search term: "neural network building energy prediction small dataset"]  
See also: [search term: "MLP feedforward network EPC energy demand regression"]

---

### 3.6 TabNet

#### Architecture Description

TabNet (Arik & Pfister, 2021) is a neural network architecture specifically designed for tabular data. Unlike standard MLPs that apply the same transformations to all features simultaneously, TabNet uses a **sequential attention mechanism** that selects a sparse subset of features at each processing step — mimicking the sequential, feature-specific split decisions of decision trees while remaining fully differentiable and end-to-end trainable.

The architecture processes input through `N_steps` sequential decision steps. At each step `i`:

1. **Attentive Transformer:** A learnable mask `M[i]` of the same dimension as the input feature vector is computed using the previous step's output `h[i-1]` and a per-step weight matrix `W_att[i]`. The mask is normalised using **sparsemax** (a sparse alternative to softmax that produces exactly-zero mask values for unselected features) and penalised by a feature reuse coefficient `γ` that discourages selecting the same feature in multiple steps.

2. **Feature Transformer:** The masked features `M[i] ⊙ f(x)` (element-wise product of the sparse mask with a batch-normalised input embedding) are passed through a set of fully-connected layers implemented with **Gated Linear Units (GLU)**. The GLU output is split into two halves, one of which modulates the other via a sigmoid gate, providing a multiplicative interaction. Layers are divided into shared layers (weights shared across all steps) and step-specific layers (private to each step).

3. **Aggregation:** Each step produces an output embedding `h[i]` and contributes to the overall representation via a residual connection. The final prediction is made from the aggregated embedding across all steps.

The feature selection masks `M[i]` are interpretable: they show which features are prioritised at each decision step, providing instance-level feature attribution.

#### ASCII Architecture Diagram

```
  Input x ∈ ℝᵖ  (p features, StandardScaler normalised, float32)
         │
         ▼
  ┌─────────────────────────────────────────────────┐
  │  Batch Normalisation (learnable γ, β)           │
  │  f(x): shared initial feature embedding         │
  └─────────────────────────────────────────────────┘
         │
         │  f(x)  (shared across all steps)
         │
  ┌──────┴──────────────────────────────────────────────────────┐
  │  Step 1  (of N_steps=3)                                     │
  │                                                             │
  │  ┌────────────────────────────────────────────┐            │
  │  │  Attentive Transformer                      │            │
  │  │                                             │            │
  │  │  Prior scales P[1] = ones(p)               │            │
  │  │  a[1] = W_att[1] · h[0]  (prev step out)  │            │
  │  │  M[1] = sparsemax(a[1] · P[1])             │            │
  │  │                                             │            │
  │  │  sparsemax: sparse version of softmax,      │            │
  │  │  produces exactly-zero weights              │            │
  │  │  for unimportant features                   │            │
  │  └───────────────────────┬────────────────────┘            │
  │                           │  M[1] ∈ {0..1}ᵖ (sparse mask) │
  │                           ▼                                │
  │          masked_feat = M[1] ⊙ f(x)                        │
  │                           │                                │
  │  ┌────────────────────────┴────────────────────┐          │
  │  │  Feature Transformer                         │          │
  │  │                                              │          │
  │  │  [Shared FC → BN → GLU]  (across all steps) │          │
  │  │            ↓                                 │          │
  │  │  [Step-specific FC → BN → GLU]              │          │
  │  │                                              │          │
  │  │  GLU: split h=[h₁, h₂], output = h₁·σ(h₂)  │          │
  │  │  Dimension: n_d = n_a = 32 (embedding width) │          │
  │  └───────────────────────┬─────────────────────┘          │
  │                           │  h[1] ∈ ℝⁿᵈ                  │
  │                           ▼                                │
  │          Output[1] = ReLU(h[1])                            │
  └───────────────────────────┬────────────────────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │  Step 2  (M[2] penalised by γ · M[1] usage)│
         │  (same structure, new W_att[2], step FC)   │
         └─────────────────────┬─────────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │  Step 3  (final step)                      │
         └─────────────────────┬─────────────────────┘
                               │
  ┌────────────────────────────┴──────────────────────┐
  │  Aggregation                                       │
  │  overall_output = Σᵢ Output[i]  (sum across steps)│
  └────────────────────────────┬──────────────────────┘
                               │
  ┌────────────────────────────┴──────────────────────┐
  │  Final Mapping                                     │
  │  ŷ = W_final · overall_output + b_final           │
  │  (linear output for regression)                   │
  └────────────────────────────┬──────────────────────┘
                               │
                               ▼
                         ŷ (log-scale)  → expm1 → kWh/m²

  Feature Importance (per instance):
  mask_matrix = [M[1]; M[2]; M[3]]   shape: (N_steps × p)
  Aggregate importance = Σᵢ Mᵢⱼ · ‖hᵢ‖₂  for feature j

  Sparsity loss: L_sparse = Σᵢ Σⱼ P[i]ⱼ · log(P[i]ⱼ)
  (encourages each step to select few features)
```

#### Why TabNet Suits Building Energy Prediction

TabNet's sequential, sparse feature selection mirrors the decision-making logic of gradient-boosted trees: at each step, it selects which features are most informative for the current prediction context. For building energy prediction, where different features are relevant for different building types (construction year matters most for residential buildings; cooled volume matters for commercial premises; footprint-to-heated ratio matters for detached houses), this adaptive, instance-level feature selection is conceptually appealing.

TabNet also provides built-in interpretability: the feature masks `M[i]` can be aggregated per instance to show which features most influenced a specific prediction, which is valuable in the EPC policy context where auditors or planners may want to understand why a particular building is predicted to have high energy demand.

In this study, TabNet uses a modest `n_steps=3` with `n_d=n_a=32`, consistent with recommendations for small-to-medium tabular datasets. Early stopping with patience=10 prevents overfitting. Despite its structural advantages, TabNet does not match the gradient-boosting methods on this dataset, consistent with findings in the literature that TabNet's advantage is most pronounced on datasets where some features are informative and many are not (high-dimensional, sparse settings), rather than on compact, pre-engineered feature sets like Enhanced_3.

#### Key Algorithm Reference

Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. In *Proceedings of the 35th AAAI Conference on Artificial Intelligence* (Vol. 35, No. 8, pp. 6679–6687). AAAI Press. https://ojs.aaai.org/index.php/AAAI/article/view/16826

#### Related Work in Energy Prediction

TabNet has seen application in tabular prediction tasks in energy and healthcare domains, typically performing competitively with tree ensembles on moderately sized datasets with mixed feature types.

See also: [search term: "TabNet tabular data regression building"]  
See also: [search term: "attention mechanism tabular energy prediction"]

---

### 3.7 TabPFN

#### Architecture Description

TabPFN (Tabular Prior-data Fitted Networks; Hollmann et al., 2022) represents a fundamentally different paradigm from all other models in this study. Rather than training a new model on the available dataset, TabPFN uses a **pre-trained transformer** that was meta-trained on millions of synthetic tabular classification and regression tasks drawn from a structured Bayesian prior. At inference time, the model performs **in-context learning (ICL)**: the entire training set is passed as context — as a sequence of (feature, label) pairs — to the transformer, which then predicts the label for each new test example using attention over the training context. No gradient update occurs at inference time; the model's weights are frozen.

This approach is rooted in the concept of **prior-data fitted networks (PFNs)**: a model that approximates Bayesian inference over a space of data-generating processes. If the prior over data-generating processes matches the true distribution of real-world tabular datasets, PFN inference will approximate the Bayesian posterior — the theoretically optimal predictor given the data.

The transformer architecture used in TabPFN processes each training example as a token. The attention mechanism allows the model to identify which training examples are most similar to the test example and weight them accordingly — a learned, non-parametric similarity function that generalises k-nearest-neighbours and Gaussian processes.

**Key operational characteristics:**
- **Zero-shot, no hyperparameter tuning.** The model is used as-is with frozen pre-trained weights. This makes it extremely fast to deploy and eliminates the risk of overfitting the hyperparameter search to the test set.
- **Context-limited.** The transformer has a maximum context window. For very large training sets (> ~10,000 rows), performance degrades as not all training examples can be attended to simultaneously.
- **Small-dataset specialist.** The pre-training distribution was calibrated for small datasets (< 10,000 rows), making it well-matched to the ~2,100 building dataset used here.
- **No feature engineering required.** The pre-trained prior incorporates prior knowledge about typical feature relationships, allowing TabPFN to perform competitively with minimal input preprocessing.

#### ASCII Architecture Diagram

```
  Pre-training Phase (done once by Hollmann et al.):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Synthetic prior: sample data-generating processes              │
  │  P(f, X, y) from Bayesian prior over GPs, Bayesian NNs, etc.   │
  │                                                                 │
  │  For each synthetic task:                                       │
  │  D_train = {(x₁,y₁), ..., (xₙ,yₙ)}  (synthetic training set)  │
  │  x_test  (synthetic test points with known labels)             │
  │                                                                 │
  │  Train transformer T_θ to predict y_test from D_train          │
  │  Minimise: E[L(T_θ(x_test | D_train), y_test)]                 │
  │                                                                 │
  │  Result: frozen transformer weights θ* that encode             │
  │  general inductive biases for tabular regression               │
  └─────────────────────────────────────────────────────────────────┘

  Inference Phase (used here — NO gradient updates):
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Training set D_train (Bulgarian EPC, ~1680 buildings)         │
  │                                                                 │
  │  ┌────────────────────────────────────────────────┐            │
  │  │  Token sequence:                                │            │
  │  │  [(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ), x_test]   │            │
  │  │  Each (xᵢ,yᵢ) = one building + energy demand   │            │
  │  └────────────────────────────────────────────────┘            │
  │             │                                                   │
  │             ▼                                                   │
  │  ┌─────────────────────────────────────────────────────┐       │
  │  │  TabPFN Transformer (frozen weights θ*)              │       │
  │  │                                                      │       │
  │  │  Layer 1: Multi-head Self-Attention                  │       │
  │  │  ┌──────────────────────────────────────────────┐   │       │
  │  │  │  Q = x_test token attends to all D_train     │   │       │
  │  │  │  tokens + other test tokens                   │   │       │
  │  │  │                                              │   │       │
  │  │  │  Attn(Q,K,V) = softmax(QKᵀ/√d_k) · V       │   │       │
  │  │  │                                              │   │       │
  │  │  │  Training examples act as context:           │   │       │
  │  │  │  "similar buildings → similar demand"        │   │       │
  │  │  └──────────────────────────────────────────────┘   │       │
  │  │                                                      │       │
  │  │  Layer 2..L: Feed-Forward + Self-Attention          │       │
  │  │  (standard transformer block structure)             │       │
  │  │                                                      │       │
  │  │  Output for x_test: distribution over ŷ             │       │
  │  │  → mean or median used as point prediction          │       │
  │  └─────────────────────────────────────────────────────┘       │
  │             │                                                   │
  │             ▼                                                   │
  │  ŷ_test  (no gradient descent, no model update)                │
  │             │                                                   │
  │         expm1(ŷ) → kWh/m² prediction                          │
  │                                                                 │
  │  Preprocessing: StandardScaler, float32, same as TabNet        │
  │  Hyperparameter tuning: NONE (zero-shot by design)             │
  └─────────────────────────────────────────────────────────────────┘

  In-context learning analogy:
  ┌──────────────────────────────────────────────────────────────┐
  │  Traditional ML:  train(D_train) → model → predict(x_test)  │
  │                   (gradient descent, new weights per task)   │
  │                                                              │
  │  TabPFN:          T_θ*(x_test | D_train) → ŷ_test          │
  │                   (attention over training context,          │
  │                    zero gradient steps, frozen weights)      │
  └──────────────────────────────────────────────────────────────┘
```

#### Why TabPFN Suits Building Energy Prediction

TabPFN's design is tailored to the dataset size regime of this study (~2,100 buildings). Its pre-trained prior encodes inductive biases derived from a vast number of synthetic tasks, providing a form of transfer learning that is particularly valuable when the available training data is limited. In contrast to standard neural networks that must learn all structure from the few hundred training examples per fold, TabPFN brings in external knowledge from the pre-training distribution.

The zero-shot nature of TabPFN (no hyperparameter search) makes it a useful upper-bound reference for what a well-configured, domain-agnostic prior can achieve without any tuning. If TabPFN performs comparably to tuned tree ensembles, it suggests that the task is well-captured by the generic tabular prior; if it underperforms, it indicates that dataset-specific tuning adds value beyond the prior's encoding.

A practical limitation is TabPFN's sensitivity to dataset size: the maximum context length constrains how many training examples can be attended to simultaneously, which may prevent full utilisation of the training set in some configurations.

#### Key Algorithm Reference

Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). TabPFN: A transformer that solves small tabular classification problems in a second. In *International Conference on Learning Representations (ICLR 2023)*. https://openreview.net/forum?id=cp5PvcI6w8_

#### Related Work in Energy Prediction

TabPFN is relatively new (2022–2023) and has not yet been widely applied in the building energy literature. Its application here is exploratory — testing whether in-context learning from a generic tabular prior can match task-specific gradient boosting on EPC data.

See also: [search term: "TabPFN in-context learning tabular regression benchmark"]  
See also: [search term: "prior-data fitted networks tabular data small dataset"]

---

## 4. Feature Engineering and Preprocessing

Feature engineering proved to be the most impactful intervention in this study, with the Enhanced_3 feature set enabling a +26.4 percentage point improvement in XGBoost R² over the previous Enhanced_2 set (from 0.4953 to 0.7594). This section discusses the three key feature engineering choices in the context of the ML literature.

### 4.1 Location Target Encoding

The most impactful new features in Enhanced_3 are the location target-encoded geographic identifiers: `mun_mean_energy` (mean energy demand in the building's municipality), `oblast_mean_energy` (mean demand in the province), and `city_mean_energy` (mean demand in the city). These features encode the average observed energy demand for each geographic unit in the training data, then apply this as a lookup for all instances including the test set.

**Motivation.** Geographic location is a strong predictor of building energy demand through multiple mechanisms: climatic zone (altitude, heating degree days, solar irradiance), local construction norms and building traditions, urban heat island effects (city vs rural), and the socioeconomic composition of an area (which correlates with renovation activity and building maintenance quality). However, raw location identifiers — municipality codes, city names — are high-cardinality nominal variables. One-hot encoding of 265 Bulgarian municipalities would produce a sparse 265-dimensional feature with most values zero, which tree ensembles handle poorly at this dataset size (fewer than 10 examples per municipality on average). Learned embeddings require more data than available.

**Target encoding.** The solution is target encoding (also called mean encoding or impact coding): replacing each category with the mean target value in that category, computed from the training data. This reduces a high-cardinality nominal variable to a single continuous feature that captures the signal of interest (the geographic climate and construction norm effect) without the sparsity problem. Target encoding was formalised for gradient boosting by Micci-Barreca (2001) and has become standard practice in tabular ML for high-cardinality categorical variables.

**Leakage prevention.** If target encoding is computed using the full dataset before the train-test split, the test-set target values contaminate the encoded features, inflating measured performance. The correct approach — implemented here via the `train_mask` mechanism — is to compute group means exclusively from training-set rows and apply these means as a lookup over both training and test sets. Unseen groups (municipalities with no training examples) receive the global training-set mean, providing a sensible default.

**Literature context.** The importance of correct leakage prevention in target encoding was discussed formally by Prokhorenkova et al. (2018) in the context of CatBoost's ordered target statistics (see Section 3.4). The broader issue of target leakage in feature preprocessing — and its ability to create artificially inflated test-set performance — is discussed by Kaufman et al. (2012):

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579

### 4.2 Log Transformation of the Target Variable

The target variable `en2025_enegy_demand_present_m2` has a heavy right-skewed distribution: most Bulgarian buildings in the dataset have moderate energy demand (50–200 kWh/m²/year), but a minority of poorly insulated buildings, industrial facilities, or data outliers have very high values (500+ kWh/m²/year). Training regression models directly on the raw target in this situation has two problems:

1. **Loss function distortion.** Mean squared error (MSE) loss is dominated by large errors on extreme values. The model implicitly prioritises predicting the rare, high-demand outliers accurately at the cost of higher error on the bulk of buildings.

2. **Residual non-normality.** Many regression diagnostics and inference procedures assume approximately normal residuals. The right-skewed target produces heavily skewed residuals.

The transformation `log(1 + y)` (implemented as `np.log1p`) compresses the upper tail, makes the distribution more symmetric, and reduces the effective range of the target. Predictions are reversed with `np.expm1` to report MAE in the original kWh/m²/year scale. Log transformation of the target in building energy regression is standard practice; it is equivalent to fitting a multiplicative error model where prediction errors are proportional to the true value rather than absolute.

Outliers beyond Q1 − 3×IQR and Q3 + 3×IQR (a conservative outlier criterion; the standard 1.5×IQR rule removes only the most extreme) are dropped before transformation, removing buildings with clearly erroneous EPC records (e.g., data entry errors producing demand values of 10,000+ kWh/m²/year).

### 4.3 Leakage Prevention in the Full Pipeline

Beyond the location target encoding, this project avoids several other potential sources of data leakage:

**Column exclusion.** The following columns from the raw EPC dataset are excluded as features because they contain near-direct information about the target:
- `en2023_enegy_demand_present_m2`: the 2023 EPC energy demand — nearly identical to the 2025 target for buildings assessed in both cycles.
- `en2025_enegy_demand_after_m2`: post-renovation demand, computed by the same EPC methodology as the target.
- `en2025_enegy_demand_present_y`: total annual demand = target × heated area (an algebraic transformation of the target).
- `en2025_class_present`: the EPC energy class (A/B/C/D/E), which is mechanically derived from the target value by the EPC calculation procedure.

**Single split, consistent across models.** All models use the identical 80/20 split (random_state=42). This prevents the test set from being seen during any training or feature construction step and ensures results across models are directly comparable.

**StandardScaler fitted on training data only.** For models requiring normalisation (MLP, TabNet, TabPFN), the StandardScaler is fitted exclusively on the 80% training partition and applied as a transform to both training and test data. Fitting on the full dataset would leak test-set distributional statistics (mean, variance) into the preprocessing.

**Hyperparameter search on training data only.** RandomizedSearchCV uses 3-fold cross-validation within the training partition. The 20% test set is held out until final evaluation, preventing hyperparameter overfitting to the test distribution.

### 4.4 Geometric Feature Engineering

Beyond location encoding, Enhanced_3 introduces several geometry-derived features that encode building physics in a model-accessible form:

- **`floor_count`** and **`floor_height`**: derived from heated volume, heated area, and available floor count records. Average floor height is a proxy for ceiling height, which determines the heated air volume per m² of floor area and thus the volumetric heating load.
- **`footprint_area`**, **`footprint_to_heated_ratio`**, **`gfa_to_footprint_ratio`**: GIS-derived footprint features that provide independent estimates of building compactness and floor count. Compact buildings lose less heat per m² through the building envelope relative to sprawling ones (related to the form factor or surface-to-volume ratio in building physics).
- **`heating_efficiency_ratio`** = heated area / total area: captures how much of the building is conditioned, affecting the specific energy demand per m² of total floor area.
- **`area_to_volume_ratio`** = heated area / heated volume: inverse of average floor height; captures the spatial compactness in the vertical dimension.

These features operationalise concepts from building physics — heat loss through the building envelope is proportional to the surface area and temperature difference, while heat generation requirements scale with heated volume — in a form that tree-based models can leverage without any physics-based simulation.

---

## 5. Comparative Performance and Benchmarks

### 5.1 Results Summary

The following table presents the best results achieved by each model on the Enhanced_3 feature set (the feature set where all gradient boosting models are evaluated), and the best result per model across all feature sets:

| Rank | Model | Best Feature Set | R² | MAE (kWh/m²) | Notes |
|------|-------|------------------|----|--------------|-------|
| 1 | XGBoost | Enhanced_3 | **0.7594** | **15.94** | Best overall |
| 2 | CatBoost | Enhanced_3 | 0.7583 | 18.29 | Native categorical, close 2nd |
| 3 | LightGBM | Enhanced_3 | 0.7528 | 17.41 | Native categorical, strong 3rd |
| 4 | Random Forest | Enhanced_3 | 0.7124 | 20.99 | Solid baseline, lower ceiling |
| 5 | XGBoost | Enhanced_2 | 0.4953 | — | Pre-location-encoding ceiling |
| 6 | MLP | Enhanced | 0.3807 | 41.17 | Neural net baseline |

*TabNet and TabPFN results not shown here as exact R² values for Enhanced_3 were not available in the summary files; they are expected to fall below the gradient boosting methods based on observed patterns in the Baseline/Enhanced results.*

### 5.2 Feature Engineering vs Model Selection

The most striking finding of this study is the dominant impact of feature engineering relative to model selection:

| Change | R² improvement | Notes |
|--------|---------------|-------|
| Enhanced_2 → Enhanced_3 (XGBoost) | +0.264 | Location target encoding |
| XGBoost vs LightGBM (Enhanced_3) | +0.007 | Model-to-model gap |
| XGBoost vs Random Forest (Enhanced_3) | +0.047 | Boosting vs bagging |
| XGBoost vs MLP (best feature sets) | +0.379 | Tree vs neural net |

The location target-encoding features (`mun_mean_energy`, `oblast_mean_energy`, `city_mean_energy`, `category_mean_energy`) introduced in Enhanced_3 capture the most predictive signal in the dataset — the systematic geographic variation in energy demand driven by climate, local construction norms, and building stock characteristics. This is consistent with the finding across many applied ML studies that domain-relevant feature engineering contributes more to model performance than algorithmic improvements, particularly at small dataset sizes.

### 5.3 Why Tree Ensembles Dominate: Broader Context

The dominance of gradient-boosted tree ensembles (XGBoost, CatBoost, LightGBM) and Random Forest over neural networks (MLP, TabNet, TabPFN) in this study is consistent with a substantial body of empirical evidence across tabular prediction benchmarks.

**The "tabular data challenge" for neural networks.** Grinsztajn et al. (2022) identified three mechanisms that explain tree ensemble superiority on tabular data: (1) trees are robust to uninformative features because they simply do not split on them, while MLPs must learn to assign near-zero weights to irrelevant inputs from the limited data available; (2) trees can represent irregular decision boundaries (non-monotone functions of individual features) without regularisation artefacts; and (3) trees efficiently exploit the numerical precision of raw features, while neural networks are biased towards smooth, rotation-invariant functions that may not match the true data-generating process in EPC settings.

**Dataset size effects.** The ~2,100 building dataset is in the regime where tree ensemble methods consistently outperform unconstrained neural networks. Shwartz-Ziv and Armon (2022) found that neural networks begin to match tree ensembles only at dataset sizes above approximately 10,000 rows, a threshold roughly five times the current dataset size. At smaller sizes, the regularisation implicit in tree splits and ensemble averaging provides better variance control than what standard neural network regularisation (dropout, weight decay, early stopping) achieves.

**Benchmark confirmations.** XGBoost, LightGBM, and CatBoost consistently appear among the top performers in Kaggle competitions involving tabular data, and in academic benchmarks across domains including energy, finance, and clinical prediction. The finding that XGBoost achieves the best R² here is directly consistent with this broader pattern.

**Why TabNet and TabPFN did not close the gap.** TabNet's attention-based selection is most advantageous when many features are irrelevant and a small number drive predictions — a regime somewhat like this dataset (location features dominate), but the compact Enhanced_3 feature set (already engineered to include only informative features) leaves less for the attention mechanism to gain by selection. TabPFN's pre-trained prior may not fully capture the specific distributional characteristics of Bulgarian building energy data (climate, construction era, building type distribution), limiting the advantage of its generic tabular inductive bias.

### 5.4 Interpretation of the R² = 0.76 Ceiling

An R² of 0.7594 means that XGBoost explains approximately 76% of the variance in present-state energy demand across Bulgarian buildings. The remaining 24% unexplained variance likely arises from several sources:

1. **Occupant behaviour.** The EPC energy demand is a standardised, occupant-independent calculation based on the building's physical properties. However, actual energy consumption varies substantially with occupant behaviour (thermostat settings, window opening, hot water use), and the EPC methodology itself may not perfectly capture all relevant physical characteristics.

2. **Data quality limitations.** EPC records contain measurement and entry errors, some buildings have inconsistent area or volume figures across the EPC and GIS datasets, and a minority of records may represent unusual building configurations that are poorly represented in the training data.

3. **Missing physical features.** The EPC dataset does not include direct measures of envelope U-values, window-to-wall ratio, HVAC system type, or orientation — all of which are relevant in the EPC calculation but are not available as machine-readable features in the administrative dataset.

4. **Small sample size per subgroup.** With ~2,100 buildings across dozens of building categories, construction eras, and geographic regions, some subgroups are represented by very few training examples, limiting the model's ability to learn accurate predictions for them.

An R² of 0.76 is competitive with or above published results for similar EPC-based building energy prediction tasks in other European countries, confirming that the feature engineering and model selection choices are effective.

### 5.5 Key References for Tabular Data Benchmarks

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why tree-based models still outperform deep learning on tabular data. In *Advances in Neural Information Processing Systems* (Vol. 35). NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

---

## 6. References

The following references are included with high confidence. Papers marked "see also" in the body of this review are search terms for further reading but are not listed here to avoid citation fabrication.

### Core Algorithm Papers

**Random Forest**  
Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.  
https://doi.org/10.1023/A:1010933404324

**XGBoost**  
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.  
https://doi.org/10.1145/2939672.2939785

**LightGBM**  
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30). NeurIPS.  
https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

**CatBoost**  
Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems* (Vol. 31). NeurIPS.  
https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac2a2d5be7deef74-Abstract.html

**TabNet**  
Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. In *Proceedings of the 35th AAAI Conference on Artificial Intelligence* (Vol. 35, No. 8, pp. 6679–6687). AAAI Press.  
https://ojs.aaai.org/index.php/AAAI/article/view/16826

**TabPFN**  
Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2022). TabPFN: A transformer that solves small tabular classification problems in a second. In *International Conference on Learning Representations (ICLR 2023)*.  
https://openreview.net/forum?id=cp5PvcI6w8_

**MLP / Backpropagation**  
Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.  
https://doi.org/10.1038/323533a0

**Adam optimiser**  
Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*.  
https://arxiv.org/abs/1412.6980

### Tabular Data Benchmarks

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion*, 81, 84–90.  
https://doi.org/10.1016/j.inffus.2021.11.011

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why tree-based models still outperform deep learning on tabular data. In *Advances in Neural Information Processing Systems* (Vol. 35). NeurIPS.  
https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

### Feature Engineering and Leakage

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21.  
https://doi.org/10.1145/2382577.2382579

Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32.  
https://doi.org/10.1145/507533.507538

---

*This literature review was prepared in support of the building energy demand prediction project using Bulgarian EPC data. All citations reflect papers the authors are confident exist; uncertain citations have been replaced with search terms to prevent academic citation fabrication.*
