# Explainability & Dimensionality Reduction Methods
## Plain-Language Guide: PCA, KMeans, SHAP, and LIME

---

## 1. PCA (Principal Component Analysis)

### What problem does it solve?

You have 33 features for each building. Many of them move together — `area_heated`, `area_total`, and `volume_heated` all increase together as buildings get bigger. This means you have 33 numbers per building, but not 33 truly independent pieces of information.

PCA asks: **what are the underlying independent "themes" that actually vary across your buildings?** Instead of 33 partially-redundant numbers, it finds the smallest set of new axes — called principal components — that together explain most of the variation in the data.

Each component is a weighted combination of your original features. PCA ranks them by how much of the total variation they explain.

---

### What each component is saying, in plain terms

Think of each component as answering one specific question about how buildings differ from each other.

| Component | Variance Explained | Plain-language meaning |
|---|---|---|
| PC1 | 22.8% | **What type of building is it?** Separates buildings by category, energy class, and how closely they match their expected energy profile |
| PC2 | 11.9% | **How big is it?** Pure physical size — area, volume, and footprint all moving together |
| PC3 | 10.1% | **How old is it?** Purely construction year, building age, and which regulatory era it was built in |
| PC4 | 9.1% | **What shape is it?** Compact tower vs. sprawling low building — form factor, independent of size |
| PC5 | 6.0% | **Is it a large apartment block?** Many small flats on many floors vs. fewer large units |
| PC6–PC10 | ~24% | Smaller, harder-to-name residual variation |

Together, the first 10 components capture **80.8%** of all the variation across your 33 features.

---

### Why these components matter

The fact that PCA found exactly these five dominant themes is not trivial — it is the data confirming that **the main ways buildings differ from each other in your dataset are: type, size, age, shape, and apartment scale.** That aligns with what building physicists already know drives energy demand. This is a good sign that your feature engineering captured real, independent signals.

If PC3 (age) and PC1 (type) had been mixed together into the same component, it would mean your age features and type features were redundant — you would only need one of them. They are separate, which confirms that both genuinely add information to your model.

---

### What you can practically use PCA for

**1. Sanity check on your features**
Each component being interpretable and distinct confirms that your 33 features are not all measuring the same thing. You have genuine independent predictors.

**2. Spotting outlier buildings**
If a building plots far from the main cloud in PC1 × PC2 space, it is unusual across both type and size simultaneously. These are worth inspecting — they may be data errors or genuinely unusual buildings (e.g. a converted industrial building registered as residential).

**3. Visual exploration without losing information**
You cannot plot 33 dimensions. PCA lets you project everything onto the two most important axes (PC1 vs. PC2) and still capture 34.7% of total variation — the best possible 2D summary of your dataset.

**4. Dimensionality reduction for lighter models**
Instead of 33 features, you could feed just the 10 PCA scores into a model and still retain 80.8% of the information. Useful if you need a simpler, faster, or more interpretable model downstream.

**5. Communicating the structure of the dataset**
Instead of showing 33 feature distributions, you can describe the building stock in 5 plain-language dimensions to stakeholders.

---

## 2. KMeans Clustering in PCA Space

### What it does

After PCA compresses your 33 features into a smaller set of independent dimensions, KMeans groups buildings that are close to each other in that compressed space. The result is a set of natural clusters — buildings that are similar to each other across all dimensions simultaneously.

The notebook used **5 clusters** applied on the first 5 PCA components.

---

### Why cluster in PCA space, not on the raw features?

If you ran KMeans directly on the 33 raw features, the correlated features would each "vote" separately for the same thing. `area_heated`, `area_total`, and `volume_heated` would each count once, so size would dominate the clusters and every other dimension would get drowned out. PCA strips that redundancy first, so clusters reflect genuine differences across *all* dimensions equally weighted.

---

### The 5 clusters found

| Cluster | Buildings | Median energy | Mean energy | What it likely represents |
|---|---|---|---|---|
| 3 | 84 | ~80 kWh/m² | ~81 kWh/m² | Recently built or renovated — low energy demand, modern standards |
| 0 | 567 | ~110 kWh/m² | ~128 kWh/m² | Average mainstream residential stock |
| 2 | 405 | ~139 kWh/m² | ~147 kWh/m² | Older mid-range stock, transitional era |
| 4 | 37 | ~140 kWh/m² | ~142 kWh/m² | Small, geometrically unusual subgroup — similar energy level to Cluster 2 |
| 1 | 184 | ~186 kWh/m² | ~183 kWh/m² | Old and inefficient — highest per-m² energy demand |

Note: The energy values are log-transformed in the analysis. The kWh/m² figures shown here reflect the approximate back-transformed scale for readability.

The clusters span roughly a **2.3× range** in median energy demand from lowest to highest.

---

### What you can practically use the clusters for

**1. Targeting renovation policy**
Cluster 1 (184 buildings, median 186 kWh/m²) is the priority group for energy renovation. You can extract those specific buildings and identify what they share in common — likely old construction year, pre-1970 era, poor insulation standards. Interventions targeted at this cluster have the highest potential impact per building.

**2. Benchmarking buildings within their peer group**
Comparing Building A to the average of *all* 2,010 buildings is misleading if Building A is a large old apartment block and the average includes small new houses. Comparing it to its cluster average controls for type, size, age, and shape simultaneously — a fairer benchmark.

**3. Understanding the within-cluster spread**
Cluster 0 has a mean (127.6) significantly higher than its median (110.4). This means most buildings in the mainstream group are near-average, but a tail of very inefficient buildings pulls the mean up. Policy targeting this cluster should focus on its upper quartile, not the average.

**4. Reporting and segmentation**
Instead of describing the dataset as "buildings with an average of 130 kWh/m²", you can report 5 distinct building profiles, each with a clear energy band and physical interpretation. This is much more actionable for planners and building owners.

---

## 3. SHAP (SHapley Additive exPlanations)

### What problem does it solve?

Your best model (RF or XGBoost on Enhanced_3) makes a prediction for each building, but it is a black box — it gives you a number without explaining why. SHAP opens that black box by answering: **for this specific building, how much did each feature push the prediction up or down from the average?**

It is grounded in game theory (Shapley values): each feature is treated like a "player" in a game, and SHAP fairly distributes the "credit" for the prediction among all features.

---

### How to read SHAP values

- A **positive SHAP value** for a feature means that feature pushed the prediction *higher* than average for this building
- A **negative SHAP value** means it pushed the prediction *lower* than average
- The sum of all SHAP values + the base value (mean prediction) = the model's actual prediction for that building

Example: If the average predicted energy demand is 130 kWh/m², and for Building X the SHAP value for `construction_year` is -20 and for `area_heated` is +5, it means the building's age is pulling the prediction 20 units below average (it is new), while its size is pushing it 5 units above (it is large).

---

### The four SHAP plots and what they show

**Beeswarm plot**
Every dot is one test building. The x-axis is the SHAP value (how much that feature shifted the prediction). Colour encodes the feature's value: red = high value, blue = low value. Reading it tells you the direction and distribution of each feature's effect.

- `heating_efficiency_ratio`: Buildings with a high ratio (red) tend to have negative SHAP values — meaning highly-heated buildings are predicted to be more efficient (lower demand per m²). Buildings with low ratios (blue) push demand up.
- `construction_year`: Old buildings (blue = early year) have strong positive SHAP values — they are predicted to consume more. New buildings (red) have negative SHAP values.

**Mean |SHAP| bar chart**
This is the global feature importance: the average absolute SHAP value per feature across all buildings. It answers "which features matter most, on average, across the whole dataset?" Unlike LightGBM's split-count importance, this is directly in the units of the prediction (log kWh/m²), so it is more interpretable.

**Dependence plots (top 3 features)**
These show how a single feature's SHAP value changes as its value increases. They reveal the shape of the relationship — is it linear? Does it plateau? Is there a threshold?

- `construction_year`: Expected to show a step-change pattern around key regulatory years (~1970, ~2000), where buildings built after energy regulations became strict suddenly receive much lower SHAP values.
- `heating_efficiency_ratio`: Expected to show a non-linear curve — the effect flattens at very high ratios once a building is already nearly fully heated.
- `footprint_area`: Expected to show that very large footprints (sprawling single-storey buildings) attract higher SHAP values (higher demand), while mid-range footprints (multi-storey compact buildings) are neutral or negative.

The colour on dependence plots shows the most-interacting feature — where the dots split into separate bands by colour, it means the effect of the x-axis feature depends on another feature's value.

**Waterfall plots (best and worst predicted buildings)**
These show the step-by-step feature contributions for one specific building — the one the model predicted best and the one it predicted worst.

The best-predicted building will have many small contributions that nearly cancel: a building that is "average" in most features, where the model has no reason to deviate far from the base value.

The worst-predicted building reveals the model's failure mode: features that point in contradictory directions (e.g. very old but recently retrofitted, so actual demand is far lower than the age signal suggests). These cases are valuable for understanding where the model struggles — usually buildings with unusual retrofit histories or data quality issues.

---

### What SHAP is used for in practice

- **Explaining individual predictions to building owners** — "Your building is predicted to consume 160 kWh/m² mainly because it was built in 1955 (+35) and has a large heated area (+12), partially offset by its efficient floor plan (-8)"
- **Validating the model** — if SHAP shows that an irrelevant feature (e.g. building ID) has high importance, the model may have learned a spurious pattern
- **Feature selection** — features with near-zero mean |SHAP| across all buildings contribute nothing and can be safely removed
- **Comparing two buildings** — waterfall plots for two buildings with similar inputs but very different predictions can highlight which feature is driving the gap

---

## 4. LIME (Local Interpretable Model-agnostic Explanations)

### Important note

LIME is **not implemented in this notebook**. This section explains what it is and how it would apply to this project, in case it is added in a future version.

---

### What problem does it solve?

Like SHAP, LIME explains individual predictions. The approach is different: instead of computing exact Shapley values, LIME fits a simple, interpretable model (a linear regression) in the local neighbourhood around the building you want to explain.

The idea is: even if the global model (XGBoost, Random Forest) is complex and non-linear, in a small region around any single building the relationship between features and predictions is approximately linear. LIME finds that local linear approximation.

---

### How LIME works, step by step

1. Take the building you want to explain (e.g. Building X with prediction 160 kWh/m²)
2. Create many slightly modified versions of Building X by randomly perturbing its features (changing `construction_year` by ±5 years, `area_heated` by ±10%, etc.)
3. Run all those perturbed buildings through the original black-box model to get their predictions
4. Fit a simple linear regression to those perturbed buildings (weighted by how close they are to Building X)
5. The coefficients of that linear regression are the LIME explanation: each feature's coefficient tells you how sensitive the prediction is to that feature, locally around Building X

---

### LIME vs. SHAP: what is the difference?

| | SHAP | LIME |
|---|---|---|
| **Basis** | Game theory (exact Shapley values) | Local linear approximation |
| **Consistency** | Globally consistent — same model, same input always gives same explanation | Approximate — results can vary slightly between runs due to random perturbations |
| **Speed** | Fast for tree models (TreeExplainer) | Slower — requires running many perturbations through the model |
| **Scope** | Both global (beeswarm, bar) and local (waterfall) | Purely local — one explanation per building |
| **Interpretability** | Requires understanding of Shapley values | Directly produces a linear model — very easy to show to non-technical users |
| **Model-agnostic** | TreeExplainer is tree-specific; KernelSHAP is model-agnostic but slow | Fully model-agnostic — works with any model |

---

### When LIME is more useful than SHAP in this project

LIME is particularly valuable when you need to **show a single building's explanation to a non-technical audience** (a building owner, a municipality officer) and want a format like:

> "The three factors most responsible for your building's high energy demand are:
> 1. It was built in 1961 (accounts for +28 kWh/m²)
> 2. Its heated volume is large relative to its floor area (accounts for +15 kWh/m²)
> 3. Its building category has a high average energy demand in this area (accounts for +10 kWh/m²)"

A LIME linear model produces exactly this kind of clean, ranked, signed explanation. SHAP waterfall plots convey the same information but are harder to read for non-technical users.

SHAP is preferred for **model development and validation** (global analysis, feature selection, consistency checks). LIME is preferred for **communicating individual decisions** to end-users.

---

### How to add LIME to this project

Install the library and add a cell after the SHAP section:

```python
# pip install lime
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_tr.values,
    feature_names=feat_names,
    mode='regression'
)

# Explain a specific building (e.g. the worst-predicted one)
idx = worst_idx  # same index used in SHAP waterfall
exp = explainer.explain_instance(
    X_te.values[idx],
    xgb_model.predict,
    num_features=10
)
exp.show_in_notebook()
```

This produces an interactive bar chart showing the top 10 features that drove the prediction for that specific building, with positive and negative contributions clearly labelled.

---

## Summary: When to Use Each Method

| Method | What it answers | Scope | Best used for |
|---|---|---|---|
| PCA | What are the independent dimensions in my data? | Global (whole dataset) | Understanding data structure, visualisation, redundancy check |
| KMeans on PCA | Are there natural groups of similar buildings? | Global (whole dataset) | Segmentation, policy targeting, benchmarking |
| SHAP | How much did each feature contribute to this prediction? | Both global and per-building | Model validation, feature selection, explaining predictions |
| LIME | What is the local linear driver of this specific prediction? | Per-building only | Communicating individual results to non-technical users |

All four methods are complementary. PCA and KMeans describe the **structure of your input data**. SHAP and LIME describe **why the model made a specific prediction**. Together they give a complete picture of both the data and the model's behaviour.
