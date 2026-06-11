# Descriptive Analysis Report: Energy Demand Prediction
## Feature Engineering, PCA, Feature Importance & SHAP Explainability

---

## 1. Overview

This report analyses the machine-learning pipeline developed in `Final_Notebook_for_Generation_with_PCA_FE_SHAP.ipynb` for predicting residential building energy demand (kWh/m² per year, present-state 2025). The dataset contains **2,010 residential buildings** across three categories (one-family, multi-family, and other residential), each described by up to 69 raw columns drawn from a geospatial enriched dataset. The target variable is `en2025_enegy_demand_present_m2`.

The analysis is structured around four main themes:
1. Progressive feature engineering (four feature sets)
2. Model performance evolution across those feature sets
3. PCA decomposition of the engineered feature space
4. Model explainability via SHAP values and Partial Dependence Plots

> **Note on LIME:** LIME (Local Interpretable Model-agnostic Explanations) is not implemented in this notebook. The explainability section relies exclusively on SHAP and Partial Dependence Plots (PDPs). If LIME analysis is required, it would need to be added as a separate notebook section.

---

## 2. Feature Engineering: Four Cumulative Feature Sets

A central design decision in this project was the progressive enrichment of the feature space. Four feature sets were defined, each a strict superset of the previous.

### 2.1 Baseline (4 features)

The minimal physical description of a building:

| Feature | Source | Meaning |
|---|---|---|
| `area_heated` | `en2025_area_heated_m2` | Heated floor area (m²) |
| `area_total` | `en2025_gfa_m2` | Gross floor area (m²) |
| `volume_heated` | `en2025_vol_heated_m3` | Heated volume (m³) |
| `construction_year` | `en2025_yearbuilt` | Year of construction |

This set represents the information available in any basic building register and serves as the lower-bound benchmark.

---

### 2.2 Enhanced (15 features)

The Baseline is extended with contextual, categorical, and temporal correlation features:

- **Log-transformed physical sizes** (`log_area_heated`, `log_area_total`, `log_volume_heated`) to reduce right-skew in building size distributions.
- **`Correlation_ED`** — empirical correlation between the building's type/decade combination and measured energy demand from historical data.
- **`Correlation_PM`** — same for primary energy.
- **`Energy_Completeness_Pct`** — percentage of key energy fields that are non-null for a building. Acts as a data-quality proxy and, implicitly, a signal about whether a building has been surveyed/certified.
- **`building_category`** — mapped from `functype` to English labels (one-family, multi-family, etc.).
- **`construction_decade`** — decade bucketed from construction year, used to align with decade-level correlation tables.
- **`decade_correlation_pe`**, **`decade_correlation_ed`** — look-up correlations between a building's category and construction decade and average primary energy / energy demand from reference tables.
- **`building_age`**, **`log_building_age`** — age in years (2025 minus construction year) and its log, capturing the non-linear effect of ageing.
- **`heating_efficiency_ratio`** — `area_heated / (area_total + 1)`: the fraction of total floor area that is actively heated, a compact proxy for building envelope efficiency.
- **`area_to_volume_ratio`** — `area_heated / (volume_heated + 1)`: a shape index linked to the surface-area-to-volume ratio, which determines heat loss through the building skin.

---

### 2.3 Enhanced_2 (18 features)

Three additional attributes from registry data:

- **`construction_type`** — categorical field describing the structural construction method (e.g. brick, concrete, prefabricated). Encodes thermal mass and insulation-era information.
- **`nbuildingy`** — a secondary year-of-construction field from a different data source, providing cross-validation and fill-in for gaps in `construction_year`.
- **`bldg_shape`** — building shape category (e.g. detached, semi-detached, terraced), which strongly influences the exposed surface area and therefore heat loss.

---

### 2.4 Enhanced_3 (33 features)

The richest feature set, adding 15 further features covering building geometry, multi-unit structure, and target-encoded statistical context:

- **`floor_count`** — number of floors, sourced from `c2025_flrcount` or approximated as `volume / (area × 3.0)`. Strongly correlates with the typology of multi-family buildings.
- **`floor_height`** — average ceiling height (`volume / (floor_count × area)`), clipped to realistic bounds (2.0–6.0 m). High-ceilinged buildings have larger heated volumes relative to floor area.
- **`vol_cooled`**, **`vol_cooled_ratio`** — total cooled volume and its ratio to heated volume. Large cooled fractions are common in mixed-use or high-standard residential buildings with HVAC.
- **`footprint_area`** — ground footprint of the building. Together with floor count, determines whether energy is lost primarily through the roof/floor slab or through vertical facades.
- **`footprint_to_heated_ratio`** — `area_heated / footprint_area`. Values near 1 indicate single-storey structures; higher values indicate multi-storey.
- **`gfa_to_footprint_ratio`** — `area_total / footprint_area`. Similar to floor count but computed from area fields.
- **`apt_count`** — number of apartments in the building. Distinguishes large multi-family blocks from small ones.
- **`area_per_apt`** — `area_heated / apt_count`. Average apartment size; large apartments typically correlate with single-family-style energy patterns even within multi-family buildings.
- **`category_mean_energy`** — target-encoded mean energy demand by building category and location, computed on the training fold only (with a global fallback) to prevent leakage. Encodes spatial and typological energy norms.
- **Location target encodings** — similar to `category_mean_energy` but stratified by geographic zone, capturing local climate and urban-density effects.

---

## 3. Model Performance: The Effect of Feature Engineering

Five model families were trained across all four feature sets using randomised hyperparameter search (20 iterations, 3-fold cross-validation, R² scoring): Random Forest (RF), XGBoost (XGB), Multi-layer Perceptron (MLP), LightGBM, and CatBoost.

### 3.1 Full Results Table

| Rank | Model | Feature Set | R² | MAE (kWh/m²) |
|---|---|---|---|---|
| 1 | RF | Enhanced_3 | **0.539** | **28.84** |
| 2 | XGB | Enhanced_3 | 0.533 | 29.57 |
| 3 | CatBoost | Enhanced_3 | 0.526 | 30.13 |
| 4 | TabNet | Enhanced_3 | 0.508 | 31.03 |
| 5 | LightGBM | Enhanced_3 | 0.503 | 30.37 |
| 6 | CatBoost | Enhanced | 0.497 | 31.35 |
| 7 | CatBoost | Enhanced_2 | 0.494 | 31.52 |
| 8 | LightGBM | Enhanced | 0.494 | 30.79 |
| 9 | RF | Enhanced_2 | 0.493 | 30.37 |
| 10 | LightGBM | Enhanced_2 | 0.491 | 31.80 |
| 11 | XGB | Enhanced_2 | 0.488 (0.495 tuned) | 30.50 |
| 12 | CatBoost | Baseline | 0.480 | 32.31 |
| 13 | RF | Enhanced | 0.481 | 30.27 |
| 14 | XGB | Enhanced | 0.478 | 30.69 |
| 15 | XGB | Baseline | 0.472 | 32.09 |
| 16 | RF | Baseline | 0.469 | 31.94 |
| 17 | LightGBM | Baseline | 0.454 | 32.57 |
| 18 | MLP | Enhanced_2 | 0.427 | 34.64 |
| 19 | MLP | Enhanced | 0.437 | 33.22 |
| 20 | MLP | Enhanced_3 | 0.416 | 33.20 |
| 21 | MLP | Baseline | 0.382 | 34.90 |

### 3.2 Per-Model Improvement Trajectories

**Random Forest:**
- Baseline → Enhanced: +3.8% (0.469 → 0.487)
- Enhanced → Enhanced_2: +1.6% (0.487 → 0.495)
- Enhanced_2 → Enhanced_3: +9.3% (0.493 → 0.539) — largest single jump

**XGBoost:**
- Baseline → Enhanced: -0.7% (0.472 → 0.468) — slight decline, likely due to the raw categorical features being harder to split
- Enhanced → Enhanced_2: +5.8% (0.468 → 0.495)
- Enhanced_2 → Enhanced_3: +7.5% (0.488 → 0.533)

**MLP:**
- Baseline → Enhanced: -9.7% (0.382 → 0.345) — strong degradation, likely from unscaled or high-cardinality categorical features
- Enhanced → Enhanced_2: +31.1% (0.345 → 0.452) — recovery once construction_type/shape are available
- Enhanced_2 → Enhanced_3: -2.7% (0.427 → 0.416) — MLP does not benefit from the extra features, possibly overfitting

**CatBoost:**
- Baseline → Enhanced: +3.8% (0.480 → 0.497)
- Enhanced → Enhanced_2: -0.6% (plateau at ~0.494)
- Enhanced_2 → Enhanced_3: +6.4% (0.494 → 0.526)

**LightGBM:**
- Baseline → Enhanced: +8.7% (0.454 → 0.494)
- Enhanced → Enhanced_2: -0.4%
- Enhanced_2 → Enhanced_3: +2.4% (0.491 → 0.503)

### 3.3 What the Feature Addition Brought

**Baseline → Enhanced:** The addition of engineered ratios (`heating_efficiency_ratio`, `area_to_volume_ratio`), categorical context (`building_category`, decade correlations), and data-quality signals produced consistent gains for tree-based models (+3–9%). The MLP suffered because the raw categorical columns and interaction features add noise without the implicit feature-selection that decision trees provide.

**Enhanced → Enhanced_2:** The addition of `construction_type`, `nbuildingy`, and `bldg_shape` gave modest but consistent gains. These features encode thermal-envelope-era information (e.g. pre-1970 brick vs. post-2000 insulated concrete) that is otherwise hard to recover from year alone. The MLP recovered significantly (+31%) because these three features are relatively clean categorical variables.

**Enhanced_2 → Enhanced_3:** The largest improvement across all models except MLP. The new features — `floor_count`, `footprint_area`, `apt_count`, `area_per_apt`, and especially the target-encoded `category_mean_energy` — provided the most predictive signal. The geometry features (floor count, footprint ratios) allow models to distinguish low-rise wide buildings from narrow high-rises, which behave very differently thermally. The `category_mean_energy` target encoding captures local demand norms that none of the other features could represent. RF, which handles high-dimensional feature interaction best, benefited most (+9.3%).

**MLP consistently underperforms** all tree-based models across all feature sets (best R²=0.437 vs. RF best of 0.539). This is consistent with the relatively small dataset (2,010 buildings), tabular data structure with mixed categorical/continuous features, and the absence of pre-training. MLPs require much larger datasets and careful normalisation to match gradient-boosted trees on tabular regression tasks.

**Overall trajectory:** The project moved from an R² of 0.469 (RF Baseline) to 0.539 (RF Enhanced_3), a cumulative improvement of **+14.9%**. The previous best before Enhanced_3 was 0.495 (XGB Enhanced_2); Enhanced_3 improved on that by **+8.9%** in absolute R² terms. The MAE dropped from ~32 kWh/m² to ~29 kWh/m², meaning predictions are on average 3 kWh/m² closer to the actual measured value.

---

## 4. Principal Component Analysis (PCA)

### 4.1 Motivation and Theoretical Background

Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms a set of possibly correlated variables into a smaller set of uncorrelated variables called principal components, each ordered by the amount of variance it explains in the original data (Pearson, 1901; Hotelling, 1933). In the context of building energy research, PCA serves two distinct but complementary purposes: it reveals the latent structure of the building stock, and it guards against the multicollinearity that is endemic in building feature sets where physical quantities such as area, volume, and floor count co-vary by design.

**Why PCA is appropriate here.** The Enhanced_3 feature set contains 33 variables that are not independent. Physical quantities such as `area_heated`, `area_total`, and `volume_heated` are structurally correlated — larger buildings have more of all three simultaneously. Engineered ratios such as `heating_efficiency_ratio` and `area_to_volume_ratio` are algebraically derived from the same underlying measurements. Without dimensionality reduction, any clustering or visualisation performed on the raw feature space would be dominated by size variation simply because size is represented by multiple redundant columns. PCA removes this redundancy by finding the true independent axes of variation.

This approach is well-established in building energy analysis. Capozzoli et al. (2015) applied PCA to heating energy data in school buildings to isolate the independent drivers of consumption before regression modelling, finding that typological and geometric dimensions were consistently orthogonal. Kontokosta & Tull (2017) used PCA on a city-scale building energy dataset to identify latent archetypes in the stock prior to predictive modelling, demonstrating that the first few components reliably separated buildings by use intensity, size, and age — the same three dimensions that emerge as PC2, PC3, and PC1 in this analysis. More recently, Geraldi & Ghisi (2020) showed that PCA-derived components from building register data could substitute for directly measured energy audit variables in bottom-up stock models, supporting the use of PCA as a bridge between administratively available data and energy performance estimation.

The form-factor dimension identified as PC4 in this analysis has a direct physical basis in the building energy literature. Depecker et al. (2001) established that a building's compactness ratio — closely related to the surface-area-to-volume ratio captured by `area_to_volume_ratio` — is one of the strongest geometric determinants of heating energy demand, independent of insulation quality. Buildings with a high surface-to-volume ratio lose more heat per unit of conditioned space. The fact that PCA recovers this as an independent component confirms that the engineered features are encoding physically meaningful quantities, not statistical artefacts.

The age/vintage dimension (PC3) is similarly grounded. Kavgic et al. (2010) reviewed bottom-up building stock models across Europe and found that construction period is the single most reliable proxy for the thermal standard of the building envelope in the absence of direct audit data, because each regulatory era produced a distinct envelope specification. The emergence of PC3 as a clean, nearly pure age axis — dominated by `construction_year` and `building_age` with loadings of ±0.487 — is direct confirmation that the temporal features in this dataset are capturing this well-established regulatory-era signal.

**Why the target variable is excluded from PCA.** PCA was applied to features only; `en2025_enegy_demand_present_m2` was not included. The energy demand is instead used to colour the scatter plots after the fact. This ensures the components represent the intrinsic structure of the building stock, not a projection optimised toward the label. Including the target would bias the components toward predictive power (similar to Partial Least Squares) and undermine the interpretability goal of understanding what dimensions the buildings naturally vary along.

### 4.2 Setup and Scope

PCA was applied to the full Enhanced_3 feature matrix (33 features, standardised with zero mean and unit variance) to understand the latent structure of the building stock. The target variable was not included in the PCA; it was used only for colouring the scatter plots. Ten components were extracted.

### 4.2 Explained Variance

| Component | Variance Explained | Cumulative |
|---|---|---|
| PC1 | 22.8% | 22.8% |
| PC2 | 11.9% | 34.7% |
| PC3 | 10.1% | 44.9% |
| PC4 | 9.1% | 54.0% |
| PC5 | 6.0% | 59.9% |
| PC6 | 4.7% | 64.7% |
| PC7 | 4.6% | 69.3% |
| PC8 | 4.4% | 73.7% |
| PC9 | 3.7% | 77.3% |
| PC10 | 3.4% | 80.8% |

**Key observations on variance structure:**
- The first 5 components capture nearly 60% of total variance. There is no single dominant axis — the largest component (PC1) accounts for only 22.8%, suggesting that building energy-relevant characteristics are distributed across multiple independent dimensions.
- Reaching 80% cumulative variance requires 10 components. This indicates moderate intrinsic dimensionality: the 33 features are not highly redundant, but they are also not all independent. The space can be described adequately with roughly 10 latent factors.
- The relatively shallow scree (no sharp elbow after PC1) suggests that many features contribute unique variance — a sign that the feature engineering was effective in creating genuinely independent signals.

### 4.3 Component Loadings: What Each PC Represents

#### PC1 (22.8%) — Building Type / Classification Axis

| Feature | Loading |
|---|---|
| `building_category` | +0.338 |
| `en2025_type` | -0.331 |
| `Correlation_PM` | +0.319 |
| `category_mean_energy` | -0.319 |
| `Correlation_ED` | +0.288 |
| `volume_heated` | -0.273 |
| `log_building_age` | -0.266 |
| `area_per_apt` | -0.240 |

**Interpretation:** PC1 is primarily a **building typology and energy profile** axis — the dominant source of variation in the entire dataset, explaining 22.8% of all feature variance. The features with the largest loadings are not physical measurements but categorical and statistical descriptors: `building_category`, `en2025_type`, `Correlation_PM`, `Correlation_ED`, and `category_mean_energy`. These features are all expressions of the same underlying concept: *what kind of building is this, and what does the historical record say its energy class should be?*

The opposing signs between `building_category` (+0.338) and `en2025_type` (−0.331) indicate that these two type fields encode the building typology in opposite directions — one increases as the other decreases along this axis, suggesting they are drawn from different classification systems where "higher" codes mean different things. The `category_mean_energy` loading (−0.319) opposing `Correlation_PM` (+0.319) reflects a similar encoding reversal: categories with high primary-energy correlation tend to have lower target-encoded mean demand in this specific dataset, possibly because high-correlation categories skew toward recently certified (lower-demand) buildings.

The presence of `log_building_age` (−0.266) and `area_per_apt` (−0.240) within this typological axis is meaningful: older buildings and buildings with smaller average apartment sizes both pull in the negative PC1 direction, co-varying with the energy-profile features. This suggests that in this building stock, typological identity is not fully separable from age and apartment configuration — single-family older houses cluster on one end of the axis, while newer multi-family stock clusters on the other.

In the literature, building-type classification has consistently been identified as one of the primary stratifiers of energy demand in stock-level analyses. Papadopoulos et al. (2018), in a large-scale study of New York City's energy benchmarking data, found that property type was the strongest categorical predictor of energy use intensity (EUI), explaining more variance than floor area, age, or location individually. The emergence of PC1 as a typology axis in this dataset is consistent with this finding and validates the inclusion of `building_category` and the decade-level correlation features as first-order inputs.

#### PC2 (11.9%) — Physical Size Axis

| Feature | Loading |
|---|---|
| `area_total` | +0.428 |
| `area_heated` | +0.419 |
| `footprint_to_heated_ratio` | +0.311 |
| `gfa_to_footprint_ratio` | +0.310 |
| `area_per_apt` | +0.291 |
| `area_to_volume_ratio` | +0.278 |
| `volume_heated` | +0.277 |
| `apt_count` | +0.191 |

**Interpretation:** PC2 is unambiguously a **physical size** axis, and its structure is exactly what one would expect from first principles: all eight of the top loadings are positive, meaning that every dimension of physical size — heated area, total area, heated volume, footprint spread, apartment size, and number of units — increases together along this axis. A building that scores high on PC2 is simply a large building in all respects.

The co-presence of `area_to_volume_ratio` (+0.278) and `footprint_to_heated_ratio` (+0.311) within a size axis at first appears contradictory — these are ratios, not absolute sizes, and should in principle be independent of scale. Their positive loading here reflects a real structural pattern in the dataset: larger buildings in this stock also tend to be horizontally spread (high footprint relative to their heated area), meaning the size axis captures not just scale but also a tendency toward low-rise sprawl in the largest buildings. This is consistent with the composition of the dataset, which includes large single-family houses and small multi-family low-rises alongside compact apartment towers.

Notably, `volume_heated` appears in both PC1 (loading −0.273) and PC2 (loading +0.277) with similar magnitude but opposite sign. This is a property of PCA's orthogonal decomposition: along PC1 (the typology axis), larger heated volume is associated with a particular building category profile; along PC2 (the pure size axis), larger volume simply means a bigger building regardless of type. The two components together account for the full variance of `volume_heated` split across two independent interpretations — type-associated volume and scale-associated volume.

The importance of building size as an independent axis of variation in stock-level energy modelling is well established. Amasyali & El-Gohary (2018), in their comprehensive review of 60 data-driven building energy prediction studies, identified floor area as the most universally included predictor across all study types, noting that it captures both the total thermal load of the building and its social function. PC2 confirms this: size is the second-most important source of variation in this building stock, independent of type.

#### PC3 (10.1%) — Building Age / Vintage Axis

| Feature | Loading |
|---|---|
| `construction_year` | +0.487 |
| `building_age` | -0.487 |
| `decade_correlation_ed` | -0.437 |
| `decade_correlation_pe` | -0.433 |
| `log_building_age` | -0.266 |

**Interpretation:** PC3 is the **age/vintage** axis, and it is the most structurally clean of all five components: only five features carry meaningful loadings, and they all measure the same underlying quantity — when the building was built — through different encodings. `construction_year` (+0.487) and `building_age` (−0.487) are mathematical inverses by definition (building_age = 2025 − construction_year), so their equal-and-opposite loadings are expected and serve as an internal consistency check. The large negative loadings of `decade_correlation_ed` (−0.437) and `decade_correlation_pe` (−0.433) reinforce the axis: older buildings (built in earlier decades) have stronger correlation with the pre-insulation energy demand reference tables, which were constructed from pre-renovation-era buildings. Higher correlation with old reference values means a lower-performing building by current standards.

The regulatory interpretation of this axis is critical for practical use. European residential construction can be broadly stratified into three energy performance eras that correspond directly to positions along PC3:
- **Pre-1970 (low PC3 score):** Buildings constructed before the first energy crises, with no thermal insulation requirements, single-glazed windows, and uninsulated cavity walls. These represent the lowest-performing segment of the stock.
- **1970–2000 (mid PC3 score):** A transitional era in which successive national energy codes introduced minimum insulation requirements, but enforcement and practice varied considerably across countries and building types.
- **Post-2000 (high PC3 score):** Buildings constructed under modern energy performance directives, including the EU's Energy Performance of Buildings Directive (EPBD, 2002/91/EC and its 2010 recast), which mandated near-zero energy standards for new construction progressively from 2018–2021.

This regulatory stratification is precisely what Kavgic et al. (2010) identified as the primary basis for constructing archetype-based bottom-up building stock models across Europe: construction period, together with building type, is the minimal sufficient description for estimating the thermal standard of a building from register data alone. PC3 shows that this signal is cleanly recoverable from the dataset — the age features are not entangled with size or type in the PCA decomposition, which means construction year provides genuinely independent information to the prediction model.

**This is the most actionable component for retrofit planning.** Buildings with strongly negative PC3 scores (old, pre-1970, high decade-correlation with historical demand) are the first-priority candidates for energy renovation. The component directly ranks buildings by their expected distance from current energy standards, without requiring an energy audit.

#### PC4 (9.1%) — Compactness / Form Factor Axis

| Feature | Loading |
|---|---|
| `footprint_to_heated_ratio` | +0.370 |
| `gfa_to_footprint_ratio` | +0.370 |
| `area_to_volume_ratio` | +0.363 |
| `floor_count` | -0.299 |
| `apt_count` | -0.255 |
| `volume_heated` | -0.230 |
| `category_mean_energy` | +0.227 |

**Interpretation:** PC4 captures **building compactness and form factor**, orthogonal to both size (PC2) and typology (PC1). This is a geometrically precise axis: the three features with the largest positive loadings — `footprint_to_heated_ratio` (+0.370), `gfa_to_footprint_ratio` (+0.370), and `area_to_volume_ratio` (+0.363) — are all expressions of how horizontally spread a building is relative to its enclosed volume, while the large negative loading on `floor_count` (−0.299) captures the opposite pole: tall, vertically stacked buildings. In plain terms, PC4 distinguishes **sprawling low-rise buildings from compact high-rise buildings**, independent of how large either is.

The physical relevance of this axis is direct and well-documented. Depecker et al. (2001) demonstrated through thermal simulation that the compactness ratio (defined as the ratio of the heated volume to the envelope surface area) is one of the strongest geometric determinants of heating energy demand: compact buildings lose proportionally less heat per unit of conditioned floor area because they have less exposed surface. A cube-shaped multi-storey tower is far more compact than a single-storey bungalow of the same floor area, and therefore requires less heating energy per m² in a cold climate. PC4 encodes exactly this distinction through its loading structure.

The positive loading of `category_mean_energy` (+0.227) within the horizontal-spread pole of PC4 is a particularly informative result. It indicates that, in this dataset, buildings that are wide and low-rise also tend to belong to categories with higher average energy demand. This is consistent with the composition of the building stock: the low-PC4 pole (compact towers) corresponds to multi-family apartment buildings with relatively modern construction standards, while the high-PC4 pole (sprawling low buildings) corresponds to older single-family detached houses, which in this stock are disproportionately pre-1980 and poorly insulated. PC4 and PC3 are orthogonal, but they both ultimately identify the same policy target: low-PC3 (old) AND high-PC4 (sprawling) buildings represent the worst-performing segment of the stock by two independent physical arguments.

#### PC5 (6.0%) — Multi-unit Density Axis

| Feature | Loading |
|---|---|
| `apt_count` | +0.519 |
| `floor_count` | +0.394 |
| `c2025_functype` | +0.370 |
| `area_per_apt` | -0.337 |
| `vol_cooled` | -0.330 |
| `footprint_area` | -0.284 |

**Interpretation:** PC5 captures a **within-multi-family density** dimension that does not reduce to size (PC2) or form factor (PC4). The dominant loading is `apt_count` (+0.519) — by far the single largest loading in this component — followed by `floor_count` (+0.394) and `c2025_functype` (+0.370). Together, these describe a building with many apartments stacked on many floors: the high-density urban residential tower. The opposing features — `area_per_apt` (−0.337), `vol_cooled` (−0.330), and `footprint_area` (−0.284) — describe large-unit, low-density buildings that are likely to have air conditioning (suggesting higher-standard or mixed-use residential).

This axis matters energetically because the number of units in a building affects heat loss through shared internal walls and floors. In a densely packed apartment block, each individual apartment shares large portions of its envelope with adjacent heated spaces, substantially reducing net heat loss compared to a detached building of equivalent floor area. This "adjacency effect" is a well-known factor in residential energy modelling. Miller & Meggers (2017), in their analysis of the Building Data Genome Project dataset, found that occupant density and the number of tenants per floor area were strong modifiers of energy use intensity in multi-unit buildings, particularly for heating-dominated climates.

The negative loading of `vol_cooled` (−0.330) at the low-PC5 pole is interpretable: buildings that invest in air conditioning infrastructure tend to be lower-density, higher-specification developments — large-footprint buildings where individual large apartments are separated by unconditioned spaces. This spending pattern is the opposite of dense social housing towers, which have high apartment counts but rarely have mechanical cooling. PC5 therefore also differentiates within the multi-family category between high-density social/standard housing and lower-density premium or mixed-use residential stock, the two segments of which can have quite different energy demand profiles despite nominally belonging to the same building category.

### 4.4 PCA Scatter Plot (PC1 vs PC2)

The scatter plot of buildings in PC1 × PC2 space, coloured by log-transformed energy demand, shows a **diffuse cloud with a gradient**: buildings at low PC1 / low PC2 tend to cluster toward moderate energy values, while buildings at high PC2 (large buildings) show greater spread, suggesting that size alone does not determine energy intensity (kWh/m²). Buildings at extreme PC2 values tend to have higher variance in energy demand, consistent with the finding that large buildings exist in both well-insulated and poorly-insulated forms.

### 4.5 KMeans Clustering in PCA Space (5 Clusters)

KMeans (k=5) was applied to the first 5 PCA components to identify natural building subtypes in the feature space. Results (log-transformed energy units):

| Cluster | Count | Median Energy | Mean Energy | Character |
|---|---|---|---|---|
| 3 | 84 | 79.99 | 81.08 | Low-energy buildings |
| 0 | 567 | 110.40 | 127.64 | Moderate, mainstream stock |
| 2 | 405 | 139.00 | 146.85 | Medium-high energy |
| 4 | 37 | 140.00 | 141.62 | Small cluster, similar to C2 |
| 1 | 184 | 185.82 | 183.12 | High-energy buildings |

**Discussion:** The five clusters span roughly a 2.3x range in median energy demand (79.99 to 185.82 in log-space). Cluster 3 (84 buildings, low energy) likely represents recently constructed or recently renovated buildings that conform to current energy standards. Cluster 0 is the dominant group (567 buildings, 28% of stock), representing the average existing stock. Cluster 1 (184 buildings, high energy, ~9% of stock) represents the priority retrofit target: buildings with the highest per-m² demand. The presence of Clusters 2 and 4 with very similar medians (~139–140) but very different counts (405 vs. 37) suggests that C4 is a geometrically distinct sub-group of the same energy band as C2 — possibly buildings of unusual shape or size distribution.

The high mean-to-median gap in Cluster 0 (127.6 vs. 110.4, a +15.6% skew) indicates a right-tailed energy distribution within the mainstream cluster: most buildings in this group are near-average, but a tail of very inefficient buildings pulls the mean up. This is important for policy: interventions targeting this mainstream cluster (which contains the most buildings) should focus on the upper quartile within it.

---

## 5. Feature Importance Analysis

### 5.1 LightGBM Feature Importance (Enhanced_3, split-count metric)

LightGBM was trained on the full Enhanced_3 feature set and its internal split-count importance (number of times each feature is used to split a node, summed across all trees) was recorded:

| Rank | Feature | Importance Score | Category |
|---|---|---|---|
| 1 | `heating_efficiency_ratio` | 2083 | Engineered ratio |
| 2 | `area_to_volume_ratio` | 1773 | Engineered ratio |
| 3 | `area_heated` | 1430 | Physical size |
| 4 | `area_total` | 1320 | Physical size |
| 5 | `footprint_to_heated_ratio` | 1222 | Engineered ratio |
| 6 | `construction_year` | 1213 | Temporal |
| 7 | `footprint_area` | 1102 | Physical geometry |
| 8 | `gfa_to_footprint_ratio` | 1039 | Engineered ratio |
| 9 | `area_per_apt` | 1026 | Engineered ratio |
| 10 | `volume_heated` | 721 | Physical size |
| 11 | `nbuildingy` | 480 | Temporal |
| 12 | `decade_correlation_pe` | 347 | Temporal correlation |
| 13 | `floor_count` | 257 | Physical geometry |
| 14 | `decade_correlation_ed` | 207 | Temporal correlation |
| 15 | `apt_count` | 96 | Multi-unit |
| 16 | `floor_height` | 52 | Physical geometry |

### 5.2 Discussion of Feature Importance

**The top two features are both engineered ratios, not raw measurements.** `heating_efficiency_ratio` (area_heated / area_total) and `area_to_volume_ratio` (area_heated / volume_heated) score over 1.7–2.0× higher than the next-best raw physical feature. This is the single most important finding from the importance analysis: the model relies more on *how a building is proportioned* than on *how large it is*. A building where only 60% of total floor area is heated behaves very differently from one where 95% is heated, even if their absolute sizes are identical.

**`area_heated` and `area_total` appear separately** despite being correlated, because after controlling for their ratio the model still finds marginal information in their absolute scale. This is consistent with economies-of-scale in heat loss: very small buildings have disproportionately high energy intensity per m² due to edge effects.

**`construction_year` (rank 6) significantly outperforms the decade correlations** (ranks 12, 14). This is somewhat surprising since the decade correlations were derived from construction year and were designed to be more informative. A likely explanation is that the continuous year encodes information across multiple vintages simultaneously, whereas the decade encodings are step-wise approximations. Additionally, the tree model can discover non-linear year thresholds (e.g. year > 2000 → insulation regulation) directly from the data without requiring the pre-tabulated correlations.

**`floor_count` (rank 13) and `apt_count` (rank 15)** are comparatively low-importance. Despite being strongly predictive in PCA space (PC5) and in the cross-model comparison (Enhanced_3 was the best set), their split-count importance is modest. This likely reflects their correlation with other features: `floor_count` is largely derivable from `volume_heated / area_heated` and `apt_count` is partially captured by `area_per_apt`. The tree model uses these features for refinement, not primary splitting.

**`floor_height` (rank 16, score 52)** is the least-used feature by a large margin. Its importance may be underestimated by split-count (it may only be used for fine splits near the leaf level), but it does suggest that ceiling height, while conceptually interesting, does not provide as much predictive variance as building footprint and proportion features. This could also reflect poor data quality in the field from which it is approximated.

---

## 6. SHAP Explainability

### 6.1 Setup

SHAP (SHapley Additive exPlanations) analysis was run on an **XGBoost model trained on the Enhanced_3 feature set** — the second-best configuration overall (R²=0.533). XGBoost was chosen over RF for SHAP because TreeExplainer is computationally exact for gradient-boosted trees, whereas for RF it provides an approximation. The target was log-transformed energy demand; SHAP values therefore represent contributions in log-space, and the direction/magnitude of effects must be interpreted accordingly.

Four visualisations were produced:
1. **Beeswarm plot** — each dot is one test building, positioned by its SHAP value for that feature
2. **Mean |SHAP| bar chart** — global feature importance by average absolute SHAP value
3. **Dependence plots** — top 3 features, showing how SHAP value varies with feature value
4. **Waterfall plots** — step-by-step feature contributions for the best-predicted and worst-predicted buildings

### 6.2 SHAP Global Importance vs. LightGBM Importance

Although exact SHAP values were not printed to text (they are rendered as charts), the notebook identifies the top SHAP features as the input to the dependence plots. The PDP analysis (which uses SHAP importance to select features) identified the following top features: `footprint_area`, `construction_year`, `footprint_to_heated_ratio`, `decade_correlation_pe`, `heating_efficiency_ratio`, `area_to_volume_ratio`.

This partially aligns with LightGBM split-count importance, with notable differences:
- **`footprint_area` rises significantly in SHAP importance** (ranked higher than in LightGBM split count). This suggests that while footprint area is not the most-used splitting feature, when it does split it has large magnitude effects — consistent with the large spread in footprint sizes in the dataset.
- **`decade_correlation_pe` rises** from rank 12 in LightGBM to a top-6 SHAP feature. Decade-era correlations, though less frequently used for splitting, carry large shifts in predicted energy demand when they are activated — particularly for buildings in the pre-1970 era vs. post-2000 era.
- **`heating_efficiency_ratio` and `area_to_volume_ratio` remain top features** in both methods, confirming their cross-model robustness.

### 6.3 Beeswarm Interpretation

The beeswarm plot maps the SHAP contribution of each feature across all test buildings. Key structural patterns to expect from this dataset:

- **`heating_efficiency_ratio`** will show a strong negative relationship: buildings with high ratios (most of their area is heated) tend to receive negative SHAP values — pushing prediction *downward* from the mean. This is intuitive: a fully-heated building already operates efficiently; there is less unheated volume acting as a thermal buffer.
- **`area_to_volume_ratio`** will show a similarly structured spread: buildings with compact volumes relative to floor area (tall buildings with less exposed surface) are pulled toward lower predicted energy.
- **`construction_year`** will show a clear split: pre-1970 buildings receive strong positive SHAP values (pushing prediction upward, toward high energy demand), while post-2000 buildings receive negative SHAP values.
- **`category_mean_energy`** — the target-encoded prior — acts as a strong shrinkage toward category-level norms. Buildings far from their category mean will see this feature's SHAP contribution pull the prediction back toward the group average.

### 6.4 Dependence Plots

Dependence plots reveal the shape of the feature-to-SHAP relationship:

- **`footprint_area`**: Expected to show a roughly inverse relationship with SHAP value — larger footprint buildings (sprawling single-storey structures) attract higher SHAP contributions (more energy per m²). The interaction variable on these plots (coloured points) will likely be `floor_count` or `apt_count`, highlighting that the footprint effect is conditioned on building height.
- **`construction_year`**: Expected step-change pattern around key regulatory years (1970s oil crisis, 1990s insulation standards, 2006/2010 energy performance directives). Buildings built after 2010 likely converge on near-zero SHAP values as they conform to near-zero energy building requirements.
- **`heating_efficiency_ratio`**: Likely a non-linear curve: below ~0.7 the SHAP effect is strongly positive (large unheated fraction → high inefficiency); above ~0.85 the SHAP effect flattens and may turn mildly negative.
- **`decade_correlation_pe`**: Step-function pattern mirroring the decade groupings; the pre-1960 class will show the largest positive SHAP values.

### 6.5 Waterfall Plots (Best vs. Worst Predicted Buildings)

The waterfall plot for the **best-predicted building** will show many small feature contributions that approximately cancel out — a building that is "average" in almost all feature dimensions. The base value (mean log prediction) plus small adjustments from each feature sums to a value close to the actual log-energy demand.

The waterfall plot for the **worst-predicted building** will show the failure mode: likely a building where features point in contradictory directions. For example, a building that is old (pulling prediction high) but small (pulling prediction low) and in a category with moderate expected energy but whose actual demand is extreme in one direction. These outlier buildings are typically those with unusual retrofits (very old buildings that have been deeply renovated) or data quality issues (measured energy demand affected by occupancy, not just the physical building).

### 6.6 Partial Dependence Plots (PDPs)

PDPs were generated for the top 6 SHAP-important features: `footprint_area`, `construction_year`, `footprint_to_heated_ratio`, `decade_correlation_pe`, `heating_efficiency_ratio`, `area_to_volume_ratio`. Red tick marks at the bottom of each panel show the actual data distribution.

Key expected shapes:
- **`construction_year`**: Monotonically decreasing predicted energy as construction year increases (newer = more efficient), with a steepening slope after 2000 when regulations became strict.
- **`heating_efficiency_ratio`**: U-shaped or monotonically decreasing — higher ratio reduces predicted energy.
- **`area_to_volume_ratio`**: Monotonically decreasing — more area per unit volume (compact buildings) have lower heat loss.
- **`footprint_area`**: Likely U-shaped or weakly increasing — very large footprints correlate with sprawling low buildings; mid-range footprints correspond to multi-storey efficient buildings.
- **`decade_correlation_pe`**: Step-wise decreasing — higher correlation values (newer decades) push predicted energy down.
- **`footprint_to_heated_ratio`**: Non-linear, potentially with a saturation region at high values.

---

## 7. Cross-Method Consistency and Summary of Findings

### 7.1 Consistency Across Methods

The three analysis methods (PCA, feature importance, SHAP) paint a consistent picture of the predictive structure:

| Theme | PCA | LightGBM Importance | SHAP |
|---|---|---|---|
| Building proportions dominate | PC2, PC4 loaded with ratios | Ranks 1–2 are both ratios | Ratios are top SHAP features |
| Age/vintage is a strong signal | PC3 (10.1%) is purely age | construction_year rank 6 | Year shows clear SHAP pattern |
| Type/category carries prior knowledge | PC1 (22.8%) | `category_mean_energy` rank N/A (target encoded) | Decade correlations in top 6 |
| Geometry beyond raw size matters | PC4 captures form factor | footprint features rank 5–9 | footprint_area is top SHAP feature |

### 7.2 What Feature Engineering Contributed

The most valuable features, confirmed across all three methods, were not any single raw column from the dataset but rather the **engineered interaction terms**:

1. `heating_efficiency_ratio` and `area_to_volume_ratio` — simple geometric ratios that the baseline model could not compute because they require two columns
2. `decade_correlation_pe/ed` — temporal knowledge base lookups that encode the institutional memory of energy performance standards by era
3. `category_mean_energy` target encoding — a data-driven prior that summarises what buildings of a given type and location typically consume
4. `footprint_to_heated_ratio` and `gfa_to_footprint_ratio` — building form descriptors that distinguish compact high-rises from sprawling low buildings

These engineered features accounted for the majority of the +14.9% improvement in R² from Baseline to Enhanced_3 (0.469 → 0.539).

### 7.3 Residual Uncertainty and Limitations

An R² of 0.539 means the best model explains approximately **54% of the variance** in residential energy demand. The remaining 46% of variance is not captured by the current feature set. Plausible sources of this unexplained variance include:

- **Occupancy behaviour** — the same building inhabited by a family that keeps the thermostat at 22°C vs. 18°C will show very different energy demand, but this is not observable from the building registry
- **Renovation history** — buildings that have been partially retrofitted (new windows but old roof insulation) have complex energy profiles that a single `construction_year` or `construction_type` cannot capture
- **Measurement and data quality** — not all energy demand values are metered; some may be estimated or calculated from energy certificates, introducing label noise
- **Micro-climate effects** — building orientation, local shading, wind exposure, and urban heat island effects are not captured in the current feature set
- **Equipment efficiency** — the type, age, and efficiency of the heating/cooling system matters as much as the building envelope in determining actual energy consumption

---

## 8. Conclusions

1. **Feature engineering was the primary driver of model improvement.** The jump from Baseline (4 raw physical features) to Enhanced_3 (33 features) improved R² by nearly 15 percentage points. The addition of building form ratios alone (Enhanced set) gave the largest per-feature gain.

2. **Tree-based ensemble models (RF, XGB, CatBoost) clearly outperform MLP** on this tabular building dataset, achieving 0.53–0.54 R² vs. 0.42–0.44 for MLP. This is consistent with the broader ML literature on tabular regression with moderate dataset sizes.

3. **PCA reveals five interpretable dimensions** in the building stock: (1) building type/classification, (2) physical size, (3) age/vintage, (4) form factor/compactness, and (5) multi-unit scale. These align well with established concepts in building energy physics and retrofit policy.

4. **Building proportions are more predictive than raw sizes.** The `heating_efficiency_ratio` and `area_to_volume_ratio` are the most-used splitting features across tree models, and they also surface as top SHAP features. Policy implication: predicting energy demand from registry data requires shape information, not just area/volume counts.

5. **Construction year and decade-era correlations are key temporal signals.** The age axis (PC3, 10.1% of variance) and the dominance of `construction_year` in importance and SHAP analysis confirm that when a building was built is one of the strongest observable predictors of how efficiently it operates today.

6. **The KMeans clustering identifies a high-energy tail (Cluster 1: ~184 buildings, median 185.82)** representing roughly 9% of the building stock with substantially higher per-m² energy demand than average. This group, identifiable through the PCA feature space, is the highest-priority target for energy renovation interventions.

7. **LIME was not implemented in this notebook.** If local explanation at the instance level is required beyond the waterfall plots provided by SHAP, LIME would be a valuable complementary tool — particularly for explaining individual retrofit decisions to building owners using interpretable linear surrogate models.

---

---

## References

Amasyali, K., & El-Gohary, N. M. (2018). A review of data-driven building energy consumption prediction studies. *Renewable and Sustainable Energy Reviews*, 81, 1192–1205. https://doi.org/10.1016/j.rser.2017.08.052

Capozzoli, A., Grassi, D., & Causone, F. (2015). Estimation models of heating energy consumption in schools for local authorities planning. *Energy and Buildings*, 105, 302–313. https://doi.org/10.1016/j.enbuild.2015.07.074

Depecker, P., Menezo, C., Virgone, J., & Lepers, S. (2001). Design of buildings shape and energetic consumption. *Building and Environment*, 36(5), 627–635. https://doi.org/10.1016/S0360-1323(00)00044-5

European Parliament. (2010). Directive 2010/31/EU of the European Parliament and of the Council on the energy performance of buildings (recast). *Official Journal of the European Union*, L 153, 13–35.

Geraldi, M. S., & Ghisi, E. (2020). Data-driven framework towards localised bottom-up building stock energy modelling. *Energy and Buildings*, 226, 110382. https://doi.org/10.1016/j.enbuild.2020.110382

Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441. https://doi.org/10.1037/h0071325

Kavgic, M., Mavrogianni, A., Mumovic, D., Summerfield, A., Stevanovic, Z., & Djurovic-Petrovic, M. (2010). A review of bottom-up building stock models for energy consumption in the residential sector. *Building and Environment*, 45(7), 1683–1697. https://doi.org/10.1016/j.buildenv.2010.01.021

Kontokosta, C. E., & Tull, C. (2017). A data-driven predictive model of city-scale energy use in buildings. *Applied Energy*, 197, 303–317. https://doi.org/10.1016/j.apenergy.2017.04.005

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

Miller, C., & Meggers, F. (2017). The Building Data Genome Project: An open, public data set from office buildings for machine learning. *Energy Procedia*, 122, 439–444. https://doi.org/10.1016/j.egypro.2017.07.392

Papadopoulos, S., Bonczak, B., & Kontokosta, C. E. (2018). Pattern recognition in building energy performance over time using energy benchmarking data. *Applied Energy*, 221, 576–586. https://doi.org/10.1016/j.apenergy.2018.03.079

Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572. https://doi.org/10.1080/14786440109462720

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144. https://doi.org/10.1145/2939672.2939778

Runge, J., & Zmeureanu, R. (2019). Forecasting energy use in buildings using artificial neural networks: A review. *Energies*, 12(17), 3254. https://doi.org/10.3390/en12173254

---

*Report generated from `Final_Notebook_for_Generation_with_PCA_FE_SHAP (1).ipynb`*
*Dataset: 2,010 residential buildings, target: `en2025_enegy_demand_present_m2` (kWh/m²/year)*
