# Feature Documentation
## Energy Demand Prediction Model — Bulgarian Buildings

**Target variable:** `en2025_enegy_demand_present_m2` — present-state energy demand in kWh/m2/year

---

## Feature Sets Overview

| Feature Set | # Features | Introduces |
|---|---|---|
| Baseline | 4 | Basic physical dimensions |
| Enhanced | 15 | Outlier handling, correlations, age, ratios, category |
| Enhanced_2 | 18 | Construction type, external year, building shape |
| Enhanced_3 | 32+ | Floor data, cooled volume, footprint, apartments, location encoding, history |

---

## Baseline Features (4 features)

All 4 features come directly from the 2025 Energy Performance Certificate dataset.

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `area_heated` | `en2025_area_heated_m2` | Direct read; European decimal format converted; NaN filled with 0 | Heated floor area of the building in m2. Strong direct driver of total energy consumption. |
| `area_total` | `en2025_gfa_m2` | Direct read; European decimal format converted; NaN filled with 0 | Gross floor area (total built area) in m2. Includes unheated spaces like staircases. |
| `volume_heated` | `en2025_vol_heated_m3` | Direct read; European decimal format converted; NaN filled with 0 | Total heated volume of the building in m3. Directly determines heating load. |
| `construction_year` | `en2025_yearbuilt` | Direct read; European decimal format converted; NaN filled with 1990 | Year the building was constructed. Older buildings typically have worse insulation and higher demand. |

---

## Enhanced Features (15 features)

Includes all Baseline features with improved outlier handling, plus 11 new features.

### Inherited from Baseline (with improvements)

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `area_heated` | `en2025_area_heated_m2` | IQR outlier capping (1.5×IQR), then median fill | Same as Baseline but extreme outliers clipped to reduce model noise. |
| `area_total` | `en2025_gfa_m2` | IQR outlier capping (1.5×IQR), then median fill | Same as Baseline but extreme outliers clipped. |
| `volume_heated` | `en2025_vol_heated_m3` | IQR outlier capping (1.5×IQR), then median fill | Same as Baseline but extreme outliers clipped. |
| `construction_year` | `en2025_yearbuilt` | European decimal conversion, NaN filled with column median | Same as Baseline but NaN filled with median instead of fixed 1990. |

### New in Enhanced

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `Correlation_ED` | `d_Correlation_Energy_Demand` | Direct read, NaN filled with 0 | Dataset-level correlation score between building attributes and energy demand. Captures data quality signal. |
| `Correlation_PM` | `d_Correlation_Primary_Energy` | Direct read, NaN filled with 0 | Correlation score between attributes and primary energy. Complements Correlation_ED. |
| `Energy_Completeness_Pct` | `d_Energy_Completeness_Pct` | Direct read, NaN filled with 0 | Percentage of energy-related fields that are complete for this building. Proxy for data quality. |
| `building_category` | `d_Category` | NaN filled with `"Other"` | Categorical label for building use (e.g., Residential One-Family, Multi-Family, Educational). One-hot encoded by tree models. |
| `construction_decade` | derived from `construction_year` | `floor(year / 10) * 10` | Decade of construction (1950, 1960, etc.). Captures era-specific construction norms without year-level noise. |
| `decade_correlation_pe` | `d_Correlation_Primary_Energy` + `d_Category` + `construction_decade` | Mean correlation per (category, decade) group, assigned to each building | Captures interaction between building type and construction era for primary energy. Reduces within-group variance. |
| `decade_correlation_ed` | `d_Correlation_Energy_Demand` + `d_Category` + `construction_decade` | Mean correlation per (category, decade) group, assigned to each building | Same as above but for energy demand. |
| `building_age` | derived from `construction_year` | `2025 - construction_year` | How old the building is in years. Direct proxy for insulation quality and energy code era. |
| `log_building_age` | derived from `building_age` | `log(1 + building_age)` (natural log) | Log-transformed age to compress the long tail. A 10-year-old vs 20-year-old matters more than 100 vs 110. |
| `heating_efficiency_ratio` | derived from `area_heated`, `area_total` | `area_heated / (area_total + 1)` | Fraction of total floor area that is heated. Low ratio means large unheated spaces; affects normalised energy. |
| `area_to_volume_ratio` | derived from `area_heated`, `volume_heated` | `area_heated / (volume_heated + 1)` | Inverse of average floor height. Taller floors mean more volume per m2 floor area, increasing heating load per m2. |

---

## Enhanced_2 Features (18 features)

Includes all Enhanced features plus 3 new columns from the external GeoData layer.

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `construction_type` | `construction_type` | NaN filled with `"Unknown"` | Structural construction method (e.g., reinforced concrete panel, masonry brick). Affects thermal mass and insulation capacity. |
| `nbuildingy` | `nbuildingy` | European decimal conversion; NaN filled with `construction_year` median | Building year from the external GIS/geodata layer. May differ from EPC year; provides a second year estimate. |
| `bldg_shape` | `_bldg_shape` | First matching column; NaN filled with `"Unknown"` | Footprint shape classification from building geometry (e.g., rectangular, L-shaped, complex). Compact shapes lose less heat. |

---

## Enhanced_3 Features (32+ features)

Includes all Enhanced_2 features plus 14+ new engineered features from previously unused columns.

### Floor Geometry

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `floor_count` | `c2025_flrcount` or `en2023_floorcount` | Direct read, clipped 1–40, NaN filled with median. If column absent: `volume_heated / (area_heated × 3.0)`, clipped 1–20. | Number of floors. High-rise buildings have different loss profiles from single-story ones. |
| `floor_height` | derived from `volume_heated`, `floor_count`, `area_heated` | `volume_heated / (floor_count × area_heated)`, clipped 2.0–6.0 m, NaN filled with 3.0 | Average storey height in metres. Taller ceilings increase heating volume per m2 floor area. |

### Cooling

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `vol_cooled` | `en2025_vol_cooled_m3` | Direct read, clipped ≥ 0, NaN filled with 0 | Volume of the building that has active cooling (m3). Indicates building systems complexity. |
| `vol_cooled_ratio` | derived from `vol_cooled`, `volume_heated` | `vol_cooled / (volume_heated + 1)` | Share of heated volume that is also cooled. Mixed-use energy profile indicator. |

### Footprint

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `footprint_area` | `en2025_fp_area_m2` | Direct read, clipped ≥ 1, NaN filled with column median | Ground-level footprint area in m2. Combined with floor_count, cross-validates floor area data. |
| `footprint_to_heated_ratio` | derived from `area_heated`, `footprint_area` | `area_heated / (footprint_area + 1)` | Approximate number of floors derived from floor areas (independent estimate from volume). |
| `gfa_to_footprint_ratio` | derived from `area_total`, `footprint_area` | `area_total / (footprint_area + 1)` | Ratio of gross area to footprint; another floor-count proxy using total area. |

### Apartment Count

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `apt_count` | `c2025_appcount` | Direct read, clipped ≥ 1, NaN filled with 1 | Number of residential apartments in the building. Distinguishes single-family from large multi-family buildings. |
| `area_per_apt` | derived from `area_heated`, `apt_count` | `area_heated / (apt_count + 1)` | Average heated area per apartment in m2. Large apartments in older blocks have different demand profiles. |

### Location Target Encoding

These features encode the average observed energy demand for each geographic group, computed on training data and applied as a lookup to the full dataset.

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `mun_mean_energy` | `en2025_mun` + target | Mean target per municipality on training data; global mean for unseen groups | Average energy demand in the building's municipality. Captures climate zone and local construction norms. |
| `oblast_mean_energy` | `en2025_oblast` + target | Mean target per oblast (province) on training data; global mean for unseen groups | Average energy demand in the building's province/oblast. Broader regional climate signal. |
| `city_mean_energy` | `en2025_city` + target | Mean target per city on training data; global mean for unseen groups | Average energy demand in the building's city. Finer-grained than oblast; captures urban heat island, dense vs rural patterns. |

### Category Target Encoding

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `category_mean_energy` | `d_Category` + target | Mean target per building category on training data; global mean for unseen groups | Average energy demand for buildings of this use category. Captures systematic differences between residential, educational, commercial, etc. |

### Historical Energy

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `hist_energy_2023` | `en2023_enegy_demand_present_m2` | Direct read, NaN filled with column median. Only added if ≥ 50 buildings have valid 2023 values. | Energy demand from the 2023 EPC in kWh/m2. Best single predictor when available — same building measured 2 years prior. |
| `has_hist_energy` | derived from `en2023_enegy_demand_present_m2` | 1 if 2023 value is not null and > 0, else 0 | Binary flag indicating whether 2023 historical data exists. Allows model to learn different behaviour for buildings with/without history. |

### Function Type

| Feature Name | Source Column | Computation | What It Represents |
|---|---|---|---|
| `c2025_functype` (or `c2025_funccode` / `c2025_proptype`) | `c2025_functype` preferred, else `c2025_funccode`, else `c2025_proptype` | First available column; NaN filled with `"Unknown"` | Granular function/use classification (finer than `d_Category`). Distinguishes offices, schools, hospitals, warehouses, etc. |
| `en2025_type` | `en2025_type` | NaN filled with `"Unknown"` | Building type code from 2025 EPC. Indicates structural typology used in EPC methodology. |

---

## Target Variable

| Column | Description |
|---|---|
| `en2025_enegy_demand_present_m2` | Present-state energy demand in kWh/m2/year from the 2025 Energy Performance Certificate. This is the value being predicted. |

---

## Models Used Per Feature Set

| Model | Baseline | Enhanced | Enhanced_2 | Enhanced_3 |
|---|---|---|---|---|
| Random Forest (RF) | Yes | Yes | Yes | Yes |
| XGBoost (XGB) | Yes | Yes | Yes | Yes |
| MLP (Neural Net) | Yes | Yes | Yes | — |
| LSTM | Yes | Yes | Yes | — |
| TabNet | Yes | Yes | Yes | — |
| TabPFN | Yes | Yes | Yes | — |
| LightGBM (LGB) | — | — | — | Yes |
| CatBoost (CB) | — | — | — | Yes |

---

## Notes on Encoding

- **European decimal format**: Some numeric columns use commas as decimal separators (e.g., `1,5` instead of `1.5`). The `convert_european_numbers()` function handles this conversion.
- **Target encoding**: Location and category features use leave-out mean encoding computed on training data only. Unseen groups (inference only) receive the global training mean to avoid data leakage.
- **Categorical features**: Models that cannot handle strings natively (RF, XGB, MLP, LSTM, TabNet, TabPFN) receive one-hot encoded or label-encoded versions internally. LightGBM and CatBoost handle categorical columns natively.
