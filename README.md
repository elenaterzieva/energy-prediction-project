# Energy Consumption Prediction for Buildings

A machine learning project for predicting building energy demand using geospatial data and multiple ML/DL model architectures. Includes CPU vs. GPU performance benchmarking across hardware accelerators.

## Overview

This project trains and evaluates four model types — Random Forest, XGBoost, MLP, and LSTM — on a dataset of ~2,100 Bulgarian buildings enriched with geospatial, structural, and energy performance features. It compares three feature sets (Baseline, Enhanced, Enhanced_2) and benchmarks training/inference performance across CPU and three NVIDIA GPU tiers.

**Best result:** Random Forest with Enhanced features — R² = 0.462, MAE = 38.05 kWh/m²

## Project Structure

```
energy-prediction-project/
├── Final_Notebook_for_Generation.ipynb          # Main workflow: training, inference, export
├── Final_Notebook_Phase_1_with_Monitoring.ipynb # Extended version with performance monitoring
└── figures-tables-comparison/
    ├── kfold_avg_*.csv          # Averaged k-fold cross-validation results
    ├── kfold_raw_*.csv          # Raw per-fold results
    ├── SUMMARY_*.csv            # Best models by R² and training time
    ├── VIZ_*.csv                # Full metrics for visualization
    ├── *.png                    # Performance charts and heatmaps
    ├── *.tex                    # LaTeX-formatted tables for publication
    ├── gifs/                    # Animated model comparison GIFs
    └── mixed_plots/             # Mixed-metric comparison plots
```

## Requirements

The project runs in **Google Colab**. Install dependencies by running the first notebook cell:

```bash
pip install contextily geopandas matplotlib rasterio tensorflow xgboost optuna
```

**Core dependencies:**
- `tensorflow` / Keras — LSTM model
- `scikit-learn` — Random Forest, MLP, preprocessing
- `xgboost` — gradient boosting
- `geopandas` / `shapely` — geospatial data handling
- `optuna` — hyperparameter optimization
- `contextily`, `rasterio` — map visualization
- `matplotlib`, `seaborn` — plotting

## Data

Input datasets are loaded from Google Drive and are not included in this repository:

| File | Format | Description |
|------|--------|-------------|
| `buildings_SO_2025_enriched_shape_neighbour_energy_panels_nsiyear_18092025.geojson` | GeoJSON | Building geometries with 50+ energy and structural features |
| `merged_energy_data_points_fixed_matching_kk_16092025.gpkg` | GeoPackage | Matched energy measurement points |

**Target variable:** `en2025_enegy_demand_present_m2` (energy demand in kWh/m²)

To use your own data, update the file paths in **Module 2** of the notebook:
```python
gdf = gpd.read_file('/content/drive/MyDrive/GATE/Energy Consumption/Final_datasets/<your_file>')
```

## Models

| Model | Library | Best R² | Best MAE | Notes |
|-------|---------|---------|---------|-------|
| Random Forest | scikit-learn | 0.462 | 38.05 kWh | Best overall accuracy |
| XGBoost | xgboost | 0.446 | — | Competitive with RF |
| MLP | scikit-learn | 0.381 | — | Fastest training (~0.3s) |
| LSTM | Keras | 0.311 | — | Highest compute cost (15–25s) |

All models are evaluated with K-fold cross-validation. Hyperparameters are tuned via `RandomizedSearchCV` (RF/XGB) and manual grid search (LSTM).

## Feature Sets

Three feature sets are compared:

- **Baseline** — core building attributes: area, volume, year built, type, location
- **Enhanced** — baseline + derived correlations and encoded category mappings
- **Enhanced_2** — alternative enhanced feature set

Enhanced features improve R² by +0.12 to +0.30 over Baseline across all models.

## Hardware Benchmarking

Models were benchmarked on four hardware configurations:

| Accelerator | VRAM | Max GPU Speedup vs CPU |
|-------------|------|------------------------|
| CPU | — | baseline |
| NVIDIA Tesla T4 | ~960 MB | ~1.2× |
| NVIDIA L4 | ~2,500 MB | ~1.8× |
| NVIDIA A100-SXM4-80GB | ~5,547 MB | ~2.4× |

Key finding: GPU speedups are modest (<2.5×) for this dataset size. CPU remains competitive, especially for Random Forest and XGBoost.

## Outputs

Results are saved to `figures-tables-comparison/`:

- **CSV files** — raw and averaged k-fold metrics per model/feature set/accelerator
- **PNG charts** — tradeoff curves, heatmaps, bar charts
- **GIF animations** — animated model comparisons across accelerators
- **LaTeX tables** — publication-ready performance summaries

## Building Type Mapping

Bulgarian `functype` codes are mapped to English categories for model training:

- Residential: one-family houses, multi-family buildings
- Garages and parking structures
- Mixed-use buildings
- Commercial, office, and service buildings

The mapping is defined in `functype_to_english` in Module 3 of the notebook.
