# ============================================================
# SHAP Values & Partial Dependence Plots
# Best model: XGBoost on Enhanced_3 features
# Requires: pip install shap   (PDP works without it)
# Run AFTER cell 15 (helpers) and cell 19 (Enhanced_3 pipeline)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Check for shap --------------------------------------------------------
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("shap loaded OK")
except ImportError:
    print("shap not installed  ->  run:  pip install shap")
    print("PDP analysis will still run.")

# ---- Helper: build Enhanced_3 feature matrix --------------------------------
def _build_enhanced3(df, target_column):
    from sklearn.preprocessing import LabelEncoder
    global DECADE_CORRELATIONS_PE, DECADE_CORRELATIONS_ED
    try:
        _ = DECADE_CORRELATIONS_PE
    except NameError:
        DECADE_CORRELATIONS_PE, DECADE_CORRELATIONS_ED = \
            create_correlation_dictionaries(df)

    df_clean, raw_target, target_transformed, use_log = \
        prepare_target_data(df, target_column)
    n = len(df_clean)
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    train_mask = pd.Series(False, index=df_clean.index)
    train_mask.iloc[idx_tr] = True

    X = create_enhanced_3_features(df_clean, target_column, train_mask=train_mask)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    return (X, target_transformed, use_log, idx_tr, idx_te,
            list(X.columns), df_clean, raw_target)


# ============================================================
# SECTION A — SHAP VALUES
# ============================================================
def run_shap_analysis(df, target_column='en2025_enegy_demand_present_m2',
                      top_n=15):
    """
    Train XGBoost on Enhanced_3 and compute SHAP values.

    Plots
    -----
    1. Beeswarm  — full distribution of feature impacts across all test buildings
    2. Bar chart — mean |SHAP| (global importance)
    3. Dependence plots — how top-3 features interact with predictions
    4. Waterfall — feature contributions for the best and worst single prediction
    """
    if not SHAP_AVAILABLE:
        print("Install shap first:  pip install shap"); return None, None

    import xgboost as xgb

    X, y_t, use_log, idx_tr, idx_te, feat_names, df_clean, raw_target = \
        _build_enhanced3(df, target_column)
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y_t.iloc[idx_tr], y_t.iloc[idx_te]

    print("Training XGBoost...", end=" ", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_tr, y_tr)
    print("done")

    print("Computing SHAP values...", end=" ", flush=True)
    try:
        # Newer SHAP API (>=0.40) — returns Explanation object directly
        explainer = shap.Explainer(model, X_tr)
        shap_exp  = explainer(X_te)
        shap_arr  = shap_exp.values
        base_val  = float(np.mean(shap_exp.base_values))
    except Exception:
        # Fallback to TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_arr  = explainer.shap_values(X_te)
        base_val  = float(explainer.expected_value)
        shap_exp  = shap.Explanation(
            values=shap_arr,
            base_values=np.full(len(X_te), base_val),
            data=X_te.values,
            feature_names=feat_names
        )
    print("done")

    # ── Plot 1: Beeswarm ────────────────────────────────────────────────────
    print("\n[Plot 1/4] SHAP Beeswarm")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_arr, X_te, feature_names=feat_names,
                      max_display=top_n, show=False)
    plt.title(
        "SHAP Beeswarm — each dot is one building\n"
        "Position on x-axis = how much that feature pushed the prediction\n"
        "Colour = feature value (red=high, blue=low)",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

    # ── Plot 2: Mean |SHAP| bar ──────────────────────────────────────────────
    print("[Plot 2/4] SHAP mean |value| bar")
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_arr, X_te, feature_names=feat_names,
                      plot_type="bar", max_display=top_n, show=False)
    plt.title(
        "Mean |SHAP| — global feature importance\n"
        "(average absolute impact on prediction across all test buildings)",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

    # ── Plot 3: Dependence plots for top 3 features ─────────────────────────
    print("[Plot 3/4] SHAP dependence plots")
    mean_abs = pd.Series(np.abs(shap_arr).mean(axis=0), index=feat_names)
    top3 = mean_abs.sort_values(ascending=False).head(3).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, feat in zip(axes, top3):
        shap.dependence_plot(feat, shap_arr, X_te,
                             feature_names=feat_names, ax=ax, show=False)
        ax.set_title("SHAP dependence: " + feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("SHAP value\n(contribution to log energy demand)", fontsize=9)
    plt.suptitle(
        "SHAP Dependence Plots — Top 3 Features\n"
        "x = feature value  |  y = SHAP contribution  |  "
        "colour = most interacting feature",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

    # ── Plot 4: Waterfall for best & worst individual predictions ────────────
    print("[Plot 4/4] Waterfall plots")
    pred_te    = model.predict(X_te)
    abs_errors = np.abs(y_te.values - pred_te)
    best_idx   = int(np.argmin(abs_errors))
    worst_idx  = int(np.argmax(abs_errors))

    best_err   = abs_errors[best_idx]
    worst_err  = abs_errors[worst_idx]
    if use_log:
        best_err  = float(np.expm1(abs_errors[best_idx]))
        worst_err = float(np.expm1(abs_errors[worst_idx]))

    print("  Best  predicted building (error = {:.1f} kWh/m2)".format(best_err))
    shap.waterfall_plot(shap_exp[best_idx], max_display=12)
    plt.title("Waterfall — best predicted building  "
              "(error = {:.1f} kWh/m2)".format(best_err), fontsize=11)
    plt.tight_layout()
    plt.show()

    print("  Worst predicted building (error = {:.1f} kWh/m2)".format(worst_err))
    shap.waterfall_plot(shap_exp[worst_idx], max_display=12)
    plt.title("Waterfall — worst predicted building  "
              "(error = {:.1f} kWh/m2)".format(worst_err), fontsize=11)
    plt.tight_layout()
    plt.show()

    return model, shap_exp


# ============================================================
# SECTION B — PARTIAL DEPENDENCE PLOTS
# ============================================================
def run_pdp_analysis(df, target_column='en2025_enegy_demand_present_m2',
                     pdp_features=None, n_features=6, grid_resolution=50):
    """
    Partial Dependence Plots for the top features from XGBoost on Enhanced_3.

    For each feature, all other features are held at their training-set mean
    and the model's predicted energy demand is plotted as the feature varies.
    Red ticks at the bottom show the actual distribution of that feature in
    the training data (rug plot).
    """
    import xgboost as xgb
    from sklearn.inspection import partial_dependence

    X, y_t, use_log, idx_tr, idx_te, feat_names, _, _ = \
        _build_enhanced3(df, target_column)
    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    y_tr, y_te = y_t.iloc[idx_tr], y_t.iloc[idx_te]

    print("Training XGBoost...", end=" ", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_tr, y_tr)
    print("done")

    # Pick features: SHAP if available, else model feature importance
    if pdp_features is None:
        if SHAP_AVAILABLE:
            expl = shap.TreeExplainer(model)
            sv   = expl.shap_values(X_te)
            imp  = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
        else:
            imp = pd.Series(model.feature_importances_, index=feat_names)
        pdp_features = imp.sort_values(ascending=False).head(n_features).index.tolist()

    print("PDP features: " + str(pdp_features))

    ncols = min(3, len(pdp_features))
    nrows = int(np.ceil(len(pdp_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, pdp_features):
        feat_idx = feat_names.index(feat)
        pd_res   = partial_dependence(
            model, X_tr, features=[feat_idx],
            kind="average", grid_resolution=grid_resolution
        )
        grid_vals = pd_res["grid_values"][0]
        avg_pred  = pd_res["average"][0]

        # Back-transform log target so y-axis is real kWh/m2
        y_vals    = np.expm1(avg_pred) if use_log else avg_pred
        y_label   = "Predicted energy demand (kWh/m2)"

        ax.plot(grid_vals, y_vals, color="#2c3e50", linewidth=2.5)
        ax.fill_between(grid_vals, y_vals,
                        alpha=0.12, color="#2c3e50")

        # Rug: actual data distribution
        feat_data = X_tr.iloc[:, feat_idx].values
        y_rug     = np.full_like(feat_data,
                                 y_vals.min() - (y_vals.max() - y_vals.min()) * 0.07)
        ax.plot(feat_data, y_rug, "|",
                color="#e74c3c", alpha=0.25, markersize=5)

        # Reference line at mean prediction
        ax.axhline(y_vals.mean(), color="grey",
                   linestyle="--", linewidth=1, alpha=0.6,
                   label="Mean prediction")

        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title("PDP: " + feat, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(pdp_features):]:
        ax.set_visible(False)

    plt.suptitle(
        "Partial Dependence Plots — XGBoost on Enhanced_3\n"
        "Each plot shows average predicted energy demand as one feature varies\n"
        "(all other features fixed at their mean)  |  "
        "Red ticks = actual data distribution",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.show()
    return model


# ============================================================
# RUN
# ============================================================
print("=" * 60)
print("  SHAP ANALYSIS")
print("=" * 60)
xgb_model, shap_explanation = run_shap_analysis(overlapping_gdf_dataset, top_n=15)

print()
print("=" * 60)
print("  PARTIAL DEPENDENCE PLOTS")
print("=" * 60)
xgb_model_pdp = run_pdp_analysis(overlapping_gdf_dataset)
