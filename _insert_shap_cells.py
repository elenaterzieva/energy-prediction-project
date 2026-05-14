"""Insert SHAP and PDP cells into the notebook after PerCategoryPerf002."""
import json
import re

NB_PATH = "Final_Notebook_for_Generation_with_4th_set_of_features.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells")
print("Last 3 IDs:", [c.get("id", "?") for c in nb["cells"][-3:]])

# ── Markdown cell ────────────────────────────────────────────────────────────
md_source = [
    "## Model Explanation: SHAP Values & Partial Dependence Plots\n",
    "\n",
    "### SHAP Values (SHapley Additive exPlanations)\n",
    "\n",
    "SHAP values quantify each feature's contribution to a specific prediction. Grounded in game theory, they answer: *how much did this feature push the prediction above or below the average?*\n",
    "\n",
    "| Plot | What it shows |\n",
    "|---|---|\n",
    "| **Beeswarm** | Every dot = one test building. X-position = SHAP value (impact on prediction). Colour = feature value (red=high, blue=low). |\n",
    "| **Bar chart** | Mean absolute SHAP value per feature — classic global importance. |\n",
    "| **Dependence** | How one feature's SHAP value changes as the feature value changes. Colour = most-interacting feature. |\n",
    "| **Waterfall** | Step-by-step feature contributions for the single best and single worst predicted building. |\n",
    "\n",
    "### Partial Dependence Plots (PDP)\n",
    "\n",
    "PDPs show the *average* model prediction as one feature varies, with all other features held at their mean. They reveal the global shape of the relationship (linear, non-linear, saturation effects) in real kWh/m² units.\n",
    "\n",
    "Red tick marks at the bottom of each panel show the actual distribution of that feature in the training data, so you can distinguish regions with good coverage from extrapolated regions.\n",
    "\n",
    "> **Model used:** XGBoost on Enhanced_3 features (best-performing model).  \n",
    "> Requires `shap` for SHAP section: `pip install shap`"
]

md_cell = {
    "cell_type": "markdown",
    "id": "SHAPAnalysis001",
    "metadata": {},
    "source": md_source
}

# ── Code cell ────────────────────────────────────────────────────────────────
with open("_tmp_shap.py", encoding="utf-8") as f:
    raw = f.read()

# Split into lines preserving newlines for notebook format
code_lines = []
for line in raw.splitlines(keepends=True):
    code_lines.append(line)

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "SHAPAnalysis002",
    "metadata": {},
    "outputs": [],
    "source": code_lines
}

# ── Insert after PerCategoryPerf002 ─────────────────────────────────────────
last_id = nb["cells"][-1].get("id", "")
if last_id == "PerCategoryPerf002":
    nb["cells"].append(md_cell)
    nb["cells"].append(code_cell)
    print("Appended after PerCategoryPerf002 (last cell)")
else:
    # Find PerCategoryPerf002 and insert after it
    target_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == "PerCategoryPerf002":
            target_idx = i
            break
    if target_idx is not None:
        nb["cells"].insert(target_idx + 1, md_cell)
        nb["cells"].insert(target_idx + 2, code_cell)
        print(f"Inserted after index {target_idx}")
    else:
        nb["cells"].append(md_cell)
        nb["cells"].append(code_cell)
        print("WARNING: PerCategoryPerf002 not found, appended at end")

print(f"Notebook now has {len(nb['cells'])} cells")
print("Last 4 IDs:", [c.get("id", "?") for c in nb["cells"][-4:]])

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved.")
