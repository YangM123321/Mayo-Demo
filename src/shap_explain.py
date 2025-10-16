# src/shap_explain.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, shap
import matplotlib.pyplot as plt

OUT = Path("out"); MODELS = Path("models"); PLOTS = OUT / "shap"
PLOTS.mkdir(parents=True, exist_ok=True)

# ---- Load features like training ----
df = pd.read_parquet(OUT / "labs_curated.parquet")
feat = (df.pivot_table(index=["patient_id","encounter_id"],
                       columns="loinc", values="lab_value", aggfunc="mean")
          .reset_index().rename_axis(None, axis=1)).fillna(0.0)
X_df = feat[["2345-7","718-7"]] if {"2345-7","718-7"}.issubset(feat.columns) else feat.iloc[:,2:]
X = X_df.values
feature_names = list(X_df.columns)

# ---- Load model ----
rf = joblib.load(MODELS / "admit_rf.joblib")

# ---- Compute SHAP ----
explainer = shap.TreeExplainer(rf)
sv_raw = explainer.shap_values(X)      # can be list or ndarray with 2 or 3 dims
ev_raw = explainer.expected_value      # scalar, 1D, or more

def to_scalar_base(ev):
    """Return a scalar base value (take the last element if needed)."""
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev = np.array(ev)
        ev = np.ravel(ev)[-1]
    return float(ev)

def pick_2d_shap(sv, n_samples, n_features):
    """
    Normalize shap values to shape (n_samples, n_features).
    Handles:
      - list of arrays (pick last output)
      - ndarray (n_samples, n_features)
      - ndarray (n_outputs, n_samples, n_features)
      - ndarray (n_samples, n_outputs, n_features)
    """
    if isinstance(sv, list):
        arr = np.asarray(sv[-1])  # pick the "positive" / last output
    else:
        arr = np.asarray(sv)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Case A: (n_outputs, n_samples, n_features)
        if arr.shape[1] == n_samples and arr.shape[2] == n_features:
            return arr[-1, :, :]  # last output
        # Case B: (n_samples, n_outputs, n_features)
        if arr.shape[0] == n_samples and arr.shape[2] == n_features:
            return arr[:, -1, :]  # last output
        # Last resort: try to squeeze and reshape if it collapses
        arr2 = np.squeeze(arr)
        if arr2.ndim == 2 and arr2.shape == (n_samples, n_features):
            return arr2
        raise ValueError(f"Unexpected SHAP shape {arr.shape} (cannot map to (samples, features))")
    # Fallback for odd cases
    arr2 = np.squeeze(arr)
    if arr2.ndim == 2:
        return arr2
    raise ValueError(f"Unexpected SHAP ndim={arr.ndim}")

# Figure expected (samples, features)
n_samples = X.shape[0]
n_features = X.shape[1]
sv_all = pick_2d_shap(sv_raw, n_samples, n_features)
base = to_scalar_base(ev_raw)

# ---- Build Explanation objects ----
exp_all = shap.Explanation(
    values=sv_all,                      # (n_samples, n_features)
    base_values=np.full(n_samples, base),
    data=X,
    feature_names=feature_names,
)

# --- 1) Global importance: robust bar plot ---
plt.figure()
shap.plots.bar(exp_all, show=False, max_display=n_features)
plt.tight_layout()
plt.savefig(PLOTS / "shap_global_bar.png", dpi=150)
plt.close()

# --- 2) Per-sample explanation ---
i = 0  # first row
sv0 = sv_all[i].astype(float)          # (n_features,)
x0 = X[i]
exp_i = shap.Explanation(
    values=sv0,
    base_values=base,
    data=x0,
    feature_names=feature_names,
)

# Try modern waterfall; if it fails, draw a simple bar instead
saved_waterfall = True
try:
    plt.figure()
    shap.plots.waterfall(exp_i, show=False, max_display=n_features)
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_waterfall_first.png", dpi=150)
    plt.close()
except Exception as e:
    saved_waterfall = False
    # Fallback: plain bar chart of SHAP values
    plt.figure()
    order = np.argsort(np.abs(sv0))[::-1]
    plt.barh([feature_names[j] for j in order], sv0[order])
    plt.gca().invert_yaxis()
    plt.title("SHAP (fallback) - first sample")
    plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_waterfall_first.png", dpi=150)
    plt.close()
    (PLOTS / "shap_waterfall_first.fallback.txt").write_text(str(e))

# --- 3) Optional HTML force plot (best effort, non-blocking) ---
saved_force = True
try:
    force_obj = shap.plots.force(exp_i, show=False)   # JS/HTML
    shap.save_html(str(PLOTS / "shap_force_first.html"), force_obj)
except Exception as e:
    saved_force = False
    (PLOTS / "shap_force_first.ERROR.txt").write_text(str(e))

print("Wrote:")
print(" ", PLOTS / "shap_global_bar.png")
print(" ", PLOTS / "shap_waterfall_first.png", "(fallback used)" if not saved_waterfall else "")
print(" ", PLOTS / "shap_force_first.html" if saved_force else "  (force plot skipped; see ERROR.txt)")
