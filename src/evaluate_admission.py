# src/evaluate_admission.py
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

OUT = Path("out")
MODELS = Path("models")
PLOTS = OUT / "ml_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(OUT / "labs_curated.parquet")
df["admit_label"] = ((df["loinc"]=="2345-7") & (df["lab_value"]>=150)) | ((df["loinc"]=="718-7") & (df["lab_value"]<11.5))
df["admit_label"] = df["admit_label"].astype(int)

feat = (df.pivot_table(index=["patient_id","encounter_id"],
                       columns="loinc",
                       values="lab_value",
                       aggfunc="mean")
          .reset_index()
          .rename_axis(None, axis=1)).fillna(0.0)
y = (df.groupby(["patient_id","encounter_id"])["admit_label"].max()
       .reindex(zip(feat["patient_id"], feat["encounter_id"]))
       .astype(int).values)

X = feat[["2345-7","718-7"]].values if {"2345-7","718-7"}.issubset(feat.columns) else feat.iloc[:,2:].values

lr = joblib.load(MODELS / "admit_lr.joblib")
rf = joblib.load(MODELS / "admit_rf.joblib")

def plot_curves(model, name):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:,1]
    else:
        # fallback to decision_function
        p = model.decision_function(X)

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}")
    plt.legend(loc="lower right")
    plt.savefig(PLOTS / f"roc_{name}.png", dpi=150)
    plt.close()

    # PR
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {name}")
    plt.legend(loc="lower left")
    plt.savefig(PLOTS / f"pr_{name}.png", dpi=150)
    plt.close()

plot_curves(lr, "logreg")
plot_curves(rf, "random_forest")
print(f"Wrote ROC/PR plots to {PLOTS}")
