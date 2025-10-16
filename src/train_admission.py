
# src/train_admission.py
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

OUT = Path("out")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

# ---- load curated data & create demo label ----
df = pd.read_parquet(OUT / "labs_curated.parquet")
# demo rule: high glucose >=150 OR low hgb <11.5 -> positive
df["admit_label"] = (
    ((df["loinc"] == "2345-7") & (df["lab_value"] >= 150)) |
    ((df["loinc"] == "718-7")  & (df["lab_value"] < 11.5))
).astype(int)

# wide features by LOINC per (patient_id, encounter_id)
feat = (df.pivot_table(index=["patient_id","encounter_id"],
                       columns="loinc",
                       values="lab_value",
                       aggfunc="mean")
          .reset_index()
          .rename_axis(None, axis=1)
          .fillna(0.0))

# y: max label per encounter
y = (df.groupby(["patient_id","encounter_id"])["admit_label"]
        .max()
        .reindex(list(zip(feat["patient_id"], feat["encounter_id"])))
        .astype(int)
        .values)

# X: use known LOINCs if present, else all numeric columns after the first two ids
feature_cols = ["2345-7", "718-7"] if {"2345-7","718-7"}.issubset(feat.columns) else feat.columns.tolist()[2:]
X = feat[feature_cols].values

# ---- tiny-data safe split logic ----
counts = Counter(y)
min_class = min(counts.values()) if len(counts) > 0 else 0
n = len(y)

if n < 4 or min_class < 2:
    # Not enough data to stratify/split reliably → train on all data
    print(f"[tiny-data mode] n={n}, class counts={dict(counts)}. Training on ALL data; no holdout.")
    X_train, y_train = X, y
    X_test, y_test = X, y
    stratify = None
else:
    stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

# ---- train models ----
lr = LogisticRegression(max_iter=500, n_jobs=None)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ---- evaluate (best-effort for tiny sets) ----
def safe_report(name, model):
    try:
        y_pred = model.predict(X_test)
        print(f"{name} report:\n", classification_report(y_test, y_pred, zero_division=0))
    except Exception as e:
        print(f"{name} evaluation skipped ({e})")

safe_report("LogReg", lr)
safe_report("RandomForest", rf)

# ---- save artifacts ----
joblib.dump(lr, MODELS / "admit_lr.joblib")
joblib.dump(rf, MODELS / "admit_rf.joblib")
with open(MODELS / "feature_list.json", "w") as f:
    json.dump(feature_cols, f)

print("Saved: models/admit_lr.joblib, models/admit_rf.joblib, models/feature_list.json")
