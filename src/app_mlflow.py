# src/app_mlflow.py
from pathlib import Path
from typing import Optional, Dict, Any
import os, traceback

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import mlflow, mlflow.pyfunc
import joblib

APP_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_ROOT / "models"
JOBLIB_FALLBACK = MODELS_DIR / "admit_mlflow_lr.joblib"

# --- MLflow store (same DB you used during training/registration) ---
MLFLOW_DB_URI = "sqlite:///mlflow.db"
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_DB_URI)
os.environ.setdefault("MLFLOW_REGISTRY_URI", MLFLOW_DB_URI)

MODEL_URI = "models:/admission_lr@champion"
FEATURE_ORDER = ["2345-7", "718-7"]   # must match training pivot order

app = FastAPI(title="Mayo Demo – MLflow Model API", version="1.1.0")

class AdmissionRequest(BaseModel):
    loinc_2345_7: Optional[float] = Field(None, description="Glucose (mg/dL)")
    loinc_718_7:  Optional[float] = Field(None, description="Hemoglobin (g/dL)")

def _load_model() -> Dict[str, Any]:
    """
    Try MLflow registry first; if that fails, fall back to local joblib.
    Return a dict with {model, source, note}.
    """
    # 1) MLflow registry
    try:
        m = mlflow.pyfunc.load_model(MODEL_URI)
        return {"model": m, "source": MODEL_URI, "note": "mlflow.pyfunc"}
    except Exception as e:
        ml_err = f"MLflow load failed: {e}"

    # 2) Local joblib fallback
    try:
        if JOBLIB_FALLBACK.exists():
            m = joblib.load(JOBLIB_FALLBACK)
            return {"model": m, "source": str(JOBLIB_FALLBACK), "note": "joblib fallback"}
        else:
            jb_err = f"Joblib file not found: {JOBLIB_FALLBACK}"
            raise RuntimeError(f"{ml_err} ; {jb_err}")
    except Exception as e:
        raise RuntimeError(f"{ml_err} ; Joblib fallback failed: {e}")

@app.on_event("startup")
def startup_load():
    app.state.model_bundle = _load_model()

@app.get("/health")
def health():
    b = app.state.model_bundle
    return {"ok": True, "model_source": b["source"], "note": b["note"], "features": FEATURE_ORDER}

@app.get("/model-info")
def model_info():
    # best-effort peek at pyfunc metadata
    info = {"model_source": app.state.model_bundle["source"], "features": FEATURE_ORDER}
    try:
        m = app.state.model_bundle["model"]
        meta = getattr(m, "metadata", None)
        if meta and hasattr(meta, "flavors"):
            info["flavors"] = list(meta.flavors.keys())
    except Exception:
        pass
    return info

@app.post("/predict/admission")
def predict(inp: AdmissionRequest):
    try:
        f_glu = float(inp.loinc_2345_7 or 0.0)
        f_hgb = float(inp.loinc_718_7  or 0.0)
        row = pd.DataFrame([[f_glu, f_hgb]], columns=FEATURE_ORDER)

        model = app.state.model_bundle["model"]

        # Prefer predict_proba (sklearn). For pyfunc, try predict and coerce.
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(row)[:, 1][0])
        else:
            yhat = model.predict(row)
            # if returns probabilities or scores, try to coerce; else cast to 0/1
            yhat = np.asarray(yhat).ravel()
            if yhat.size and 0.0 <= float(yhat[0]) <= 1.0:
                proba = float(yhat[0])
            else:
                proba = float(yhat[0] >= 0.5)

        label = int(proba >= 0.5)
        return {
            "ok": True,
            "model_source": app.state.model_bundle["source"],
            "features": {"2345-7": f_glu, "718-7": f_hgb},
            "probability_admit": proba,
            "predicted_label": label,
            "details": {"threshold": 0.5, "feature_order": FEATURE_ORDER},
        }
    except Exception as e:
        # Log full traceback to server console and return readable error
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

