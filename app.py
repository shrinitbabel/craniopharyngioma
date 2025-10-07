import os
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pycaret.classification import load_model, predict_model

# --------- Load model once on startup ----------
MODEL_PATH = "permdi_blend_ridge_cat_v1"   # file is permdi_blend_ridge_cat_v1.pkl in repo root
model = load_model(MODEL_PATH)             # PyCaret adds .pkl automatically

FEATURES = [
    "Age ", "BMI", "Tumor size", "Tumor size.1", "Tumor size.2", "Tumor volume",
    "Last BMI on file",
    "Pre-op visual impairment (y/n) - symptomatic complaint of visual loss",
    "Pre-op hydrocephalus (y/n)", "Pre-op intacranial pressure symptoms",
    "Pre-op headache", "Pre-op endocrine changes (y/n)",
    "Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional",
    "Intraventricular extension",
    "Surgical Approach", "Resection outcome (total, subtotal, partial)"
]

BIN_LIKE = [
    "Pre-op visual impairment (y/n) - symptomatic complaint of visual loss",
    "Pre-op hydrocephalus (y/n)", "Pre-op intacranial pressure symptoms",
    "Pre-op headache", "Pre-op endocrine changes (y/n)",
    "Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional",
    "Intraventricular extension",
]

def prepare_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Recompute Tumor volume if missing/NaN but axes exist (ellipsoid)
    if "Tumor volume" not in df.columns and all(c in df.columns for c in ["Tumor size","Tumor size.1","Tumor size.2"]):
        df["Tumor volume"] = (4/3)*math.pi * (df["Tumor size"]/2)*(df["Tumor size.1"]/2)*(df["Tumor size.2"]/2)

    # Ensure expected columns exist
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan

    # Light coercion for binary-like cols
    for c in BIN_LIKE:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize categories
    if "Surgical Approach" in df.columns:
        df["Surgical Approach"] = df["Surgical Approach"].astype(str).str.strip().str.lower()
        df["Surgical Approach"] = df["Surgical Approach"].replace({
            "endo": "endoscopic",
            "endo.": "endoscopic"
        })

    if "Resection outcome (total, subtotal, partial)" in df.columns:
        df["Resection outcome (total, subtotal, partial)"] = (
            df["Resection outcome (total, subtotal, partial)"].astype(str).str.strip().str.lower()
        )
        df["Resection outcome (total, subtotal, partial)"] = df["Resection outcome (total, subtotal, partial)"].replace({
            "near-total": "near total",
            "near_total": "near total"
        })

    return df[FEATURES]

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Permanent DI Risk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://babels.ai",
        "https://www.babels.ai",
        "https://cranio.babels.ai",
        "https://cranio-api.babels.ai",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



FORM_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Permanent DI Risk</title>
<style>
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; margin: 2rem; }
label { display:block; margin-top:0.6rem; }
input, select { padding:0.3rem; width: 280px; }
button { margin-top:1rem; padding:0.5rem 1rem; }
.small { color:#666; font-size:0.9em; }
</style>
</head>
<body>
<h2>Permanent DI Risk (Pilot)</h2>
<form method="post" action="/predict-form">
  <label>Age <input type="number" name="Age " step="1" required></label>
  <label>BMI <input type="number" name="BMI" step="0.1"></label>
  <label>Tumor size (A) <input type="number" name="Tumor size" step="0.1" required></label>
  <label>Tumor size (B) <input type="number" name="Tumor size.1" step="0.1" required></label>
  <label>Tumor size (C) <input type="number" name="Tumor size.2" step="0.1" required></label>
  <label>Last BMI on file <input type="number" name="Last BMI on file" step="0.1"></label>
  <label>Pre-op visual impairment (0/1) <input type="number" name="Pre-op visual impairment (y/n) - symptomatic complaint of visual loss" min="0" max="1"></label>
  <label>Pre-op hydrocephalus (0/1) <input type="number" name="Pre-op hydrocephalus (y/n)" min="0" max="1"></label>
  <label>Pre-op intracranial pressure symptoms (0/1) <input type="number" name="Pre-op intacranial pressure symptoms" min="0" max="1"></label>
  <label>Pre-op headache (0/1) <input type="number" name="Pre-op headache" min="0" max="1"></label>
  <label>Pre-op endocrine changes (0/1) <input type="number" name="Pre-op endocrine changes (y/n)" min="0" max="1"></label>
  <label>Pre-op panhypopit (0/1) <input type="number" name="Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional" min="0" max="1"></label>
  <label>Intraventricular extension (0/1) <input type="number" name="Intraventricular extension" min="0" max="1"></label>
  <label>Surgical Approach
    <select name="Surgical Approach">
      <option value="craniotomy">craniotomy</option>
      <option value="endoscopic">endoscopic</option>
    </select>
  </label>
  <label>Resection outcome
    <select name="Resection outcome (total, subtotal, partial)">
      <option value="total">total</option>
      <option value="subtotal">subtotal</option>
      <option value="near total">near total</option>
    </select>
  </label>
  <div class="small">This pilot is research-only, not a clinical device.</div>
  <button type="submit">Predict</button>
</form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return FORM_HTML

@app.get("/health")
def health():
    return {"status": "ok"}

import traceback
from fastapi import HTTPException
from fastapi.responses import JSONResponse

import traceback
from fastapi.responses import JSONResponse

def _proba_pos_from_predict_model(out):
    # pick the positive-class score column
    score_cols = [c for c in out.columns if c.startswith("Score_")]
    if not score_cols:
        return None

    # common names PyCaret uses
    for key in ("Score_1", "Score_True", "Score_yes", "Score_positive"):
        if key in out.columns:
            return float(out.loc[0, key])

    # if exactly two score columns, prefer the one that's not Score_0
    if len(score_cols) == 2 and "Score_0" in score_cols:
        other = score_cols[0] if score_cols[1] == "Score_0" else score_cols[1]
        return float(out.loc[0, other])

    # last resort: take the max-prob score col (not ideal but safe)
    row = out.loc[0, score_cols]
    return float(row.max())

@app.post("/predict")
def predict_api(payload: dict = Body(...)):
    try:
        df = pd.DataFrame([payload])
        X = prepare_inputs(df)

        # 1) Try direct predict_proba on the PyCaret pipeline (best)
        try:
            p1 = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            # 2) Try PyCaret helper with raw scores
            out = predict_model(model, data=X, raw_score=True)
            p1 = _proba_pos_from_predict_model(out)
            if p1 is None:
                # 3) Margin -> sigmoid fallback
                margin = float(model.decision_function(X)[0])
                p1 = 1.0 / (1.0 + math.exp(-margin))

        # If you maintain a site-wide threshold:
        THRESH = float(os.getenv("THRESHOLD", 0.5))
        label = int(p1 >= THRESH)

        return {"prob_permanent_DI": p1, "label": label, "threshold": THRESH}

    except Exception as e:
        print("PREDICT ERROR:", e)
        traceback.print_exc()
        return JSONResponse(
            {
                "error": str(e),
                "received_keys": list(payload.keys()),
                "prepared_columns": list(X.columns) if "X" in locals() else None,
            },
            status_code=400,
        )

from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()
    raw = {k: (None if v == "" else v) for k, v in form.items()}

    # cast numerics; leave category strings alone
    cat_keys = {"Surgical Approach", "Resection outcome (total, subtotal, partial)"}
    payload = {}
    for k, v in raw.items():
        if v is None:
            payload[k] = None
            continue
        if k in cat_keys:
            payload[k] = str(v).strip()
        else:
            try:
                payload[k] = float(v)
            except:
                payload[k] = v  # fallback

    df = pd.DataFrame([payload])
    X = prepare_inputs(df)

    try:
        # same robust prediction logic as /predict
        try:
            p1 = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            out = predict_model(model, data=X, raw_score=True)
            p1 = _proba_pos_from_predict_model(out)
            if p1 is None:
                margin = float(model.decision_function(X)[0])
                p1 = 1.0 / (1.0 + math.exp(-margin))

        THRESH = float(os.getenv("THRESHOLD", 0.5))
        label = int(p1 >= THRESH)

        return HTMLResponse(
            f"<pre>Permanent DI probability: {p1:.3f}\nLabel (0/1): {label}\n</pre>"
        )

    except Exception as e:
        return HTMLResponse(
            f"<pre>Internal error: {e}</pre>", status_code=500
        )

# Optional: if someone hits /predict-form with GET, send them to the form
@app.get("/predict-form")
def predict_form_get():
    return RedirectResponse(url="/", status_code=303)
