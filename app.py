import os
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pycaret.classification import load_model, predict_model
import shap
from typing import Optional, List, Dict, Any


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

# Re-use your prepare_inputs, FEATURES, BIN_LIKE

# ---- OHE helpers (same logic you used in notebook/plotly helpers) ----
import re
def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", str(s)).strip().lower()

def _ohe_group_cols(base_name: str, cols: List[str]) -> List[str]:
    base_n = _norm(base_name)
    group = []
    for c in cols:
        if "_" in c:
            left = c.rsplit("_", 1)[0]
            if _norm(left) == base_n:
                group.append(c)
    return group

def group_shap_for_feature(name: str, shap_df: pd.DataFrame) -> pd.Series:
    if name in shap_df.columns:
        return shap_df[name]
    group_cols = _ohe_group_cols(name, shap_df.columns.tolist())
    if group_cols:
        return shap_df[group_cols].sum(axis=1)
    # lenient match
    nm = _norm(name)
    for c in shap_df.columns:
        if _norm(c) == nm:
            return shap_df[c]
    raise ValueError(f"Cannot map feature '{name}' to SHAP columns.")

def reconstruct_category_from_ohe(base_name: str, X: pd.DataFrame) -> pd.Series:
    cols = _ohe_group_cols(base_name, X.columns.tolist())
    if not cols:
        if base_name in X.columns:
            return X[base_name].astype("category")
        return pd.Series([None]*len(X), index=X.index, dtype="category", name=base_name)
    levels = [c.rsplit("_", 1)[1] for c in cols]
    M = X[cols].to_numpy()
    row_sums = M.sum(axis=1)
    best = M.argmax(axis=1)
    labels = pd.Series(np.array(levels, dtype=object)[best], index=X.index, name=base_name)
    labels[row_sums == 0] = None
    return labels.astype("category")

def ensure_feature_series(name: Optional[str], X: pd.DataFrame) -> Optional[pd.Series]:
    if name is None:
        return None
    if name in X.columns:
        return X[name]
    s = reconstruct_category_from_ohe(name, X)
    if s.isna().all():
        nm = _norm(name)
        for c in X.columns:
            if _norm(c) == nm:
                return X[c]
        raise ValueError(f"{name} not found (nor reconstructable from one-hot).")
    return s

# ---- SHAP cache ----
_SHAP_CACHE: Dict[str, Any] = {
    "ready": False,
    "X_explain": None,             # pd.DataFrame (transformed features)
    "shap_vals_kernel": None,      # np.ndarray [n_samples, n_features]
    "feat_cols": None,             # list[str]
}

def _build_kernel_shap_cache():
    """Compute & cache Kernel SHAP for the blended proba over the TRAINED dataset."""
    if _SHAP_CACHE["ready"]:
        return

    # Load the same csv and preprocessing you already used in training (mirrors your notebook),
    # but minimal: we just need the transformed X and the models available.
    raw = pd.read_csv("data.csv")
    use_cols = [c for c in (FEATURES + [target_col]) if c in raw.columns]
    raw = raw[use_cols].copy()
    raw = raw[~raw[target_col].isna()].copy()

    from pycaret.classification import setup, get_config, create_model, tune_model, calibrate_model
    _ = setup(
        data=raw, target=target_col, session_id=42, fold=3,
        fold_strategy="stratifiedkfold", fix_imbalance=True, use_gpu=False, verbose=False
    )

    X_train_df = pd.DataFrame(get_config("X_train_transformed"), columns=[c for c in get_config("dataset_transformed").columns if c != get_config("target_param")])
    X_test_df  = pd.DataFrame(get_config("X_test_transformed"),  columns=X_train_df.columns)
    X_explain  = pd.concat([X_train_df, X_test_df], axis=0).reset_index(drop=True)

    # Re-create calibrated ridge + tuned catboost exactly as before
    ridge = create_model("ridge")
    ridge_cal = calibrate_model(ridge, method="isotonic")

    cat = create_model("catboost")
    cat_t = tune_model(cat, optimize="F1")

    # unwrap cat for proba
    cat_est = cat_t
    for attr in ["trained_model", "trained_model_", "model", "estimator"]:
        if hasattr(cat_est, attr):
            cat_est = getattr(cat_est, attr)

    def proba_blend(X):
        Xd = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=X_explain.columns)
        p_cat = cat_est.predict_proba(Xd)[:, 1]
        p_rid = ridge_cal.predict_proba(Xd)[:, 1]
        return 0.5 * (p_cat + p_rid)

    # background (balanced, small)
    y_train_t = get_config("y_train_transformed")
    pos_idx = np.where(np.asarray(y_train_t) == 1)[0]
    neg_idx = np.where(np.asarray(y_train_t) == 0)[0]
    n_pos = min(len(pos_idx), 15); n_neg = min(len(neg_idx), 15)
    bg_idx = np.concatenate([np.random.choice(pos_idx, n_pos, replace=False),
                             np.random.choice(neg_idx, n_neg, replace=False)])
    background = X_train_df.iloc[bg_idx].reset_index(drop=True)

    explainer = shap.KernelExplainer(model=proba_blend, data=background)
    shap_vals_kernel = explainer.shap_values(X_explain, nsamples="auto")

    _SHAP_CACHE.update({
        "ready": True,
        "X_explain": X_explain,
        "shap_vals_kernel": shap_vals_kernel,
        "feat_cols": X_explain.columns.tolist(),
    })

@app.get("/explain/features")
def explain_features():
    _build_kernel_shap_cache()
    cols = _SHAP_CACHE["feat_cols"]

    # heuristic: infer base names from OHE
    bases = {}
    for c in cols:
        if "_" in c:
            base = c.rsplit("_", 1)[0]
            bases.setdefault(base, set()).add(c)
    original_candidates = sorted(set(list(bases.keys()) + [c for c in cols if "_" not in c]))
    return {"encoded": cols, "original_like": original_candidates}

@app.get("/explain/summary")
def explain_summary():
    _build_kernel_shap_cache()
    X = _SHAP_CACHE["X_explain"]
    S = _SHAP_CACHE["shap_vals_kernel"]
    shap_df = pd.DataFrame(S, columns=X.columns)

    # encoded level
    enc = (np.abs(shap_df).mean(axis=0)
           .reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})) \
           .sort_values("mean_abs_shap", ascending=False)

    # grouped by base (sum of OHE columns)
    groups = {}
    for col in shap_df.columns:
        base = col.rsplit("_", 1)[0] if "_" in col else col
        groups.setdefault(base, 0.0)
        groups[base] += float(np.abs(shap_df[col]).mean())
    grp = pd.DataFrame({"feature": list(groups.keys()), "mean_abs_shap": list(groups.values())}) \
            .sort_values("mean_abs_shap", ascending=False)

    return {"encoded": enc.to_dict(orient="records"),
            "grouped": grp.to_dict(orient="records")}

@app.post("/explain/dependence")
def explain_dependence(payload: Dict[str, Any] = Body(...)):
    """
    Body: {
      "feature_x": str,                    # original or encoded
      "feature_color": str|null,
      "feature_size": str|null,
      "feature_symbol": str|null,
      "y_mode": "by_x"|"sum"
    }
    """
    _build_kernel_shap_cache()
    X = _SHAP_CACHE["X_explain"].copy()
    S = _SHAP_CACHE["shap_vals_kernel"]
    shap_df = pd.DataFrame(S, columns=X.columns)

    fx = payload.get("feature_x")
    fc = payload.get("feature_color")
    fs = payload.get("feature_size")
    fsy = payload.get("feature_symbol")
    y_mode = payload.get("y_mode", "by_x")

    if fx is None:
        raise HTTPException(status_code=400, detail="feature_x required")

    x_series = ensure_feature_series(fx, X)
    if y_mode == "sum":
        y_series = shap_df.sum(axis=1)
    else:
        y_series = group_shap_for_feature(fx, shap_df)

    out = pd.DataFrame({"x": x_series, "shap": y_series})

    if fc: out["color"] = ensure_feature_series(fc, X)
    if fs:
        tmp = ensure_feature_series(fs, X)
        out["size"] = (pd.Categorical(tmp).codes + 1) if not pd.api.types.is_numeric_dtype(tmp) else tmp
    if fsy: out["symbol"] = ensure_feature_series(fsy, X)

    # convert categories to strings for JSON
    for col in out.columns:
        if str(out[col].dtype) == "category":
            out[col] = out[col].astype(str)

    # small safety: drop NaNs rows to keep plotting robust
    out = out.replace({np.inf: np.nan, -np.inf: np.nan}).dropna(subset=["x", "shap"])

    return {"rows": out.to_dict(orient="records")}
