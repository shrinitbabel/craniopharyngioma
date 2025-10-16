import os
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pycaret.classification import load_model, predict_model


# --------- Load model once on startup ----------
MODEL_PATH = "permdi_blend_ridge_cat_v1"   # file is permdi_blend_ridge_cat_v1.pkl in repo root
RECURRENCE_MODEL_PATH = "recurrence_nb_calibrated_v1"
model = load_model(MODEL_PATH)             # PyCaret adds .pkl automatically
recurrence = load_model(RECURRENCE_MODEL_PATH)

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

# --- SHAP helpers (global) ---
from typing import List, Optional, Dict, Any

def _proba_pipeline_fn(X_in: pd.DataFrame) -> np.ndarray:
    """
    Positive-class proba for the loaded PyCaret pipeline on ORIGINAL columns.
    We run your prepare_inputs -> model.predict_proba.
    """
    Xp = prepare_inputs(X_in)
    return model.predict_proba(Xp)[:, 1]


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
<head>
  <meta charset="utf-8">
  <title>Permanent DI Risk</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; margin: 2rem; max-width: 980px; }
    label { display:block; margin-top:0.6rem; }
    input, select { padding:0.3rem; width: 280px; }
    button { margin-top:1rem; padding:0.5rem 1rem; }
    .small { color:#666; font-size:0.9em; }
    .row { display:flex; gap:1.25rem; flex-wrap:wrap; }
    .card { border:1px solid #e5e5e5; border-radius:8px; padding:1rem; margin-top:1rem; background:#fafafa; }
    .ok { color:#065f46; }
    .err { color:#b91c1c; }
  </style>
  <!-- Plotly CDN -->
  <script src="https://cdn.plot.ly/plotly-2.26.2.min.js"></script>
</head>
<body>

<h2>Permanent DI Risk (Pilot)</h2>

<form id="predict-form">
  <div class="row">
    <div>
      <label>Age <input type="number" name="Age " step="1" required></label>
      <label>BMI <input type="number" name="BMI" step="0.1" required></label>
      <label>Tumor size (A) <input type="number" name="Tumor size" step="0.1" required></label>
      <label>Tumor size (B) <input type="number" name="Tumor size.1" step="0.1" required></label>
      <label>Tumor size (C) <input type="number" name="Tumor size.2" step="0.1" required></label>
      <label>Last BMI on file <input type="number" name="Last BMI on file" step="0.1"></label>
    </div>

    <div>
      <label>Pre-op visual impairment (0/1)
        <input type="number" name="Pre-op visual impairment (y/n) - symptomatic complaint of visual loss" min="0" max="1" value="0">
      </label>
      <label>Pre-op hydrocephalus (0/1)
        <input type="number" name="Pre-op hydrocephalus (y/n)" min="0" max="1" value="0">
      </label>
      <label>Pre-op intracranial pressure symptoms (0/1)
        <input type="number" name="Pre-op intacranial pressure symptoms" min="0" max="1" value="0">
      </label>
      <label>Pre-op headache (0/1)
        <input type="number" name="Pre-op headache" min="0" max="1" value="0">
      </label>
      <label>Pre-op endocrine changes (0/1)
        <input type="number" name="Pre-op endocrine changes (y/n)" min="0" max="1" value="0">
      </label>
      <label>Pre-op panhypopit (0/1)
        <input type="number" name="Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional" min="0" max="1" value="0">
      </label>
      <label>Intraventricular extension (0/1)
        <input type="number" name="Intraventricular extension" min="0" max="1" value="0">
      </label>
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
    </div>
  </div>

  <div class="small">This pilot is research-only, not a clinical device.</div>
  <button type="submit">Predict</button>
</form>

<div id="msg" class="small" style="margin-top:6px;"></div>

<!-- Results -->
<div id="results" class="card" style="display:none;"></div>

<!-- SHAP controls -->
<div id="shap-card" class="card" style="display:none;">
  <h3 style="margin-top:0">Build a SHAP dependence plot</h3>
  <div class="row">
    <div>
      <label>X feature
        <select id="feat-x"></select>
      </label>
    </div>
    <div>
      <label>Color (optional)
        <select id="feat-color">
          <option value="">(none)</option>
        </select>
      </label>
    </div>
    <div>
      <label>Background size
        <input id="bg-size" type="number" min="1" max="60" value="25" style="width:100px;">
      </label>
    </div>
    <div>
      <label>nsamples
        <input id="nsamples" type="number" min="20" max="400" value="100" style="width:100px;">
      </label>
    </div>
  </div>
  <button id="btn-shap">Build plot</button>
  <div id="shap-msg" class="small" style="margin-top:6px;"></div>
  <div id="plot" style="height:520px; margin-top:10px;"></div>
</div>

<script>
  // Keep these in sync with your backend FEATURES (exact spelling, including "Age ")
  const ALL_FEATURES = [
    "Age ", "BMI", "Tumor size", "Tumor size.1", "Tumor size.2", "Tumor volume",
    "Last BMI on file",
    "Pre-op visual impairment (y/n) - symptomatic complaint of visual loss",
    "Pre-op hydrocephalus (y/n)", "Pre-op intacranial pressure symptoms",
    "Pre-op headache", "Pre-op endocrine changes (y/n)",
    "Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional",
    "Intraventricular extension",
    "Surgical Approach", "Resection outcome (total, subtotal, partial)"
  ];

  const BIN_LIKE = new Set([
    "Pre-op visual impairment (y/n) - symptomatic complaint of visual loss",
    "Pre-op hydrocephalus (y/n)", "Pre-op intacranial pressure symptoms",
    "Pre-op headache", "Pre-op endocrine changes (y/n)",
    "Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional",
    "Intraventricular extension",
  ]);

  const form = document.getElementById("predict-form");
  const results = document.getElementById("results");
  const msg = document.getElementById("msg");
  const shapCard = document.getElementById("shap-card");
  const shapMsg = document.getElementById("shap-msg");
  const featX = document.getElementById("feat-x");
  const featColor = document.getElementById("feat-color");
  const plotDiv = document.getElementById("plot");
  const bgInput = document.getElementById("bg-size");
  const nsInput = document.getElementById("nsamples");

  // populate selects
  function fillFeatureSelects() {
    featX.innerHTML = "";
    featColor.innerHTML = '<option value="">(none)</option>';
    for (const f of ALL_FEATURES) {
      const o1 = document.createElement("option");
      o1.value = f; o1.textContent = f;
      featX.appendChild(o1);

      const o2 = document.createElement("option");
      o2.value = f; o2.textContent = f;
      featColor.appendChild(o2);
    }
    // a useful default
    featX.value = "BMI";
    featColor.value = "";
  }
  fillFeatureSelects();

  // Read form -> payload (numbers except categories)
  function readPayload() {
    const fd = new FormData(form);
    const cats = new Set(["Surgical Approach","Resection outcome (total, subtotal, partial)"]);
    const payload = {};
    for (const [k, vRaw] of fd.entries()) {
      const v = (vRaw || "").toString().trim();
      if (cats.has(k)) payload[k] = v || null;
      else payload[k] = (v === "" ? null : Number(v));
    }
    // If Last BMI empty, mirror BMI
    if (payload["BMI"] != null && (payload["Last BMI on file"] == null || isNaN(payload["Last BMI on file"]))) {
      payload["Last BMI on file"] = payload["BMI"];
    }
    // ensure all bin fields at least exist
    for (const b of BIN_LIKE) {
      if (!(b in payload)) payload[b] = 0;
    }
    return payload;
  }

  // make N jittered rows around current payload (±8% for numerics)
  function makeCohortRows(base, n=30) {
    const rows = [];
    for (let i=0; i<n; i++) {
      const r = {};
      for (const k of Object.keys(base)) {
        const v = base[k];
        if (typeof v === "number" && Number.isFinite(v)) {
          const noise = (Math.random()-0.5)*0.16; // ±8%
          r[k] = v + v*noise;
        } else {
          r[k] = v;
        }
      }
      rows.push(r);
    }
    return rows;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.textContent = "Predicting…";
    shapMsg.textContent = "";
    results.style.display = "none";
    shapCard.style.display = "none";

    const payload = readPayload();

    try {
      // Predict permanent DI
      const r1 = await fetch("/predict", {
        method: "POST",
        headers: {"content-type":"application/json"},
        body: JSON.stringify(payload)
      });
      const d1 = await r1.json();
      if (!r1.ok || d1.error) throw new Error(d1.error || "Prediction failed");

      // Predict recurrence (optional, but you asked to show both earlier)
      const r2 = await fetch("/predict-recurrence", {
        method: "POST",
        headers: {"content-type":"application/json"},
        body: JSON.stringify(payload)
      });
      const d2 = await r2.json();

      // Show results
      results.innerHTML =
        "<div><b>Permanent DI probability:</b> " + (d1.prob_permanent_DI ?? 0).toFixed(3) + "</div>" +
        "<div><b>Permanent DI label (0/1):</b> " + (d1.label ?? 0) + "</div>" +
        "<div><b>Permanent DI threshold:</b> " + (d1.threshold ?? 0.5) + "</div>" +
        "<hr>" +
        "<div><b>Recurrence probability:</b> " + (d2.prob_recurrence ?? 0).toFixed(3) + "</div>" +
        "<div><b>Recurrence label (0/1):</b> " + (d2.label ?? 0) + "</div>" +
        "<div><b>Recurrence threshold:</b> " + (d2.threshold ?? 0.5) + "</div>";
      results.style.display = "block";
      msg.textContent = "";

      // Store last payload & reveal SHAP builder
      window.__lastPayload = payload;
      shapCard.style.display = "block";

    } catch (err) {
      console.error(err);
      msg.innerHTML = '<span class="err">' + (err?.message || String(err)) + '</span>';
    }
  });

  document.getElementById("btn-shap").addEventListener("click", async () => {
    shapMsg.textContent = "Building SHAP plot…";
    const payload = window.__lastPayload;
    if (!payload) { shapMsg.innerHTML = '<span class="err">Submit a prediction first.</span>'; return; }

    const rows = makeCohortRows(payload, Math.max(5, Math.min(60, Number(bgInput.value)||25)));
    const body = {
      rows,
      feature_x: featX.value,
      feature_color: featColor.value || undefined,
      y_mode: "by_x",
      nsamples: Number(nsInput.value)||100,
      background_size: Number(bgInput.value)||25
    };

    try {
      const r = await fetch("/shap/dependence", {
        method: "POST",
        headers: {"content-type":"application/json"},
        body: JSON.stringify(body)
      });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || "SHAP failed");

      const trace = {
        x: data.x,
        y: data.shap,
        mode: "markers",
        type: "scatter",
        marker: {
          size: 8,
          color: data.color || undefined,
          showscale: !!data.color
        },
        hovertemplate: featX.value + "=%{x}<br>SHAP=%{y:.3f}<extra></extra>"
      };
      Plotly.newPlot(plotDiv, [trace], {
        title: "SHAP dependence: " + featX.value,
        xaxis: { title: featX.value },
        yaxis: { title: "SHAP value (Permanent DI)" },
        margin: { l: 60, r: 20, t: 50, b: 50 }
      }, {responsive:true});

      shapMsg.innerHTML = '<span class="ok">Done.</span>';
    } catch (err) {
      console.error(err);
      shapMsg.innerHTML = '<span class="err">' + (err?.message || String(err)) + '</span>';
    }
  });
</script>

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

def _predict_pos_proba_safe(any_model, X: pd.DataFrame) -> float:
    """
    Best-effort positive-class probability from a PyCaret pipeline or sklearn model.
    Tries predict_proba, then predict_model(raw_score), then decision_function->sigmoid.
    """
    import math
    from pycaret.classification import predict_model

    # 1) Direct predict_proba
    try:
        proba = any_model.predict_proba(X)
        # assume positive class is column 1 when binary
        return float(proba[:, 1][0])
    except Exception:
        pass

    # 2) PyCaret helper (raw scores expose Score_* columns)
    try:
        out = predict_model(any_model, data=X, raw_score=True)
        p = _proba_pos_from_predict_model(out)
        if p is not None:
            return float(p)
    except Exception:
        pass

    # 3) decision_function -> sigmoid (fallback)
    try:
        margin = float(any_model.decision_function(X)[0])
        return 1.0 / (1.0 + math.exp(-margin))
    except Exception:
        # As a last resort, call predict() and map to {0,1}
        try:
            y = any_model.predict(X)[0]
            return float(y) if y in (0, 1, True, False) else float(y == 1)
        except Exception as _:
            raise RuntimeError("Unable to compute probability for the given model and inputs.")


@app.post("/predict")
def predict_api(payload: dict = Body(...)):
    try:
        df = pd.DataFrame([payload])
        X = prepare_inputs(df)

        p1 = _predict_pos_proba_safe(model, X)

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

@app.post("/predict-recurrence")
def predict_recurrence_api(payload: dict = Body(...)):
    try:
        df = pd.DataFrame([payload])
        X = prepare_inputs(df)

        p_rec = _predict_pos_proba_safe(recurrence, X)
        THRESH_R = float(os.getenv("THRESHOLD_RECURRENCE", 0.5))
        label_rec = int(p_rec >= THRESH_R)

        return {
            "prob_recurrence": p_rec,
            "label": label_rec,
            "threshold": THRESH_R,
        }

    except Exception as e:
        print("RECURRENCE PREDICT ERROR:", e)
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
from fastapi.responses import HTMLResponse, RedirectResponse

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
        elif k in cat_keys:
            payload[k] = str(v).strip()
        else:
            try:
                payload[k] = float(v)
            except Exception:
                payload[k] = v  # fallback

    df = pd.DataFrame([payload])
    X = prepare_inputs(df)

    # thresholds (env or defaults)
    THRESH_DI = float(os.getenv("THRESHOLD", 0.5))
    THRESH_REC = float(os.getenv("THRESHOLD_RECURRENCE", 0.5))

    # compute Permanent DI
    di_block = ""
    try:
        p_di = _predict_pos_proba_safe(model, X)
        label_di = int(p_di >= THRESH_DI)
        di_block = (
            f"Permanent DI probability: {p_di:.3f}\n"
            f"Permanent DI label (0/1): {label_di}\n"
            f"Permanent DI threshold: {THRESH_DI}\n"
        )
    except Exception as e:
        di_block = f"Permanent DI: error computing ({e})\n"

    # compute Recurrence
    rec_block = ""
    try:
        p_rec = _predict_pos_proba_safe(recurrence, X)
        label_rec = int(p_rec >= THRESH_REC)
        rec_block = (
            f"Recurrence probability: {p_rec:.3f}\n"
            f"Recurrence label (0/1): {label_rec}\n"
            f"Recurrence threshold: {THRESH_REC}\n"
        )
    except Exception as e:
        rec_block = f"Recurrence: error computing ({e})\n"

    return HTMLResponse(f"<pre>{di_block}\n{rec_block}</pre>")


# Optional: keep this so GET /predict-form redirects to the input form
@app.get("/predict-form")
def predict_form_get():
    return RedirectResponse(url="/", status_code=303)

@app.post("/shap/dependence")
def shap_dependence(payload: dict = Body(...)):
    """
    Body:
      rows: list[dict]   (REQUIRED)
      feature_x: str     (REQUIRED)
      feature_color: str (OPTIONAL)
      y_mode: "by_x" | "sum" (default "by_x")
      nsamples: "auto" | int (default "auto")
      background_size: int   (default = min(50, n_rows))
    """
    try:
        import shap  # lazy import so app still boots if SHAP missing
    except Exception:
        return JSONResponse(
            {"error": "SHAP not available on server. Add to requirements: shap==0.44.1, numba==0.57.1, llvmlite==0.40.1"},
            status_code=501,
        )

    from difflib import get_close_matches

    try:
        rows: List[Dict[str, Any]] = payload.get("rows") or []
        feature_x: Optional[str] = payload.get("feature_x")
        feature_color: Optional[str] = payload.get("feature_color")
        y_mode: str = payload.get("y_mode", "by_x")
        nsamples = payload.get("nsamples", "auto")
        bg_size: Optional[int] = payload.get("background_size")

        if not rows or not isinstance(rows, list):
            return JSONResponse({"error": "'rows' must be a non-empty list of payloads"}, status_code=400)
        if not feature_x or not isinstance(feature_x, str):
            return JSONResponse({"error": "'feature_x' is required"}, status_code=400)

        # Build DataFrame in ORIGINAL feature space (exact keys from client)
        X_raw = pd.DataFrame(rows)
        cols = list(X_raw.columns)

        # minor helper to normalize lookup (handle trailing spaces/case)
        def _find_col(name: str) -> Optional[str]:
            if name in cols:
                return name
            # try whitespace/case tolerant match
            target = name.strip().lower()
            for c in cols:
                if c.strip().lower() == target:
                    return c
            # try fuzzy
            m = get_close_matches(name, cols, n=1)
            return m[0] if m else None

        fx = _find_col(feature_x)
        if not fx:
            return JSONResponse({"error": f"feature_x '{feature_x}' not found in provided rows"}, status_code=400)

        fc = None
        if feature_color:
            fc = _find_col(feature_color)
            if not fc:
                return JSONResponse({"error": f"feature_color '{feature_color}' not in rows"}, status_code=400)

        # Background for KernelExplainer: small sample of submitted rows
        n = len(X_raw)
        if bg_size is None:
            bg_size = min(50, n)
        replace = bg_size > n
        bg = X_raw.sample(bg_size, random_state=42, replace=replace).reset_index(drop=True)

        # ---- model function that always receives a DataFrame with proper columns ----
        def proba_fn_local(X_in):
            if isinstance(X_in, np.ndarray):
                X_df = pd.DataFrame(X_in, columns=cols)
            else:
                X_df = X_in.reindex(columns=cols, fill_value=np.nan)
            Xp = prepare_inputs(X_df)
            return model.predict_proba(Xp)[:, 1]

        # --- SHAP computation (use all rows, no nsamples prompt) ---
        explainer = shap.KernelExplainer(model=proba_fn_local, data=X_raw)
        shap_vals = explainer.shap_values(X_raw)  # full dataset, no nsamples
        if isinstance(shap_vals, list):
            shap_vals = np.array(shap_vals)


        # Build explainer and compute SHAP
        explainer = shap.KernelExplainer(model=proba_fn_local, data=bg)
        shap_vals = explainer.shap_values(X_raw, nsamples=nsamples)
        if isinstance(shap_vals, list):  # some SHAP versions return list
            shap_vals = np.array(shap_vals)

        # y values
        if y_mode == "sum":
            y = np.abs(shap_vals).sum(axis=1).tolist()
        else:
            j = cols.index(fx)
            y = shap_vals[:, j].tolist()

        # x and optional color
        x = X_raw[fx].astype(object).where(~X_raw[fx].isna(), None).tolist()
        color = None
        if fc:
            color = X_raw[fc].astype(object).where(~X_raw[fc].isna(), None).tolist()

        meta = {
            "expected_value": (float(explainer.expected_value)
                               if hasattr(explainer, "expected_value") and np.isscalar(explainer.expected_value)
                               else None),
            "columns": cols,
            "y_mode": y_mode,
            "nsamples": nsamples,
            "background_size": bg_size,
            "fx": fx,
            "fc": fc,
        }

        return {"x": x, "shap": y, "color": color, "meta": meta}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
