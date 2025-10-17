import os
import math
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import json as _json


from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

from pycaret.classification import load_model, predict_model

# --------- Load model once on startup ----------
MODEL_PATH = "permdi_blend_ridge_cat_v1"
RECURRENCE_MODEL_PATH = "recurrence_nb_calibrated_v1"
model = load_model(MODEL_PATH)
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
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize categories
    if "Surgical Approach" in df.columns:
        df["Surgical Approach"] = df["Surgical Approach"].astype(str).str.strip().str.lower()
        df["Surgical Approach"] = df["Surgical Approach"].replace({"endo": "endoscopic", "endo.": "endoscopic"})

    if "Resection outcome (total, subtotal, partial)" in df.columns:
        col = "Resection outcome (total, subtotal, partial)"
        df[col] = df[col].astype(str).str.strip().str.lower().replace({
            "near-total": "near total", "near_total": "near total"
        })

    return df[FEATURES]

# --- App & CORS ---
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

# --- HTML (no native submit; explicit button click); purple→orange; transparent Plotly ---
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
  <script src="https://cdn.plot.ly/plotly-2.26.2.min.js"></script>
</head>
<body>

<h2>Permanent DI Risk (Pilot)</h2>

<form id="predict-form" onsubmit="return false;">
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
  <button id="btn-predict" type="button">Predict</button>
</form>

<div id="msg" class="small" style="margin-top:6px;"></div>

<div id="results" class="card" style="display:none;"></div>

<!-- SHAP controls -->
<div id="shap-card" class="card" style="display:none;">
  <h3 style="margin-top:0">XAI analysis</h3>
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
    <div class="flex items-end">
      <button id="btn-shap" type="button">Build SHAP plot</button>
    </div>
  </div>

  <!-- brand progress bar -->
  <div id="xai-progress-wrap" style="display:none; margin-top:8px; width:100%; height:8px; border-radius:8px; background:#e5e7eb; overflow:hidden;">
    <div id="xai-progress"
         style="height:100%; width:0%;
                background-image:linear-gradient(90deg, #7c3aed, #f59e0b);
                transition: width .2s ease; border-radius:8px;">
    </div>
  </div>

  <div id="shap-msg" class="small" style="margin-top:6px;"></div>
  <div id="plot" style="height:520px; margin-top:10px;"></div>
</div>

<script>
document.addEventListener("DOMContentLoaded", () => {
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
  const btnPredict = document.getElementById("btn-predict");
  const results = document.getElementById("results");
  const msg = document.getElementById("msg");
  const shapCard = document.getElementById("shap-card");
  const shapMsg = document.getElementById("shap-msg");
  const featX = document.getElementById("feat-x");
  const featColor = document.getElementById("feat-color");
  const plotDiv = document.getElementById("plot");
  const progressWrap = document.getElementById("xai-progress-wrap");
  const progressBar = document.getElementById("xai-progress");

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
    featX.value = "BMI";
    featColor.value = "";
  }
  fillFeatureSelects();

  function readPayload() {
    const fd = new FormData(form);
    const cats = new Set(["Surgical Approach","Resection outcome (total, subtotal, partial)"]);
    const payload = {};
    for (const [k, vRaw] of fd.entries()) {
      const v = (vRaw || "").toString().trim();
      if (cats.has(k)) payload[k] = v || null;
      else payload[k] = (v === "" ? null : Number(v));
    }
    if (payload["BMI"] != null && (payload["Last BMI on file"] == null || isNaN(payload["Last BMI on file"]))) {
      payload["Last BMI on file"] = payload["BMI"];
    }
    for (const b of BIN_LIKE) if (!(b in payload)) payload[b] = 0;
    return payload;
  }

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

  btnPredict.addEventListener("click", async () => {
    msg.textContent = "Predicting…";
    shapMsg.textContent = "";
    results.style.display = "none";
    shapCard.style.display = "none";

    const payload = readPayload();

    try {
      const [r1, r2] = await Promise.all([
        fetch("/predict", { method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(payload) }),
        fetch("/predict-recurrence", { method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(payload) }),
      ]);
      const d1 = await r1.json();
      const d2 = await r2.json();
      if (!r1.ok || d1?.error) throw new Error(d1?.error || "Permanent DI prediction failed");
      if (!r2.ok || d2?.error) throw new Error(d2?.error || "Recurrence prediction failed");

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

      window.__lastPayload = payload;
      shapCard.style.display = "block";
    } catch (err) {
      console.error(err);
      msg.innerHTML = '<span class="err">' + (err?.message || String(err)) + '</span>';
    }
  });

  document.getElementById("btn-shap").addEventListener("click", async () => {
    shapMsg.textContent = "";
    progressWrap.style.display = "block";
    progressBar.style.width = "0%";

    const payload = window.__lastPayload;
    if (!payload) { shapMsg.innerHTML = '<span class="err">Submit a prediction first.</span>'; return; }

    const rows = makeCohortRows(payload, 30);
    const body = { rows, feature_x: featX.value, feature_color: featColor.value || undefined, y_mode: "by_x" };

    try {
      const res = await fetch("/shap/dependence-stream", {
        method: "POST", headers: {"content-type":"application/json"}, body: JSON.stringify(body)
      });
      if (!res.ok || !res.body) { shapMsg.innerHTML = '<span class="err">SHAP stream failed</span>'; return; }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        let idx;
        while ((idx = buf.indexOf("\\n\\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);

          const lines = frame.split("\\n");
          const ev = lines.find(l => l.startsWith("event:"))?.slice(6).trim();
          const dataLine = lines.find(l => l.startsWith("data:"))?.slice(5).trim();
          if (!ev || !dataLine) continue;

          if (ev === "progress") {
            const { pct } = JSON.parse(dataLine);
            progressBar.style.width = pct + "%";
          } else if (ev === "done") {
            const data = JSON.parse(dataLine);

            Plotly.newPlot(plotDiv, [{
              x: data.x,
              y: data.shap,
              mode: "markers",
              type: "scatter",
              marker: {
                size: 8,
                color: data.color || undefined,
                colorscale: [
                  [0, "rgba(128,0,128,0.9)"],  // purple
                  [1, "rgba(255,140,0,0.9)"]   // orange
                ],
                colorbar: {
                  title: featColor.value || "",
                  titleside: "right",
                  tickfont: { size: 11 }
                },
                showscale: !!(data.color && data.color.length)
              },
              hovertemplate: featX.value + "=%{x}<br>SHAP=%{y:.3f}<extra></extra>"
            }], {
              title: "SHAP dependence: " + featX.value,
              xaxis: { title: featX.value },
              yaxis: { title: "SHAP value (Permanent DI)" },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 60, r: 40, t: 50, b: 50 }
            }, { responsive: true });

            progressBar.style.width = "100%";
            shapMsg.innerHTML = '<span class="ok">Done.</span>';
          } else if (ev === "error") {
            shapMsg.innerHTML = '<span class="err">' + JSON.parse(dataLine) + '</span>';
          }
        }
      }
    } catch (err) {
      console.error(err);
      shapMsg.innerHTML = '<span class="err">' + (err?.message || String(err)) + '</span>';
    }
  });
});
</script>

</body>
</html>
"""

# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
def root():
    return FORM_HTML

@app.get("/health")
def health():
    return {"status": "ok"}

def _proba_pos_from_predict_model(out: pd.DataFrame) -> Optional[float]:
    score_cols = [c for c in out.columns if c.startswith("Score_")]
    if not score_cols:
        return None
    for key in ("Score_1", "Score_True", "Score_yes", "Score_positive"):
        if key in out.columns:
            return float(out.loc[0, key])
    if len(score_cols) == 2 and "Score_0" in score_cols:
        other = score_cols[0] if score_cols[1] == "Score_0" else score_cols[1]
        return float(out.loc[0, other])
    row = out.loc[0, score_cols]
    return float(row.max())

def _predict_pos_proba_safe(any_model, X: pd.DataFrame) -> float:
    try:
        return float(any_model.predict_proba(X)[:, 1][0])
    except Exception:
        pass
    try:
        out = predict_model(any_model, data=X, raw_score=True)
        p = _proba_pos_from_predict_model(out)
        if p is not None:
            return float(p)
    except Exception:
        pass
    try:
        margin = float(any_model.decision_function(X)[0])
        return 1.0 / (1.0 + math.exp(-margin))
    except Exception:
        try:
            y = any_model.predict(X)[0]
            return float(y) if y in (0, 1, True, False) else float(y == 1)
        except Exception as _:
            raise RuntimeError("Unable to compute probability for the given model and inputs.")

@app.post("/predict")
def predict_api(payload: dict = Body(...)):
    try:
        X = prepare_inputs(pd.DataFrame([payload]))
        p1 = _predict_pos_proba_safe(model, X)
        THRESH = float(os.getenv("THRESHOLD", 0.5))
        return {"prob_permanent_DI": p1, "label": int(p1 >= THRESH), "threshold": THRESH}
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "received_keys": list(payload.keys()), "prepared_columns": list(X.columns) if "X" in locals() else None},
            status_code=400,
        )

@app.post("/predict-recurrence")
def predict_recurrence_api(payload: dict = Body(...)):
    try:
        X = prepare_inputs(pd.DataFrame([payload]))
        p = _predict_pos_proba_safe(recurrence, X)
        THRESH_R = float(os.getenv("THRESHOLD_RECURRENCE", 0.5))
        return {"prob_recurrence": p, "label": int(p >= THRESH_R), "threshold": THRESH_R}
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "received_keys": list(payload.keys()), "prepared_columns": list(X.columns) if "X" in locals() else None},
            status_code=400,
        )

@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form():
    return HTMLResponse("<pre>Use the main page.</pre>")

# --- non-stream SHAP (full dataset, no nsamples/background) ---
@app.post("/shap/dependence")
def shap_dependence(payload: dict = Body(...)):
    try:
        import shap  # lazy import
    except Exception:
        return JSONResponse({"error": "SHAP not available on server."}, status_code=501)

    from difflib import get_close_matches
    try:
        rows: List[Dict[str, Any]] = payload.get("rows") or []
        feature_x: Optional[str] = payload.get("feature_x")
        feature_color: Optional[str] = payload.get("feature_color")
        y_mode: str = payload.get("y_mode", "by_x")
        if not rows or not feature_x:
            return JSONResponse({"error": "rows and feature_x required"}, status_code=400)

        X_raw = pd.DataFrame(rows)
        cols = list(X_raw.columns)
        def _find(name: Optional[str]) -> Optional[str]:
            if not name: return None
            if name in cols: return name
            tgt = name.strip().lower()
            for c in cols:
                if c.strip().lower() == tgt: return c
            m = get_close_matches(name, cols, n=1)
            return m[0] if m else None

        fx = _find(feature_x)
        if not fx: return JSONResponse({"error": f"feature_x '{feature_x}' not found"}, status_code=400)
        fc = _find(feature_color)

        def proba_fn_local(X_in):
            if isinstance(X_in, np.ndarray): X_df = pd.DataFrame(X_in, columns=cols)
            else: X_df = X_in.reindex(columns=cols, fill_value=np.nan)
            return model.predict_proba(prepare_inputs(X_df))[:, 1]

        explainer = shap.KernelExplainer(model=proba_fn_local, data=X_raw)
        shap_vals = explainer.shap_values(X_raw)
        if isinstance(shap_vals, list): shap_vals = np.array(shap_vals)

        y = (np.abs(shap_vals).sum(axis=1).tolist() if y_mode == "sum"
             else shap_vals[:, cols.index(fx)].tolist())
        x = X_raw[fx].astype(object).where(~X_raw[fx].isna(), None).tolist()
        color = X_raw[fc].astype(object).where(~X_raw[fc].isna(), None).tolist() if fc else None
        return {"x": x, "shap": y, "color": color, "meta": {"columns": cols, "fx": fx, "fc": fc, "y_mode": y_mode}}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- streaming SHAP with progress (SSE) ---
@app.post("/shap/dependence-stream")
def shap_dependence_stream(payload: dict = Body(...)):
    try:
        import shap
    except Exception:
        def _err():
            yield b"event: error\ndata: \"SHAP not installed\"\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    rows = payload.get("rows") or []
    feature_x = payload.get("feature_x")
    feature_color = payload.get("feature_color")
    y_mode = payload.get("y_mode", "by_x")
    if not rows or not feature_x:
        def _bad(): yield b"event: error\ndata: \"rows and feature_x required\"\n\n"
        return StreamingResponse(_bad(), media_type="text/event-stream")

    X_raw = pd.DataFrame(rows)
    cols = list(X_raw.columns)

    def proba_fn_local(X_in):
        if isinstance(X_in, np.ndarray): X_df = pd.DataFrame(X_in, columns=cols)
        else: X_df = X_in.reindex(columns=cols, fill_value=np.nan)
        return model.predict_proba(prepare_inputs(X_df))[:, 1]

    explainer = shap.KernelExplainer(model=proba_fn_local, data=X_raw)

    def _gen():
        n = len(X_raw)
        shap_rows = []
        for i in range(n):
            sv = explainer.shap_values(X_raw.iloc[[i]])
            if isinstance(sv, list): sv = np.array(sv)
            shap_rows.append(sv[0].tolist())
            pct = int((i + 1) / n * 100)
            yield f"event: progress\ndata: {{\"pct\": {pct}}}\n\n".encode()

        shap_vals = np.array(shap_rows)
        y = (np.abs(shap_vals).sum(axis=1).tolist() if y_mode == "sum"
             else shap_vals[:, cols.index(feature_x)].tolist())
        x = X_raw[feature_x].astype(object).where(~X_raw[feature_x].isna(), None).tolist()
        color = None
        if feature_color and feature_color in cols:
            color = X_raw[feature_color].astype(object).where(~X_raw[feature_color].isna(), None).tolist()

        final = {"x": x, "shap": y, "color": color, "meta": {"columns": cols, "fx": feature_x, "fc": feature_color}}
        yield f"event: done\ndata: {_json.dumps(final)}\n\n".encode()

    return StreamingResponse(_gen(), media_type="text/event-stream")
