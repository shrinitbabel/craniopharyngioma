from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

app = FastAPI(title="Babels.ai — AI Labs API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

model = load_model("permdi_blend_ridge_cat_v1")

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

class PredictPayload(BaseModel):
    Age: float = Field(alias="Age ")
    BMI: float
    Tumor_size: float = Field(alias="Tumor size")
    Tumor_size_1: float = Field(alias="Tumor size.1")
    Tumor_size_2: float = Field(alias="Tumor size.2")
    Last_BMI_on_file: float = Field(alias="Last BMI on file")
    Preop_visual_impairment: int = Field(alias="Pre-op visual impairment (y/n) - symptomatic complaint of visual loss")
    Preop_hydro: int = Field(alias="Pre-op hydrocephalus (y/n)")
    Preop_icp: int = Field(alias="Pre-op intacranial pressure symptoms")
    Preop_headache: int = Field(alias="Pre-op headache")
    Preop_endo: int = Field(alias="Pre-op endocrine changes (y/n)")
    Preop_panhypo: int = Field(alias="Pre-op panhypopit (y/n) - define number of hormone axes that are non-functional")
    IV_extension: int = Field(alias="Intraventricular extension")
    Surgical_Approach: str = Field(alias="Surgical Approach")
    Resection_outcome: str = Field(alias="Resection outcome (total, subtotal, partial)")
    Tumor_volume: float | None = Field(default=None, alias="Tumor volume")

    class Config:
        populate_by_name = True

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload: PredictPayload):
    d = payload.model_dump(by_alias=True)
    # compute tumor volume if missing
    if d.get("Tumor volume") is None:
        d["Tumor volume"] = (4/3)*np.pi * (d["Tumor size"]/2)*(d["Tumor size.1"]/2)*(d["Tumor size.2"]/2)
    X = pd.DataFrame([d], columns=FEATURES)
    out = predict_model(model, data=X, raw_score=True)
    proba = float(out.loc[0, "Score_1"])
    label = int(out.loc[0, "prediction_label"])
    return {"label": label, "probability": proba, "threshold_used": 0.5, "version": "permdi_blend_ridge_cat_v1"}
