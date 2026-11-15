# Craniopharyngioma.AI

An explainable ensemble-learning framework for personalized prediction of postoperative outcomes after craniopharyngioma resection: **permanent diabetes insipidus (DI)** and **tumor recurrence**.

This repository contains the code, trained models, analysis notebooks, and figures for the published article:

> Babel S, Patel A, Parasher A, Tabor M, Agazzi S, van Loveren H, Vakharia K. **Craniopharyngioma.AI: An Explainable Ensemble Learning Framework for Personalized Craniopharyngioma Outcome Prediction.** *Journal of Neurological Surgery Part B: Skull Base*. 2026. doi:[10.1055/a-2992-8671](https://doi.org/10.1055/a-2992-8671)
>
> 📄 Article: https://www.thieme-connect.de/products/ejournals/abstract/10.1055/a-2992-8671

> [!IMPORTANT]
> **Research use only.** The models were trained on a small single-institution cohort (n = 59) without external validation. Results are exploratory and hypothesis-generating. The web application is **not** a validated clinical tool and must not be used to guide patient care.

## Overview

- **Cohort:** patients with histologically confirmed craniopharyngioma resected at a single institution between 2008 and 2023.
- **Features:** clinical, radiographic, and intraoperative variables (age, BMI, tumor dimensions and volume, intraventricular extension, pre-op symptoms, surgical approach, extent of resection).
- **Models (PyCaret):**
  - Permanent DI: blended Ridge + CatBoost ensemble (AUC 0.77 ± 0.05, F1 0.81 ± 0.04)
  - Recurrence: blended Naïve Bayes ensemble (AUC 0.78 ± 0.12, F1 0.67 ± 0.15)
- **Explainability:** SHAP (Kernel explainer) for global feature ranking and per-patient explanations.
- **Trade-off explorer:** for a given patient, sweeps extent of resection and surgical approach to trace the recurrence vs. permanent-DI risk curve.
- **Web app:** FastAPI service with an interactive form, a JSON API, and SHAP plots.

## Repository structure

```
.
├── app.py                 # FastAPI web app + REST API (entry point)
├── models/                # Trained PyCaret pipelines (.pkl)
├── public/                # Static assets served by the app (trade-off explorer)
├── data/                  # Patient data: available upon request (see data/README.md)
├── results/               # Model leaderboards and cross-validation outputs
├── notebooks/             # Analysis notebooks (EDA → modeling → explainability)
├── figures/
│   ├── manuscript/        # Figures as published in the article
│   ├── eda/               # Exploratory data analysis plots and tables
│   └── xai/               # SHAP plots (permanent DI at top level, recurrence/ subfolder)
├── requirements.txt
├── Dockerfile
├── start.sh               # Start command used for Render deployment
└── CITATION.cff
```

### Notebooks

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | Cohort description, missingness, distributions, EDA figures |
| `02_data_cleaning_and_baselines.ipynb` | Cleaning `data.csv` → `data_clean.csv`, baseline models |
| `03_permanent_di_model.ipynb` | AutoML comparison and final permanent-DI ensemble |
| `04_recurrence_model.ipynb` | Recurrence model development |
| `05_recurrence_and_radiation_models.ipynb` | Final recurrence ensemble and post-op radiation experiments |
| `06_explainability_permanent_di.ipynb` | SHAP analysis for the permanent-DI model |
| `07_explainability_recurrence.ipynb` | SHAP analysis for the recurrence model |
| `08_reviewer_followup_analysis.ipynb` | Follow-up analyses supporting the peer-review revision |
| `scratch_model_check.ipynb` | Quick sanity checks of saved models |

Each notebook starts with a setup cell that moves the working directory to the repository root, so paths such as `data/data.csv` and `models/...` resolve whether you launch Jupyter from the root or from `notebooks/`.

## Getting started

Python **3.10** is required (the pinned numeric stack is not compatible with newer versions).

```bash
git clone https://github.com/shrinitbabel/craniopharyngioma.git
cd craniopharyngioma

python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the web app

```bash
uvicorn app:app --reload --port 8000
```

Then open:

- http://localhost:8000/: interactive prediction form
- http://localhost:8000/tradeoff-view: recurrence vs. permanent-DI trade-off explorer
- http://localhost:8000/docs: interactive API documentation

### Run with Docker

```bash
docker build -t craniopharyngioma-ai .
docker run -p 8080:8080 craniopharyngioma-ai
```

### API example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Age ": 40, "BMI": 27,
        "Tumor size": 30, "Tumor size.1": 25, "Tumor size.2": 20,
        "Surgical Approach": "endoscopic",
        "Resection outcome (total, subtotal, partial)": "total",
        "Pre-op visual impairment (y/n) - symptomatic complaint of visual loss": 1
      }'
```

| Endpoint | Method | Description |
| --- | --- | --- |
| `/predict` | POST | Probability of permanent DI |
| `/predict-recurrence` | POST | Probability of tumor recurrence |
| `/tradeoff` | POST | Per-patient recurrence vs. DI curve across resection strategies |
| `/tradeoff-cohort` | GET | Precomputed cohort-level trade-off data |
| `/shap/dependence` | POST | SHAP dependence plot for a feature |
| `/health` | GET | Health check |

Feature names are the original dataset column headers (including the trailing space in `"Age "`); see `/docs` for the full list. Missing features are imputed by the model pipeline. Decision thresholds can be set with the `THRESHOLD` and `THRESHOLD_RECURRENCE` environment variables (default `0.5`).

### Reproduce the analysis

The patient dataset is not included in this repository. It is **available upon request from the corresponding author**; see [`data/README.md`](data/README.md). Once you have it, place the files in `data/` and run the notebooks in numerical order. Retraining overwrites the pipelines in `models/` and the plots in `figures/`.

## Data availability

The data are available upon reasonable request. Please email the corresponding author, Kunal Vakharia ([kvakharia@usf.edu](mailto:kvakharia@usf.edu)). See [`data/README.md`](data/README.md).

## Citation

If you use this code or the models, please cite:

```bibtex
@article{babel2026craniopharyngioma,
  title     = {Craniopharyngioma.AI: An Explainable Ensemble Learning Framework for Personalized Craniopharyngioma Outcome Prediction},
  author    = {Babel, Shrinit and Patel, Ankit and Parasher, Arjun and Tabor, Mark and Agazzi, Siviero and van Loveren, Harry and Vakharia, Kunal},
  journal   = {Journal of Neurological Surgery Part B: Skull Base},
  publisher = {Thieme},
  year      = {2026},
  month     = oct,
  doi       = {10.1055/a-2992-8671},
  url       = {https://www.thieme-connect.de/products/ejournals/abstract/10.1055/a-2992-8671}
}
```

GitHub's **"Cite this repository"** button uses [`CITATION.cff`](CITATION.cff).

## License

Released under the [Apache License 2.0](LICENSE).
