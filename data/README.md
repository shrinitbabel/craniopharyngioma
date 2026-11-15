# Data

The de-identified patient dataset used in this study is **not publicly available** because it contains protected patient information.

The data are available **upon reasonable request**. Please email the corresponding author:

**Kunal Vakharia, MD**, Department of Neurosurgery, University of South Florida
✉️ [kvakharia@usf.edu](mailto:kvakharia@usf.edu)

Requests may require institutional approval and a data use agreement.

## Expected files

To rerun the notebooks, place the files you receive in this folder:

| File | Description |
| --- | --- |
| `data.csv` | Raw cohort table (one row per patient), the input to all notebooks |
| `data_clean.csv` | Cleaned table produced by `notebooks/02_data_cleaning_and_baselines.ipynb` |
| `data_imputed.csv` | Imputed table used in exploratory modeling |

Everything in this folder except this README is gitignored.

The trained models in `models/` and the web app do **not** need these files.
