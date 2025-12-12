# Lung Cancer Segmentation Viewer (Streamlit)

This repository is ready to deploy on **Streamlit Community Cloud** and to host on **GitHub**.

## 1) Folder structure expected

Put your data inside the repo using this structure:

```
DEMO/
  ORIGINAL/
  GT_RAS_PNG_RECORTE/
  MEJOR_SEMIAUTOMATICO/
  PRUEBAAUTO - ROIM/
  PRUEBAAUTO_PROBS/        (optional)
  FIGS_ERRORMAPS/          (optional)

lung_app_metrics/          (optional)
  dice_model1.npy
  iou_model1.npy
  dice_model2.npy
  iou_model2.npy
  dice_model3.npy
  iou_model3.npy

assets/                    (optional)
  ADC_P5.png
  SCC_P5.png
  segmentation_rgb.png
  mri_mask.png
```

If you don't want to store the data in the repo locally, you can run with environment variables:

- `DEMO_DIR=/absolute/path/to/DEMO`
- `METRICS_DIR=/absolute/path/to/lung_app_metrics`

> For Streamlit Community Cloud, the simplest option is to include the `DEMO/` and `lung_app_metrics/` folders in the repository.
> If the files are large, use **Git LFS**.

## 2) Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3) Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud → **New app**
3. Select your GitHub repo and set:
   - **Main file path**: `app.py`
4. Click **Deploy**.
