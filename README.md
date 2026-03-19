# Mammogram Breast Cancer Detection

CNN-based transfer learning for BENIGN vs MALIGNANT classification. Three prototypes: A (EfficientNetB0), B (ResNet50), C (EfficientNetB0 + ResNet50 ensemble). All use ImageNet weights, focal loss, fine-tune last 30 layers.

## Install

```bash
pip install -r requirements.txt
```

## Dataset

CBIS-DDSM in `common/archive/`: `csv/` (mass_case_*, calc_case_* CSVs) and `jpeg/` (images by SeriesInstanceUID).

## Model weights

Weights are included in the repo so that cloning and running the app yields the same results without retraining. Ensure `PrototypeA/best_model.weights.h5`, `PrototypeB/best_model.weights.h5`, `PrototypeC/best_model.weights.h5`, and `PrototypeC/best_model_config.json` are committed.

## Training

From the project root:

```bash
cd PrototypeA
python train.py
cd ..

cd PrototypeB
python train.py
cd ..

cd PrototypeC
python main.py
cd ..
```

Prototype A and B use `train.py` directly. Prototype C uses `main.py` because it saves and loads `best_model_config.json` (hyperparameters) for correct model reconstruction at inference.

Outputs: `best_model.weights.h5`, `confusion_matrix.png`, `training_history.png`, `evaluation_metrics.json`. Prototype C also saves `best_model_config.json`.

## Evaluation

After training (or if weights already exist), run evaluation from each prototype folder:

```bash
cd PrototypeA
python main.py --mode eval
cd ..

cd PrototypeB
python main.py --mode eval
cd ..

cd PrototypeC
python main.py --mode eval
cd ..
```

This loads the weights, runs the test set, prints metrics (accuracy, AUC, etc.), and writes `confusion_matrix.png` and `evaluation_metrics.json`.

## Inference App

- **UI**: This uploads a single mammogram (JPEG/PNG)( this is to mimic the deployment in a clinical setting)
- **Backend**: Preprocessing (resize 224×224, RGB, model-specific normalization) and optional Test-Time Augmentation (TTA) for Prototype B and C.
- **Output**: Predictions from all three models with architecture in brackets: A (EfficientNetB0), B (ResNet50), C (EfficientNetB0 + ResNet50). TTA progress bars for B and C when enabled.

## Run locally

Weights must exist in PrototypeA/, PrototypeB/, PrototypeC/.

```bash
python main.py
```

Open http://localhost:8080, upload JPEG/PNG, click Predict.

Alternative (with auto-reload during development):

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8080 --reload
```

## API

- GET / — upload form
- POST /predict — file (image), optional use_tta_b, use_tta_c
- GET /evaluation — metrics
- GET /prototypes/{proto}/{filename} — confusion_matrix.png, training_history.png
- GET /health — health check

## Deploy to GCP

Prerequisites: model weights in PrototypeA/B/C, Google Cloud SDK, GCP project with App Engine enabled.

```bash
gcloud auth login
gcloud config set project my-project-2976-488110
gcloud app create --region=us-central
gcloud app deploy
```

App URL: https://my-project-2976-488110.uc.r.appspot.com/

## Environment variables

Set these to override default weight paths:

- `WEIGHTS_A` — Prototype A weights (default: `PrototypeA/best_model.weights.h5`)
- `WEIGHTS_B` — Prototype B weights (default: `PrototypeB/best_model.weights.h5`)
- `WEIGHTS_C` — Prototype C weights (default: `PrototypeC/best_model.weights.h5`)
- `CONFIG_C` — Prototype C config JSON (default: `PrototypeC/best_model_config.json`)
