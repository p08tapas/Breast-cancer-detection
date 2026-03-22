# Mammogram Breast Cancer Detection

CNN-based transfer learning for BENIGN vs MALIGNANT classification. Three prototypes: A (EfficientNetB0), B (ResNet50), C (EfficientNetB0 + ResNet50 ensemble). All use ImageNet weights, focal loss, fine-tune last 30 layers.


## Install

```bash
pip install -r requirements.txt
```

## Dataset

The training code requires the CBIS-DDSM dataset in 'common/archive/'.

Please download it from the link: [CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)[Note: Dataset is not included in the git link given above]

Required layout after extraction:
- Place the 4 CSV files in 'common/archive/csv/': mass_case_description_train_set.csv, mass_case_description_test_set.csv, calc_case_description_train_set.csv, calc_case_description_test_set.csv
- Place the JPEG images in 'common/archive/jpeg/', with one subfolder per SeriesInstanceUID (each subfolder contains its images)

## Model weights

Weights are not included in the repo (excluded by '.gitignore').Hence we would need to download the saved weights in order to get the same result as mentioned in the report.

Please download from the link: [Breast cancer detection using Deep Learning](https://www.kaggle.com/models/tapassanyal/breast-cancer-detection-using-deep-learning)

The Kaggle model has 3 files. Please place each one in its prototype folder and rename to `best_model.weights.h5`:
- 'PrototypeA_best_model.weights.h5' → save as 'PrototypeA/best_model.weights.h5'
- 'PrototypeB_best_model.weights.h5` → save as `PrototypeB/best_model.weights.h5'
- 'PrototypeC_best_model.weights.h5` → save as `PrototypeC/best_model.weights.h5'

Prototype C also needs 'PrototypeC/best_model_config.json'. This is already included in the git

If one needs training from scratch:  In that case, please run the training steps below. The weight files will be generated in PrototypeA, PrototypeB, and PrototypeC.


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

Prototype A and B use 'train.py' directly. Prototype C uses 'main.py' because it saves and loads 'best_model_config.json' (hyperparameters) for correct model reconstruction at inference.

Outputs: 'best_model.weights.h5', 'confusion_matrix.png', 'training_history.png', 'evaluation_metrics.json'. Prototype C also saves 'best_model_config.json'.

## Evaluation

After training (or if weights already exist), we can run evaluation from each prototype folder:

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

This loads the weights, runs the test set, prints metrics (accuracy, AUC, etc.), and writes 'confusion_matrix.png' and 'evaluation_metrics.json'.

## Inference App

- UI: This uploads a single mammogram (JPEG/PNG)( this is to mimic the deployment in a clinical setting)
- Backend: Preprocessing (resize 224×224, RGB, model-specific normalization) and optional Test-Time Augmentation (TTA) for Prototype B and C.
- Output: Predictions from all three models with architecture in brackets: A (EfficientNetB0), B (ResNet50), C (EfficientNetB0 + ResNet50). TTA progress bars for B and C when enabled.

## Running the app locally

Weights must exist in 'PrototypeA/', 'PrototypeB/', and 'PrototypeC/'. Please download them from Kaggle (please see above) or we need to train the models first.

```bash
python main.py
```

Open http://localhost:8080, upload JPEG/PNG, click Predict.


## API

- GET / — upload form
- POST /predict — file (image), optional use_tta_b, use_tta_c
- GET /evaluation — metrics
- GET /prototypes/{proto}/{filename} — confusion_matrix.png, training_history.png
- GET /health — health check

## Deploy to GCP

Prerequisites: model weights in 'PrototypeA/', 'PrototypeB/', and 'PrototypeC/' (download from Kaggle or train locally-please see above), Google Cloud SDK, GCP project with App Engine enabled.

```bash
gcloud auth login
gcloud config set project my-project-2976-488110
gcloud app create --region=us-central
gcloud app deploy
```

App URL: https://my-project-2976-488110.uc.r.appspot.com/

## Environment variables

Please set these to override default weight paths:

- 'WEIGHTS_A' — Prototype A weights (default: 'PrototypeA/best_model.weights.h5')
- 'WEIGHTS_B' — Prototype B weights (default: 'PrototypeB/best_model.weights.h5')
- 'WEIGHTS_C' — Prototype C weights (default: 'PrototypeC/best_model.weights.h5')
- 'CONFIG_C' — Prototype C config JSON (default: 'PrototypeC/best_model_config.json')
