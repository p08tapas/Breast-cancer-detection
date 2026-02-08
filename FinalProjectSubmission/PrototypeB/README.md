# Breast Cancer Detection

A CNN-based transfer learning solution for detecting breast cancer in mammograms using ResNet50.

## Requirements

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn scipy
```

## Model

- Architecture: ResNet50 with transfer learning (ImageNet pre-trained weights)
- Input: 224x224 RGB images
- Output: Binary classification (BENIGN/MALIGNANT)

## Output Files

- `best_model.weights.h5`: Model weights
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Training curves




