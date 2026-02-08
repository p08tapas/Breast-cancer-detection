# Breast Cancer Detection

A CNN-based transfer learning solution for detecting breast cancer in mammograms using EfficientNetB0.

## Requirements

"""bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn scipy
"""

## Usage

'''bash
python main.py
'''

## Model

- Architecture: EfficientNetB0 with transfer learning
- Input: 224x224 RGB images
- Output: Binary classification (BENIGN/MALIGNANT)

## Output Files

- "best_model.weights.h5": Model weights
- "confusion_matrix.png": Confusion matrix visualization
- "training_history.png": Training curves

