# Here we run all three CNN prototypes on a single mammogram.
# This handles preprocessing and optional TTA for Prototype B and C.

import os
import sys
import json
import numpy as np

# Project root so we can import PrototypeA/B/C
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.preprocess import load_and_resize, preprocess_for_prototype_a, preprocess_for_prototype_b, preprocess_for_prototype_c

IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.5

# Paths to evaluation_metrics.json for loading optimal thresholds (from each prototype's evaluate.py)
EVAL_METRICS_PATHS = {
    "PrototypeA": os.path.join(PROJECT_ROOT, "PrototypeA", "evaluation_metrics.json"),
    "PrototypeB": os.path.join(PROJECT_ROOT, "PrototypeB", "evaluation_metrics.json"),
    "PrototypeC": os.path.join(PROJECT_ROOT, "PrototypeC", "evaluation_metrics.json"),
}


def _load_optimal_thresholds():
    """
    This willload optimal thresholds from each prototype's evaluation_metrics.json.
    Uses evaluation weights (optimal_threshold) for single-mammogram inference.
    Falls back to DEFAULT_THRESHOLD if there is no optimal_threshold.
    """
    thresholds = {}
    for key, path in EVAL_METRICS_PATHS.items():
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                opt = data.get("optimal_threshold")
                if opt is not None:
                    thresholds[key] = float(opt)
                else:
                    thresholds[key] = DEFAULT_THRESHOLD
            except (json.JSONDecodeError, TypeError, ValueError):
                thresholds[key] = DEFAULT_THRESHOLD
        else:
            thresholds[key] = DEFAULT_THRESHOLD
    return thresholds


# TTA configs aligned with PrototypeB/evaluate.py (7 transforms)
TTA_CONFIGS_B = [
    {},
    {"horizontal_flip": True},
    {"horizontal_flip": True, "brightness_range": [0.9, 1.1]},
    {"rotation_range": 10},
    {"brightness_range": [0.85, 1.15]},
    {"zoom_range": 0.1},
    {"width_shift_range": 0.1, "height_shift_range": 0.1},
]

# Top 10 weighted TTA configs aligned with PrototypeC/evaluate.py
TTA_CONFIGS_C = [
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": 0, "brightness_range": None, "weight": 1.3},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": True, "zoom_range": 0, "brightness_range": None, "weight": 1.3},
    {"rotation_range": 5, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": 0, "brightness_range": None, "weight": 1.1},
    {"rotation_range": 10, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": 0, "brightness_range": None, "weight": 1.0},
    {"rotation_range": 10, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": True, "zoom_range": 0, "brightness_range": None, "weight": 1.0},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": 0.05, "brightness_range": None, "weight": 1.0},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": -0.05, "brightness_range": None, "weight": 1.0},
    {"rotation_range": 0, "width_shift_range": 0.05, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": 0, "brightness_range": None, "weight": 0.9},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0.05, "horizontal_flip": False, "zoom_range": 0, "brightness_range": None, "weight": 0.9},
    {"rotation_range": 0, "width_shift_range": 0, "height_shift_range": 0, "horizontal_flip": False, "zoom_range": 0, "brightness_range": [0.85, 0.95], "weight": 0.95},
]


def _get_tta_c_params(cfg):
    # This will extract ImageDataGenerator params from a TTA config (no weight).
    params = {}
    if cfg.get("rotation_range"):
        params["rotation_range"] = cfg["rotation_range"]
    if cfg.get("width_shift_range"):
        params["width_shift_range"] = cfg["width_shift_range"]
    if cfg.get("height_shift_range"):
        params["height_shift_range"] = cfg["height_shift_range"]
    if cfg.get("horizontal_flip"):
        params["horizontal_flip"] = True
    if cfg.get("zoom_range") is not None and cfg["zoom_range"] != 0:
        params["zoom_range"] = abs(cfg["zoom_range"])
    if cfg.get("brightness_range"):
        params["brightness_range"] = cfg["brightness_range"]
    return params


def _augment_image(x_rgb, **kwargs):
    # This will apply one augmentation to a single image (1, H, W, 3). Returns (1, H, W, 3).
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(**kwargs)
    it = datagen.flow(x_rgb, batch_size=1, shuffle=False)
    return next(it)


def _apply_tta_c_config(x_rgb, params):
    # This will apply ImageDataGenerator params to x_rgb and return augmented image (1, H, W, 3).
    if not params:
        return x_rgb
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(**params)
    it = datagen.flow(x_rgb, batch_size=1, shuffle=False)
    return next(it)


def _load_model_a(weights_path):
    # This will load the model for Prototype A.

    from PrototypeA.model import build_model
    model = build_model(use_focal_loss=True)
    model.load_weights(weights_path)
    model.trainable = False
    return model


def _load_model_b(weights_path):
    # This will load the model for Prototype B.
    
    from PrototypeB.model import build_model
    model = build_model(use_focal_loss=True)
    model.load_weights(weights_path)
    model.trainable = False
    return model


def _load_model_c(weights_path, config_path=None):
    # This will load the model for Prototype C.
    
    from PrototypeC.model import build_model
    kwargs = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        kwargs = {k: v for k, v in cfg.items() if k in ("learning_rate", "dropout_rate", "dense_units", "use_focal_loss")}
    model = build_model(**kwargs) if kwargs else build_model()
    model.load_weights(weights_path)
    model.trainable = False
    return model


# Display names with architecture for UI
MODEL_DISPLAY_NAMES = {
    "PrototypeA": "Prototype A (EfficientNetB0)",
    "PrototypeB": "Prototype B (ResNet50)",
    "PrototypeC": "Prototype C (EfficientNetB0 + ResNet50)",
}


def predict_single(
    image_path_or_array,
    weights_a,
    weights_b,
    weights_c,
    use_tta_b=True,
    use_tta_c=True,
    num_tta_c=10,
    config_c_path=None,
    progress_callback=None,
):
    """
    Run all three models on one mammogram.

    image_path_or_array: path (str) or numpy array (H, W) or (H, W, 3)
    weights_*: paths to best_model.weights.h5 for each prototype
    use_tta_b / use_tta_c: whether to use TTA for B and C
    num_tta_c: number of TTA configs to use for C (max 10)
    config_c_path: optional path to best_model_config.json for C
    progress_callback: optional callable(current, total) for unified progress (1 step A, then B steps, then C steps)

    Returns dict: probabilities, labels, threshold, model_display_names
    """
    total_steps = 1 + (len(TTA_CONFIGS_B) if (use_tta_b and TTA_CONFIGS_B) else 1) + (min(num_tta_c, len(TTA_CONFIGS_C)) if (use_tta_c and TTA_CONFIGS_C) else 1)
    step = [0]

    def report():
        step[0] += 1
        if progress_callback:
            progress_callback(step[0], total_steps)

    x_rgb = load_and_resize(image_path_or_array, (IMG_SIZE, IMG_SIZE))
    if x_rgb.ndim == 3:
        x_rgb = np.expand_dims(x_rgb, axis=0)

    # Model A: single pass, EfficientNet preprocess
    x_a = preprocess_for_prototype_a(x_rgb.copy())
    model_a = _load_model_a(weights_a)
    p_a = float(model_a.predict(x_a, verbose=0).flatten()[0])
    report()

    # Model B: ResNet50 preprocess, optional TTA (load model once)
    model_b = _load_model_b(weights_b)
    if use_tta_b and TTA_CONFIGS_B:
        preds_b = []
        for i, tta_cfg in enumerate(TTA_CONFIGS_B):
            report()
            if tta_cfg:
                x_b_tta = _augment_image(x_rgb.copy(), **tta_cfg)
            else:
                x_b_tta = x_rgb.copy()
            x_b_tta = preprocess_for_prototype_b(x_b_tta)
            preds_b.append(float(model_b.predict(x_b_tta, verbose=0).flatten()[0]))
        p_b = float(np.mean(preds_b))
    else:
        x_b = preprocess_for_prototype_b(x_rgb.copy())
        p_b = float(model_b.predict(x_b, verbose=0).flatten()[0])
        report()

    # Model C: EfficientNet preprocess, optional weighted TTA (load model once)
    model_c = _load_model_c(weights_c, config_c_path)
    configs_c = TTA_CONFIGS_C[:num_tta_c]
    if use_tta_c and configs_c:
        preds_c = []
        weights_c_list = []
        for i, cfg in enumerate(configs_c):
            report()
            weight = cfg.get("weight", 1.0)
            params = _get_tta_c_params(cfg)
            x_c_tta = _apply_tta_c_config(x_rgb.copy(), params)
            x_c_tta = preprocess_for_prototype_c(x_c_tta)
            preds_c.append(float(model_c.predict(x_c_tta, verbose=0).flatten()[0]))
            weights_c_list.append(weight)
        weights_arr = np.array(weights_c_list)
        weights_arr = weights_arr / weights_arr.sum()
        p_c = float(np.average(preds_c, weights=weights_arr))
    else:
        x_c = preprocess_for_prototype_c(x_rgb.copy())
        p_c = float(model_c.predict(x_c, verbose=0).flatten()[0])
        report()

    probs = {"PrototypeA": p_a, "PrototypeB": p_b, "PrototypeC": p_c}

    # We use the optimal thresholds from each prototype's evaluation_metrics.json
    # (same weights used during evaluation) for single-mammogram inference
    thresholds = _load_optimal_thresholds()
    labels = {
        k: "MALIGNANT" if v >= thresholds.get(k, DEFAULT_THRESHOLD) else "BENIGN"
        for k, v in probs.items()
    }

    return {
        "probabilities": probs,
        "labels": labels,
        "threshold": DEFAULT_THRESHOLD,  # legacy fallback
        "thresholds": thresholds,  # per-model optimal thresholds from evaluation
        "model_display_names": MODEL_DISPLAY_NAMES,
    }
