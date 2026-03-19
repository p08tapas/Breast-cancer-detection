
# Shared preprocessing for a single mammogram image.
# Resize to 224x224, ensuring RGB, then applying model-specific normalization.

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224


def load_and_resize(image_path_or_array, target_size=(IMG_SIZE, IMG_SIZE)):
    
    # Load image from path (or use array), resize to target_size, return RGB array.
    # Grayscale images are converted to 3 channels.
    
    if isinstance(image_path_or_array, (str, os.PathLike)):
        img = load_img(image_path_or_array, color_mode="rgb", target_size=target_size)
        x = img_to_array(img)
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.ndim == 2:
            # Grayscale: (H, W) -> (H, W, 3)
            x = np.stack([image_path_or_array] * 3, axis=-1)
        else:
            x = image_path_or_array
        if x.shape[:2] != target_size:
            # Resize via tf or PIL; use tf for consistency
            x = tf.image.resize(x, target_size, method="bilinear").numpy()
        x = np.clip(x, 0, 255).astype(np.float32)
    else:
        raise TypeError("image_path_or_array must be path (str) or numpy array")
    return x





def preprocess_for_prototype_a(x_rgb):
    # EfficientNet-style preprocessing for Prototype A (EfficientNetB0).
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(x_rgb.copy())


def preprocess_for_prototype_b(x_rgb):
    # ResNet50-style preprocessing for Prototype B (ResNet50).
    from tensorflow.keras.applications.resnet50 import preprocess_input
    return preprocess_input(x_rgb.copy())





def preprocess_for_prototype_c(x_rgb):
    # EfficientNet-style preprocessing for Prototype C (ensemble trained with this).
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(x_rgb.copy())


def prepare_single_image(image_path_or_array, model_key):
    # Load image, resize to IMG_SIZE, ensuring RGB, then applying model-specific preprocessing.
    # model_key: 'A' | 'B' | 'C'
    # Returns array of shape (1, IMG_SIZE, IMG_SIZE, 3) for model input.

    x = load_and_resize(image_path_or_array, (IMG_SIZE, IMG_SIZE))
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    if model_key.upper() == "A":
        x = preprocess_for_prototype_a(x)
    elif model_key.upper() == "B":
        x = preprocess_for_prototype_b(x)
    elif model_key.upper() == "C":
        x = preprocess_for_prototype_c(x)
    else:
        raise ValueError("model_key must be 'A', 'B', or 'C'")
    return x
