import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from model import build_model, IMG_SIZE



BATCH_SIZE = 32
EPOCHS = 30  # I have increased epochs for better convergence




def create_generators(train_df, test_df):
    """Creating the data generators with proper preprocessing."""

     # Splitting train into train and validation sets
    train_labels_int = train_df['label'].astype(int)
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_labels_int
    )





    # Oversampling the minority class in the training set
    train_labels_int = train_split['label'].astype(int)
    n_benign = (train_labels_int == 0).sum()
    n_malignant = (train_labels_int == 1).sum()

    
    if n_benign != n_malignant:
        if n_malignant < n_benign:
            minority = train_split[train_split['label'] == '1']
            n_add = n_benign - n_malignant
            oversampled = minority.sample(n=n_add, replace=True, random_state=42)
            train_split = pd.concat([train_split, oversampled], ignore_index=True)
        else:
            minority = train_split[train_split['label'] == '0']
            n_add = n_malignant - n_benign
            oversampled = minority.sample(n=n_add, replace=True, random_state=42)
            train_split = pd.concat([train_split, oversampled], ignore_index=True)
        train_split = train_split.sample(frac=1, random_state=42).reset_index(drop=True)




    
    # Data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        fill_mode='nearest'
    )
    
      
    # Not applying augmentation for validation and test sets
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
   
    # Here we are creating the data generators now
    train_gen = train_datagen.flow_from_dataframe(
        train_split,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',  # I am using RGB for ImageNet models (grayscale images will be converted)
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_dataframe(
        val_split,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    steps_per_epoch = max(1, int(np.ceil(len(train_split) / BATCH_SIZE)))
    validation_steps = max(1, int(np.ceil(len(val_split) / BATCH_SIZE)))
    
    return train_gen, val_gen, test_gen, steps_per_epoch, validation_steps

def train_model(train_df, test_df, weights_path='best_model.weights.h5', resume_from_weights=False):
    """Train model."""
    model = build_model()
    
    if resume_from_weights and os.path.exists(weights_path):
        model.load_weights(weights_path)
    
    train_gen, val_gen, test_gen, steps_per_epoch, validation_steps = create_generators(train_df, test_df)
    
    callbacks = [
        EarlyStopping(monitor='val_auc', mode='max', patience=12, min_delta=0.001, restore_best_weights=True, verbose=0),
        ModelCheckpoint(weights_path, monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, min_delta=0.0001, verbose=0, cooldown=2)
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
    )
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    
    actual_epochs = len(history.history['loss'])
    if 'val_accuracy' in history.history:
        best_val_acc = max(history.history['val_accuracy'])
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    return model, history, test_gen



if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from common.data_loader import load_dataset
    
    train_df, test_df = load_dataset()
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: No data loaded!")
    else:
        model, history, test_gen = train_model(train_df, test_df)
        print("\nTraining completed!")

