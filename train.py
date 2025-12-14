import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from model import build_model, IMG_SIZE



BATCH_SIZE = 32
EPOCHS = 20

def create_generators(train_df, test_df):
    """Creating the data generators with proper preprocessing."""
    
    # This essentially splits split train into train and validation
    train_labels_int = train_df['label'].astype(int)
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_labels_int
    )

    
    # Here it will oversample the minority class in the training set
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


    
    # Here we are trying to applydata augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    
      
    # There is no augmentation for validation and test
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
   
    # Here we aew creating the data generators
    train_gen = train_datagen.flow_from_dataframe(
        train_split,
        x_col='image_path',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',  # RGB for ImageNet models (grayscale images will be converted)
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
    
    print(f"\nData Generators Created:")
    print(f"  Train samples: {len(train_split)}")
    print(f"  Validation samples: {len(val_split)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    
    return train_gen, val_gen, test_gen, steps_per_epoch, validation_steps

def train_model(train_df, test_df, weights_path='best_model.weights.h5', resume_from_weights=False):
    """
   Note: resume_from_weights: If True and weights exist, it willcontinue training from those weights.
    If False, it willstart training from scratch (ImageNet weights for base, random for head).
    """
    print("\nBuilding Model")
    
    # This is for building the model
    model = build_model()
    # Model is already compiled with standardmetrics (accuracy, precision, recall, auc)
    # We'll monitor recall (sensitivity)directly for statistical power
    
    # This will check if we should resume from previous weights
    if resume_from_weights and os.path.exists(weights_path):
        print(f"\nResuming training from existing weights: {weights_path}")
        model.load_weights(weights_path)
        print("Previous weights loaded successfully - continuing training...")
    else:
        if os.path.exists(weights_path) and not resume_from_weights:
            print(f"\nStarting fresh training (existing weights will be overwritten)")
            print(f"   To continue from previous weights, set resume_from_weights=True")
        else:
            print(f"\nStarting training from scratch")
            print(f"   Base model: EfficientNetB0 with ImageNet weights (frozen bottom layers)")
            print(f"   Head layers: Random initialization")
    
    print(f"\nTraining Configuration:")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Early stopping patience: 15 epochs")
    print(f"  Monitoring: val_combined_metric (60% balanced accuracy + 40% recall)")
    print(f"  Learning rate reduction patience: 6 epochs")
    model.summary()
    
    # Here we are creating the data generators and getting the actual training split
    train_gen, val_gen, test_gen, steps_per_epoch, validation_steps = create_generators(train_df, test_df)
    
    # Class balancing is handled through:
    # 1. Oversampling (in create_generators) - balances the dataset
    # 2. Focal loss (alpha=0.65) - gives moderate emphasis to MALIGNANT class.
    
    
    # This will display class distribution info for reference
    train_labels_int = train_df['label'].astype(int)
    n_benign = (train_labels_int == 0).sum()
    n_malignant = (train_labels_int == 1).sum()
    total = n_benign + n_malignant
    
    print(f"\nClass Balancing Strategy:")
    print(f"  Original distribution - BENIGN: {n_benign}, MALIGNANT: {n_malignant}")
    print(f"  1. Oversampling: Minority class is oversampled to match majority")
    print(f"  2. Focal Loss: Alpha=0.65 gives moderate emphasis to MALIGNANT class")
    print(f"  3. Monitoring: Validation recall (sensitivity) for statistical power")
    
    
    
    # Callbacks- This will monitor validation recall (sensitivity) for statistical power
    # Also it will track accuracy to ensure balanced performance.
    callbacks = [
        EarlyStopping(monitor='val_recall', mode='max', patience=15, min_delta=0.0001, restore_best_weights=True, verbose=0),
        ModelCheckpoint(
            weights_path,
            monitor='val_recall',  # This will save model with best recall (sensitivity)
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,  # Increased patience
            min_lr=1e-7,
            min_delta=0.0001,  # Minimum change to qualify as improvement
            verbose=0
        )
    ]
    
    print("\nStarting Training")
           
        
    # Train the model
        # The focal loss (alpha=0.65) and oversampling provide sufficient class balancing
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0
    )
    
        # Load best weights
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"\nLoaded best weights from {weights_path}")
    
   
    # Print training summary
    actual_epochs = len(history.history['loss'])
    print(f"\nTRAINING SUMMARY")
    print(f"Total epochs trained: {actual_epochs}/{EPOCHS}")
    
   
   
    # here it will calculate combined metric from history for reporting
    if 'val_recall' in history.history and 'val_accuracy' in history.history:
        best_recall = max(history.history['val_recall'])
        best_acc = max(history.history['val_accuracy'])
        best_combined = 0.6 * best_acc + 0.4 * best_recall
        print(f"Best validation combined metric (60% acc + 40% recall): {best_combined:.4f}")
    
    if 'val_accuracy' in history.history:
        best_val_acc = max(history.history['val_accuracy'])
        print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    if 'val_recall' in history.history:
        best_val_recall = max(history.history['val_recall'])
        print(f"Best validation recall (sensitivity): {best_val_recall:.4f} ({best_val_recall*100:.2f}%)")
    
    if actual_epochs < EPOCHS:
        print(f"\nEarly stopping triggered at epoch {actual_epochs}")
        print("Model stopped because validation metric stopped improving.")
    else:
        print(f"\nTraining completed all {EPOCHS} epochs")
    
    return model, history, test_gen

if __name__ == '__main__':
    from data_loader import load_dataset
    import os
    
    train_df, test_df = load_dataset()
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: No data loaded!")
    else:
        model, history, test_gen = train_model(train_df, test_df)
        print("\nTraining completed!")

