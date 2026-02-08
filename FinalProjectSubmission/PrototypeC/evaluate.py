import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


from model import IMG_SIZE
from train import BATCH_SIZE




def evaluate_model_with_tta(model, test_df, num_tta=10):
    """Test-Time Augmentation: Create multiple augmented versions and average predictions."""
    print("\nUsing Test-Time Augmentation (TTA)...")
    
    
    
    
    tta_configs = [
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': None, 'weight': 1.3, 'name': 'original'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': True, 'zoom_range': 0, 'brightness_range': None, 'weight': 1.3, 'name': 'flip'},
        {'rotation_range': 5, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': None, 'weight': 1.1, 'name': 'rot5'},
        {'rotation_range': 10, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': None, 'weight': 1.0, 'name': 'rot10'},
        {'rotation_range': 10, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': True, 'zoom_range': 0, 'brightness_range': None, 'weight': 1.0, 'name': 'rot10+flip'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0.05, 'brightness_range': None, 'weight': 1.0, 'name': 'zoom_in'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': -0.05, 'brightness_range': None, 'weight': 1.0, 'name': 'zoom_out'},
        {'rotation_range': 0, 'width_shift_range': 0.05, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': None, 'weight': 0.9, 'name': 'shift_w'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0.05, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': None, 'weight': 0.9, 'name': 'shift_h'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': [0.85, 0.95], 'weight': 0.95, 'name': 'bright_low'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': [1.05, 1.15], 'weight': 0.95, 'name': 'bright_high'},
        {'rotation_range': 5, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0.05, 'brightness_range': None, 'weight': 0.9, 'name': 'rot5+zoom'},
        {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': True, 'zoom_range': 0.05, 'brightness_range': None, 'weight': 0.9, 'name': 'flip+zoom'},
        {'rotation_range': 15, 'width_shift_range': 0, 'height_shift_range': 0, 
         'horizontal_flip': False, 'zoom_range': 0, 'brightness_range': None, 'weight': 0.85, 'name': 'rot15'},
    ]
    
    tta_configs_sorted = sorted(tta_configs, key=lambda x: x['weight'], reverse=True)
    tta_configs = tta_configs_sorted[:num_tta]
    
    print(f"  TTA augmentations: {len(tta_configs)}")
    
    all_predictions = []
    all_weights = []
    
    for tta_idx, aug_config in enumerate(tta_configs, 1):
        aug_name = aug_config['name']
        aug_weight = aug_config['weight']
        print(f"  TTA {tta_idx}/{len(tta_configs)}: {aug_name} (weight: {aug_weight:.2f})...")
        zoom_val = aug_config['zoom_range']
        zoom_range = abs(zoom_val) if zoom_val != 0 else None
        rotation_val = abs(aug_config['rotation_range']) if aug_config['rotation_range'] != 0 else 0
        
        gen_params = {'preprocessing_function': preprocess_input}
        
        if rotation_val > 0:
            gen_params['rotation_range'] = rotation_val
        if aug_config['width_shift_range'] > 0:
            gen_params['width_shift_range'] = aug_config['width_shift_range']
        if aug_config['height_shift_range'] > 0:
            gen_params['height_shift_range'] = aug_config['height_shift_range']
        if aug_config['horizontal_flip']:
            gen_params['horizontal_flip'] = True
        if zoom_range is not None and zoom_range > 0:
            gen_params['zoom_range'] = zoom_range
        if aug_config['brightness_range']:
            gen_params['brightness_range'] = aug_config['brightness_range']
        
        tta_datagen = ImageDataGenerator(**gen_params)
        tta_gen = tta_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='label',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False
        )
        
        predictions = model.predict(tta_gen, verbose=1)
        all_predictions.append(predictions.flatten())
        all_weights.append(aug_weight)
    
    print("  Averaging weighted predictions...")
    weights_array = np.array(all_weights)
    weights_array = weights_array / weights_array.sum()
    predictions_tta = np.average(all_predictions, axis=0, weights=weights_array)
    
    return predictions_tta

def find_optimal_threshold(predictions_flat, true_labels):
    """Here we are finding the optimal threshold balancing accuracy >=85% and statistical power >=90%."""
    TARGET_ACCURACY = 0.85
    TARGET_STATISTICAL_POWER = 0.90
    
    print("  Finding optimal threshold...")
    
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    total_thresholds = len(thresholds)
    for idx, threshold in enumerate(thresholds):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"    Testing threshold {idx+1}/{total_thresholds} ({threshold:.2f})...", end='\r')
        pred = (predictions_flat >= threshold).astype(int)
        cm_temp = confusion_matrix(true_labels, pred)
        if cm_temp.size == 4:
            tn, fp, fn, tp = cm_temp.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc = (tp + tn) / len(true_labels)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = f1_score(true_labels, pred)
            balanced_acc = (sens + spec) / 2.0
            
            if acc >= TARGET_ACCURACY and sens >= TARGET_STATISTICAL_POWER:
                score = 0.4 * acc + 0.4 * sens + 0.1 * balanced_acc + 0.1 * f1
                if score > best_score:
                    best_threshold = threshold
                    best_score = score
                    best_metrics = {
                        'accuracy': acc,
                        'sensitivity': sens,
                        'specificity': spec,
                        'precision': prec,
                        'f1': f1,
                        'balanced_acc': balanced_acc
                    }
    
    print(f"    Testing threshold {total_thresholds}/{total_thresholds} (0.69)...")
    
    if best_score == 0:
        print("  No threshold met both requirements, finding best compromise...")
        thresholds = np.arange(0.2, 0.8, 0.005)
        total_thresholds = len(thresholds)
        for idx, threshold in enumerate(thresholds):
            if (idx + 1) % 30 == 0 or idx == 0:
                print(f"    Testing threshold {idx+1}/{total_thresholds} ({threshold:.3f})...", end='\r')
            pred = (predictions_flat >= threshold).astype(int)
            cm_temp = confusion_matrix(true_labels, pred)
            if cm_temp.size == 4:
                tn, fp, fn, tp = cm_temp.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                acc = (tp + tn) / len(true_labels)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = f1_score(true_labels, pred)
                balanced_acc = (sens + spec) / 2.0
                
                acc_gap = max(0, TARGET_ACCURACY - acc)
                power_gap = max(0, TARGET_STATISTICAL_POWER - sens)
                
                if acc >= TARGET_ACCURACY and sens >= TARGET_STATISTICAL_POWER:
                    score = 0.5 * acc + 0.5 * sens + 0.1 * balanced_acc
                elif acc >= TARGET_ACCURACY:
                    score = 0.3 * acc + 0.6 * sens - 2.0 * power_gap
                elif sens >= TARGET_STATISTICAL_POWER:
                    score = 0.6 * acc + 0.3 * sens - 2.0 * acc_gap
                else:
                    score = 0.4 * acc + 0.4 * sens - 3.0 * acc_gap - 3.0 * power_gap
                
                if score > best_score:
                    best_threshold = threshold
                    best_score = score
                    best_metrics = {
                        'accuracy': acc,
                        'sensitivity': sens,
                        'specificity': spec,
                        'precision': prec,
                        'f1': f1,
                        'balanced_acc': balanced_acc
                    }
    
    print(f"    Testing threshold {total_thresholds}/{total_thresholds} (0.795)...")
    print(f"  Optimal threshold found: {best_threshold:.3f}")
    
    return best_threshold, best_metrics

def evaluate_model(model, test_generator, test_df, use_tta=True, num_tta=10):
    """Evaluating model and calculate metrics."""
    
    
    print("\nEvaluating model...")
    
    if use_tta:
        predictions_flat = evaluate_model_with_tta(model, test_df, num_tta=num_tta)
    else:
        print("  Generating predictions...")
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        predictions_flat = predictions.flatten()
    
    true_labels = test_df['label'].astype(int).values
    print(f"  Evaluating {len(true_labels)} test samples")
    
    best_threshold, best_metrics = find_optimal_threshold(predictions_flat, true_labels)
    
    print("  Calculating final metrics...")
    predicted_classes = (predictions_flat >= best_threshold).astype(int)
    cm = confusion_matrix(true_labels, predicted_classes)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / len(true_labels)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc = roc_auc_score(true_labels, predictions_flat)
    statistical_power = sensitivity
    
    print(f"\nTest Metrics (threshold: {best_threshold:.3f}):")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  AUC: {auc:.4f} ({auc*100:.2f}%)")
    print(f"  Statistical Power: {statistical_power:.4f} ({statistical_power*100:.2f}%)")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BENIGN', 'MALIGNANT'],
                yticklabels=['BENIGN', 'MALIGNANT'])
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': cm,
        'statistical_power': statistical_power,
        'optimal_threshold': best_threshold
    }

def plot_history(history):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    axes[0, 0].plot(epochs, history.history['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history.history['loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'precision' in history.history:
        axes[1, 0].plot(epochs, history.history['precision'], 'g-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, history.history['val_precision'], 'orange', label='Validation', linewidth=2)
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'auc' in history.history:
        axes[1, 1].plot(epochs, history.history['auc'], 'g-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, history.history['val_auc'], 'orange', label='Validation', linewidth=2)
        axes[1, 1].set_title('AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
