import os
import sys
import json
import argparse
import random
import numpy as np
import tensorflow as tf

# Adding the parent directory to the path to import the common module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.data_loader import load_dataset
from train import train_model, TARGET_ACCURACY, TARGET_AUC, MAX_EPOCHS, create_generators
from evaluate import evaluate_model, plot_history
from model import build_model




def adaptive_training(train_df, test_df, weights_path='best_model.weights.h5', max_attempts=1):
    """Here we are training the model with single configuration."""
    config = {'learning_rate': 5e-4, 'dropout_rate': 0.4, 'dense_units': 256, 'use_focal_loss': True}
    
    model, history, test_gen, val_acc, val_auc = train_model(
        train_df, test_df, weights_path,
        resume_from_weights=False,
        **config
    )
    
    metrics = evaluate_model(model, test_gen, test_df)
    return model, history, test_gen, metrics

def load_model_with_config(weights_path):
    """Loading model with saved configuration."""
    config_path = weights_path.replace('.weights.h5', '_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = build_model(
            use_focal_loss=config['use_focal_loss'],
            learning_rate=config['learning_rate'],
            dropout_rate=config['dropout_rate'],
            dense_units=config['dense_units']
        )
    else:
        model = build_model()
    
    model.load_weights(weights_path)
    return model




def main():
    # Setting reproducibility for consistent results
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    
    
    parser = argparse.ArgumentParser(description='Breast Cancer Detection - Ensemble CNN')
    parser.add_argument('--mode', choices=['train', 'eval', 'auto'], default='auto',
                       help='train: train model, eval: evaluate only, auto: train if no weights exist')
    parser.add_argument('--weights', default='best_model.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--resume', action='store_true',
                       help='Continue training from existing weights')
    args = parser.parse_args()
    
    train_df, test_df = load_dataset()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: No data loaded. Check dataset paths.")
        return
    
    weights_exist = os.path.exists(args.weights)
    
    should_train = False
    if args.mode == 'train':
        if not weights_exist or args.resume:
            should_train = True
    elif args.mode == 'eval':
        should_train = False
    else:
        should_train = not weights_exist
    
    model = None
    history = None
    test_gen = None
    metrics = None
    
    
    if should_train:
        # Here we are training the model
        if args.resume:
            model, history, test_gen, val_acc, val_auc = train_model(
                train_df, test_df, args.weights, resume_from_weights=True
            )
            _, _, test_gen, _, _ = create_generators(train_df, test_df)
            metrics = evaluate_model(model, test_gen, test_df)
        else:
            model, history, test_gen, metrics = adaptive_training(
                train_df, test_df, args.weights, max_attempts=1
            )
    else:
        # Here we are loading the model
        if not weights_exist:
            print(f"Error: Weights file '{args.weights}' not found. Run with --mode train first.")
            return
        
        model = load_model_with_config(args.weights)
        _, _, test_gen, _, _ = create_generators(train_df, test_df)
        metrics = evaluate_model(model, test_gen, test_df)
    
    if history is not None:
        plot_history(history)
    
    if metrics:
        print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Test AUC: {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
        print(f"Statistical Power: {metrics['statistical_power']:.4f} ({metrics['statistical_power']*100:.2f}%)")

if __name__ == '__main__':
    main()
