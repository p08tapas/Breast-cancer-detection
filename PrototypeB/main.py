import os
import sys
import argparse
import random
import numpy as np
import tensorflow as tf


# Adding the parent directory to the path to import the common module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.data_loader import load_dataset
from train import train_model
from evaluate import evaluate_model, plot_history
from model import build_model



def main():
    # Setting reproducibility for consistent results
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    parser = argparse.ArgumentParser(description='Breast Cancer Detection - Transfer Learning')
    parser.add_argument('--mode', choices=['train', 'eval', 'auto'], default='auto',
                       help='train: train model, eval: evaluate only, auto: train if no weights exist')
    parser.add_argument('--weights', default='best_model.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--resume', action='store_true',
                       help='If training, continue from existing weights instead of starting fresh')
    args = parser.parse_args()
    
    train_df, test_df = load_dataset()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: No data loaded. Check dataset paths.")
        return
    
    should_train = False
    if args.mode == 'train':
        should_train = True
    elif args.mode == 'eval':
        should_train = False
    else:
        should_train = not os.path.exists(args.weights)
    
    model = None
    history = None
    test_gen = None
    
    
    
    
    if should_train:
        model, history, test_gen = train_model(train_df, test_df, args.weights, resume_from_weights=args.resume)
        from sklearn.model_selection import train_test_split
        train_labels_int = train_df['label'].astype(int)
        _, val_split = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_labels_int)
        from train import create_generators
        _, val_gen, _, _, _ = create_generators(train_df, test_df)
    
    
    else:
        if not os.path.exists(args.weights):
            print(f"Error: Weights file '{args.weights}' not found. Run with --mode train first.")
            return
        
        model = build_model()
        model.load_weights(args.weights)
        from train import create_generators
        _, val_gen, test_gen, _, _ = create_generators(train_df, test_df)
        from sklearn.model_selection import train_test_split
        train_labels_int = train_df['label'].astype(int)
        _, val_split = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_labels_int)
    
    metrics = evaluate_model(model, test_gen, test_df, val_gen, val_split)
    
    
    
    
    if history is not None:
        plot_history(history)
    
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Test AUC: {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    print(f"Statistical Power: {metrics['statistical_power']:.4f} ({metrics['statistical_power']*100:.2f}%)")

if __name__ == '__main__':
    main()

