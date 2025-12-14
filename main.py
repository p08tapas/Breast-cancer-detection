import os
import argparse
from data_loader import load_dataset
from train import train_model
from evaluate import evaluate_model, plot_history
from model import build_model

def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Detection - Transfer Learning')
    parser.add_argument('--mode', choices=['train', 'eval', 'auto'], default='auto',
                       help='train: train model, eval: evaluate only, auto: train if no weights exist')
    parser.add_argument('--weights', default='best_model.weights.h5',
                       help='Path to model weights file')
    parser.add_argument('--resume', action='store_true',
                       help='If training, continue from existing weights instead of starting fresh')
    args = parser.parse_args()
    
    print("Breast Cancer Detection - Transfer Learning CNN")
     
   
    # Loading the data
    train_df, test_df = load_dataset()
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("Error: No data loaded. Check dataset paths.")
        return
    
      
    # Here it will determine if training is needed
    should_train = False
    if args.mode == 'train':
        should_train = True
    elif args.mode == 'eval':
        should_train = False
    else:  
        if not os.path.exists(args.weights):
            should_train = True
        else:
           
            # Will check if weights are compatible by trying to load them
            test_model = build_model()
            test_model.load_weights(args.weights)
            should_train = False  # Weights are compatible
            
    
    model = None
    history = None
    test_gen = None
    
    if should_train:
        print("\n[MODE: TRAIN]")
        model, history, test_gen = train_model(train_df, test_df, args.weights, resume_from_weights=args.resume)
    else:
        print("\n[MODE: EVAL]")
        if not os.path.exists(args.weights):
            print(f"Error: Weights file '{args.weights}' not found. Run with --mode train first.")
            return
        
        print(f"Loading model weights from {args.weights}...")
        model = build_model()
        
       
        # This will try to load weights and handle incompatible weights 
        model.load_weights(args.weights)
        print("Weights loaded successfully.")
             
        # Creating test generator
        from train import create_generators
        _, _, test_gen, _, _ = create_generators(train_df, test_df)
    
    
    # Evaluate
    print("\n")
    metrics = evaluate_model(model, test_gen, test_df)
    
    # This will plot history if it is available
    if history is not None:
        plot_history(history)
    
   
    # Summary
    print("\nFINAL SUMMARY")
    print(f"Test Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Test AUC:           {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    print(f"Statistical Power:  {metrics['statistical_power']:.4f} ({metrics['statistical_power']*100:.2f}%)")

if __name__ == '__main__':
    main()

