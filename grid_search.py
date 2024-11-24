import sys
import argparse
import os
import itertools
import time
import numpy as np
from datetime import datetime

sys.path.append('..')
import utilities.config as c
import utilities.model_utils as mu
from models.transfer_CNN import ResNet

def grid_search(config_tuple:tuple, datasize_cut:int=35000, save_dir:str='Results'):
    """
    Perform grid search for ResNet hyperparameters.
    """
    K, N, S = config_tuple
    # For testing
    # param_grid = {
    #     'batch_size': [64],
    #     'learning_rate': [0.01],
    #     'base_model': ['resnet18'],
    #     'dropout': [0.7],
    #     'custom_head': [
    #         [1024, 512], 
    #     ]
    # }
    
    # Define parameter grids
    param_grid = {
        'base_model': ['resnet18'], # 1
        'batch_size': [64, 128, 256, 512], # 4
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005], # 4
        'dropout': [0.0, 0.25, 0.5], # 3
        'custom_head': [
            [2048, 1024],
            [4096, 512],
            [2048, 512, 128],
        ] # 3
    }

    # Load data
    start_time = time.time()
    print("Loading and preparing datasets...")
    X, y = mu.prepare_data(config_tuple, datasize_cut=datasize_cut)
    print(f'Successfully prepared {len(X)} datapoints.\n')
    end_time = time.time()
    print(f"Time taken to prepare data: {end_time - start_time:.2f} seconds")

    # Training parameters that stay constant
    base_train_params = {
        'epochs': 50,
        'val_split': 0.15,
        'test_split': 0.15,
        'random_state': 42,
        'epsilon': 0.05,
        'init_weights': None
    }

    # Track best configuration
    best_val_mse = float('inf')
    best_config = None
    best_model_path = None

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = np.prod([len(v) for v in param_values])
    
    start_time = time.time()
    print(f"Starting grid search with {total_combinations} combinations...")
    print(f"Device: {c.DEVICE}")
    for i, values in enumerate(itertools.product(*param_values)):
        current_params = dict(zip(param_names, values))
        print(f"\nTesting combination {i+1}/{total_combinations}:")
        print(current_params)
        
        # Create model configuration
        model_config = {
            'model': ResNet,
            'params': {
                'K': K,  # Adjust based on your needs
                'name': f"{current_params['base_model']}_model",
                'base_model': current_params['base_model'],
                'pretrained': True,
                'dropout': current_params['dropout'],
                'custom_head': current_params['custom_head']
            }
        }
        
        # Create training parameters
        train_params = base_train_params.copy()
        train_params.update({
            'batch_size': current_params['batch_size'],
            'learning_rate': current_params['learning_rate']
        })
        
        # Train and evaluate model
        try:
            results = mu.train_evaluate_and_save_models([model_config], X, y, train_params, save_dir=save_dir)
            
            # Get validation MSE from the results
            val_mse = min(results[0]['history']['val_mse'])
            
            # Update best configuration if needed
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_config = current_params
                # Get the path to the saved model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f"Results/{current_params['base_model']}/{current_params['base_model']}_model_{timestamp}"
                
                print("\nNew best configuration found!")
                print(f"Validation MSE: {val_mse}")
                print(f"Configuration: {current_params}")
                print(f"Model saved at: {best_model_path}")
            
        except Exception as e:
            print(f"Error with configuration {current_params}: {e}")
            continue
    end_time = time.time()
    # Print final results
    print("\n" + "="*50)
    print("Grid Search Results")
    print("="*50)
    print(f"Best Validation MSE: {best_val_mse}")
    print("\nBest Configuration:")
    for param, value in best_config.items():
        print(f"{param}: {value}")
    print(f"\nBest model saved at: {best_model_path}")
    print(f"Time taken to perform grid search: {end_time - start_time:.2f} seconds ({(end_time - start_time)/3600:.2f} hours).")
    print(f"Time taken to perform grid search per configuration: {(end_time - start_time)/total_combinations:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for ResNet hyperparameters.")
    parser.add_argument('--save_dir', type=str, default='Results', help="Directory to save the results.")
    parser.add_argument('-K', type=int, default=2, help="The number of all quanutum dots in the system (including sensors).")
    parser.add_argument('-S', type=int, default=0, help="Number of sensors in the system.")

    args = parser.parse_args()
    K = args.K
    S = args.S
    save_dir = args.save_dir
    config_tuple = (K, K-S, S) # (K, N, S)

    grid_search(config_tuple, save_dir=save_dir)
