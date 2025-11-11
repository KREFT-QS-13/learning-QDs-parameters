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

def grid_search(config_tuple:tuple, datasize_cut:int=32000, save_dir:str='Results', system_name:str=''):
    """
    Perform grid search for ResNet hyperparameters.
    """
    K, N, S = config_tuple
    # For testing
    # param_grid = {
    #     'batch_size': [64],
    #     'learning_rate': [0.01],
    #     'base_model': ['resnet10'],
    #     'dropout': [0.25],
    #     'custom_head': [
    #         [1024, 512], 
    #     ]
    # }
    
    # Define parameter grids
    param_grid = {
        'base_model': ['resnet18', 'resnet10'], # 2
        'batch_size': [32], # 1
        'learning_rate': [0.0001], # 1
        'dropout': [0.1], # 1
        'custom_head': [
            [8192, 256],
        ] # 1
    }

    # Load data
    start_time = time.time()
    print("Loading and preparing datasets...")
    maxwell_mode = True
    X, y = mu.prepare_data(config_tuple, 
                          param_names=['csd', 'C_DD', 'C_DG'], 
                          all_batches=False,
                          batches=np.arange(1,33), 
                          datasize_cut=datasize_cut,
                          maxwell=maxwell_mode,
                          system_name=system_name)
    param_names_to_save = ['csd', 'C_DD', 'C_DG', f'{maxwell_mode}']
    print(f'Successfully prepared {len(X)} datapoints with input size {c.RESOLUTION}x{c.RESOLUTION}.\n')
    end_time = time.time()
    print(f"Time taken to prepare data: {end_time - start_time:.2f} seconds")

    # Training parameters that stay constant
    base_train_params = {
        'epochs': 30,
        'val_split': 0.1,
        'test_split': 0.1,
        'random_state': 42,
        'epsilon': 0.5,
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
                'config_tuple': config_tuple,
                'base_model': current_params['base_model'],
                'name': f"{current_params['base_model']}-{N}-{S}",
                'pretrained': False,
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
            results = mu.train_evaluate_and_save_models(config_tuple,
                                                        [model_config], 
                                                        X, y,
                                                        param_names_to_save,
                                                        train_params, 
                                                        save_dir=save_dir)
            
            # Get validation MSE from the results
            val_mse = min(results[0]['history']['val_mse'])
            
            # Update best configuration if needed
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_config = current_params
                # Get the path to the saved model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f"Results/{current_params['base_model']}/{current_params['base_model']}-{N}-{S}_{timestamp}"
                
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
    parser.add_argument('-K', type=int, required=True, help="The number of all quantum dots in the system (including sensors).")
    parser.add_argument('-S', type=int, required=True, help="Number of sensors in the system.")
    parser.add_argument('--system_name', type=str, default='', help="The name of the system, i.e. the name of the folder where the datasets is saved.")
    
    args = parser.parse_args()
    K = args.K
    S = args.S
    N = K - S
    save_dir = args.save_dir
    system_name = args.system_name
    config_tuple = (K, N, S)

    grid_search(config_tuple, save_dir=save_dir, system_name=system_name)
