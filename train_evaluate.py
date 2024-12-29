import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import time

import utilities.config as c
import utilities.model_utils as mu
from models.transfer_CNN import ResNet
from models.vanilla_CNN import VanillaCNN, CustomCNN


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument('--mode', type=int, default=1, 
                        help='Mode for preprocessing capacitance matrices (output size).')
    
    parser.add_argument('-K', type=int, required=True,  
                        help='The number of quantum dots in the system. Default vaule 2.')
   
    parser.add_argument('-S', type=int, required=True,  
                        help='The number of sensors in the system.')
    
    args = parser.parse_args()

    mode = args.mode
    K = args.K
    S = args.S
    N = K - S
    config_tuple = (K, N, S)
    print(f"Configuration tuple of the system: K={K}, N={N}, S={S}.")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU. (device: {c.DEVICE})")
    else:
        print(f"CUDA is not available. Using CPU. (device: {c.DEVICE})")
    
    
    # Define model configurations
    model_configs = [
        # {
        #     'model': ResNet, 
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'name': 'resnet18_model',
        #         'base_model': 'resnet18',
        #         'pretrained': True,
        #         'dropout': 0.25,
        #         'custom_head': [2048, 1024],
        #         'filters_per_layer': None,
        #     }
        # },
        {
            'model': ResNet, 
            'params': {
                'config_tuple': config_tuple,
                'name': 'resnet12_model',
                'base_model': 'resnet12',
                'pretrained': True,
                'dropout': 0.1,
                'custom_head': [2048, 1024],
            }
        },
    ]

    model_configs = model_configs*2

    # Define training parameters for each model
    train_params_list = [
        # {   
        #     'batch_size': 16, # 32, 64
        #     'epochs': 50, #5, 20, 50, 100
        #     'learning_rate': 0.0005, #0.0005, 0.0001, 0.005, 0.001
        #     'val_split': 0.2,
        #     'test_split': 0.2,
        #     'random_state': 42,
        #     'epsilon':0.5,
        #     'init_weights': None,
        # },
        # {   
        #     'batch_size': 128,
        #     'epochs': 50, 
        #     'learning_rate': 0.005,
        #     'val_split': 0.2,
        #     'test_split': 0.2,
        #     'random_state': 42,
        #     'epsilon': 0.5,
        #     'load_conv_only': True,
        #     'init_weights': "./Results/resnet18/resnet18_model_20241122_155316/resnet18_model.pth", 
        #     'regularization_coeff': 0.8,
        #     'criterion': nn.MSELoss(),
        # },
        # {   
        #     'batch_size': 64,
        #     'epochs': 10, 
        #     'learning_rate': 0.001,
        #     'val_split': 0.2,
        #     'test_split': 0.2,
        #     'random_state': 42,
        #     'epsilon': 0.5,
        #     'init_weights': None, 
        #     'regularization_coeff':  1.25,
        #     'criterion': nn.MSELoss(),
        # },
        {   
            "batch_size": 64,
            "epochs": 25,
            "learning_rate": 0.001,
            "val_split": 0.2,
            "test_split": 0.2,
            "random_state": 42,
            "epsilon": 0.1,
            "init_weights": None,
            "regularization_coeff": 0.0,
            "criterion": nn.MSELoss(),
         },  
        {   
            "batch_size": 128,
            "epochs": 25,
            "learning_rate": 0.001,
            "val_split": 0.2,
            "test_split": 0.2,
            "random_state": 42,
            "epsilon": 0.1,
            "init_weights": None,
            "regularization_coeff": 0.0,
            "criterion": nn.MSELoss(),
         },           
    ]  
    # train_params_list = train_params_list
    assert len(model_configs) == len(train_params_list), "Number of model configurations and training parameters must match."
    
    start_time = time.time()
    # Load your data
    print("Loading and preparing datasets...")
    
    datasize_cut = 64000
    # for training with csd images:
    X, y = mu.prepare_data(config_tuple, datasize_cut=datasize_cut, param_names=['csd', 'C_DD', 'C_DG'])
    param_names = ['csd', 'C_DD', 'C_DG']
    
    
    ## For training with csd gradients:
    # X, y = mu.prepare_data(config_tuple, datasize_cut=datasize_cut, param_names=['csd_gradient', 'C_DD', 'C_DG'])
    # param_names = ['csd_gradient', 'C_DD', 'C_DG']

    # for testing with csd images:
    # X, y = mu.prepare_data(config_tuple, all_batches=False, batches=np.arange(1,5))
    # param_names = ['csd', 'C_DD', 'C_DG']
   
    print(f'Successfully prepared {len(X)} datapoints with input size {c.RESOLUTION}x{c.RESOLUTION}.\n')
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    
    # Combine model configurations with their respective training parameters
    models_configs = [{'model_config': mc, 'train_params': tp} for mc, tp in zip(model_configs, train_params_list)]

    # Train, evaluate, and save models
    for config in models_configs:  
        # summary(config['model_config']['model'](**config['model_config']['params']), (1, 1, c.RESOLUTION, c.RESOLUTION))
        results = mu.train_evaluate_and_save_models(config_tuple, [config['model_config']], X, y, param_names, config['train_params'])

    print("Training, evaluation, and saving complete!")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
