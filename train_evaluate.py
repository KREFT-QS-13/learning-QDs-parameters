import argparse
import torch
import numpy as np
import time

import utilities.config as c
import utilities.model_utils as mu
from models.transfer_CNN import ResNet
from models.vanilla_CNN import VanillaCNN, CustomCNN
from models.multihead_CNN import MultiBranchCNN

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument('--mode', type=int, default=1, 
                        help='Mode for preprocessing capacitance matrices (output size).')
    
    parser.add_argument('-K', type=int, required=True,  
                        help='The number of quantum dots in the system. Default vaule 2.')
   
    parser.add_argument('-S', type=int, required=True,  
                        help='The number of sensors in the system.')
    
    parser.add_argument('--system_name', type=str, required=False, default='',   
                        help='The name of the system, i.e. the name of the folder where the datasets is saved.')
    
    parser.add_argument('-GCM', action='store_true', default=False,
                       help='If set, use General CM (True), otherwise use Local CM (False)')
    args = parser.parse_args()

    mode = args.mode 
    maxwell_mode = args.GCM
    K = args.K
    S = args.S
    N = K - S
    system_name = args.system_name
    config_tuple = (K, N, S)
    print(f"Configuration tuple of the system: K={K}, N={N}, S={S}. System name: {system_name}.")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU. (device: {c.DEVICE})")
    else:
        print(f"CUDA is not available. Using CPU. (device: {c.DEVICE})")
    
    
    # Define model configurations
    model_configs = [
        {
            'model': ResNet, 
            'params': {
                'config_tuple': config_tuple,
                'base_model': 'resnet10',
                'name': f'Rn10-{N}-{S}-{system_name}',
                'pretrained': False,
                'dropout': 0.1,
                'custom_head': [4096, 512],
             }
        },
        # {
        #     'model': ResNet, 
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'base_model': 'resnet12',
        #         'name': f'Rn12-{N}-{S}-{system_name}',
        #         'pretrained': False,
        #         'dropout': 0.1,
        #         'custom_head': [4096, 512],
        #      }
        # },
        # {
        #     'model': ResNet, 
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'base_model': 'resnet14',
        #         'name': f'Rn14-{N}-{S}-{system_name}',
        #         'pretrained': False,
        #         'dropout': 0.1,
        #         'custom_head': [4096, 512],
        #      }
        # },
        # {
        #     'model': ResNet, 
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'base_model': 'resnet16',
        #         'name': f'Rn16-{N}-{S}-{system_name}',
        #         'pretrained': False,
        #         'dropout': 0.1,
        #         'custom_head': [4096, 512],
        #      }
        # },
        #         {
        #     'model': ResNet, 
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'base_model': 'resnet18',
        #         'name': f'Rn18-{N}-{S}-{system_name}',
        #         'pretrained': False,
        #         'dropout': 0.1,
        #         'custom_head': [4096, 512],
        #      }
        # },
        # {
        #     'model': ResNet, 
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'base_model': 'resnet10',
        #         'name': f'Rn10-{N}-{S}-{system_name}',
        #         'pretrained': False,
        #         'dropout': 0.1,
        #         'custom_head': [8192, 256],
        #      }
        # },

        # {
        #     'model': MultiBranchCNN,
        #     'params': {
        #         'config_tuple': config_tuple,
        #         'name': 'multibranch_model',
        #         'base_model': 'resnet18',
        #         'num_branches': 3,
        #         'num_attention_heads': 4,
        #         'custom_head': [2048, 1024],
        #         'prediction_head': [512, 256],
        #         'dropout': 0.1
        #     }
        # }
    ]

    # Define training parameters for each model
    train_params_list = [
        {   
            'batch_size': 32,
            'epochs': 5, 
            'learning_rate': 0.0001, 
            'val_split': 0.1,
            'test_split': 0.1,
            'random_state': 42,
            'epsilon':0.1,
            'init_weights': None,
        },
        # {   
        #     'batch_size': 32,
        #     'epochs': 30, 
        #     'learning_rate': 0.0001, 
        #     'val_split': 0.1,
        #     'test_split': 0.1,
        #     'random_state': 5,
        #     'epsilon':0.1,
        #     'init_weights': None,
        # },
        # {   
        #     'batch_size': 32,
        #     'epochs': 30, 
        #     'learning_rate': 0.0001, 
        #     'val_split': 0.1,
        #     'test_split': 0.1,
        #     'random_state': 8,
        #     'epsilon':0.1,
        #     'init_weights': None,
        # },
        # {   
        #     'batch_size': 32,
        #     'epochs': 30, 
        #     'learning_rate': 0.0001, 
        #     'val_split': 0.1,
        #     'test_split': 0.1,
        #     'random_state': 73,
        #     'epsilon':0.1,
        #     'init_weights': None,
        # },
        # {   
        #     'batch_size': 32,
        #     'epochs': 30, 
        #     'learning_rate': 0.0001, 
        #     'val_split': 0.1,
        #     'test_split': 0.1,
        #     'random_state': 997,
        #     'epsilon':0.1,
        #     'init_weights': None,
        # },
    ]  

    # train_params_list = train_params_list*len(model_configs)
    print(len(model_configs), len(train_params_list))
    old_size_tp = len(train_params_list)
    old_size_mc = len(model_configs)
    train_params_list = [tp for tp in train_params_list for _ in range(old_size_mc)]
    model_configs = model_configs*old_size_tp
    # train_params_list = train_params_list
    print(len(model_configs), len(train_params_list))
    assert len(model_configs) == len(train_params_list), "Number of model configurations and training parameters must match."
    
    start_time = time.time()
    # Load your data
    print("Loading and preparing datasets...")
    
    datasize_cut = 32000
    # for training with csd:
    X,y = mu.prepare_data(config_tuple, 
                          param_names=['csd', 'C_DD', 'C_DG'], 
                          all_batches=False,
                          batches=np.arange(1,33), 
                          datasize_cut=datasize_cut,
                          maxwell=maxwell_mode,
                          system_name=system_name)
    param_names_to_save= ['csd', 'C_DD', 'C_DG', f'{maxwell_mode}']
    

    # for testing with csd images:
    # X, y = mu.prepare_data(config_tuple, 
    #                        param_names=['csd', 'C_DD', 'C_DG'],
    #                        all_batches=False,
    #                        batches=np.arange(1,65), 
    #                        datasize_cut=datasize_cut,
    #                        maxwell=maxwell_mode,
    #                        system_name=system_name)
    # param_names_to_save = ['csd_gradient', 'C_DD', 'C_DG', f'{maxwell_mode}', f'{system_name}']
   
    print(f'Successfully prepared {len(X)} datapoints with input size {c.RESOLUTION}x{c.RESOLUTION}.\n')
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    
    # Combine model configurations with their respective training parameters
    models_configs = [{'model_config': mc, 'train_params': tp} for mc, tp in zip(model_configs, train_params_list)]

    # Train, evaluate, and save models
    for config in models_configs:  
        # summary(config['model_config']['model'](**config['model_config']['params']), (1, 1, c.RESOLUTION, c.RESOLUTION))
        results = mu.train_evaluate_and_save_models(config_tuple, 
                                                    [config['model_config']],
                                                    X, y,
                                                    param_names_to_save, 
                                                    config['train_params'])

    print("Training, evaluation, and saving complete!")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
