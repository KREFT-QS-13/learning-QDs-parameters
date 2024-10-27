import argparse
import torch

import utilities.config as c
import utilities.model_utils as mu
from models.transfer_CNN import TransferLearningCNN
from models.vanilla_CNN import VanillaCNN


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument('--mode', type=int, default=1, help='Mode for preprocessing capacitance matrices (output size).')
    args = parser.parse_args()

    c.set_global_MODE(args.mode)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Load your data
    print("Loading and preparing datasets...")
    datasize = 5000
    X, y = mu.prepare_data()
    # X, y = mu.prepare_data(all_batches=False, batches=np.arange(1,5)) # for testing
    print(f'Successfully prepared {len(X)} datapoints with input size {c.RESOLUTION}x{c.RESOLUTION}.\n')

    # Define model configurations
    model_configs = [
        # {'model': VanillaCNN, 'params': {'name': 'VanillaCNN'}},        
        {'model': TransferLearningCNN, 'params': {'name': 'resnet18_model', 'base_model': 'resnet18', 'pretrained': True}},
        # {'model': TransferLearningCNN, 'params': {'name': 'resnet34_model', 'base_model': 'resnet34', 'pretrained': True}},
    ]

    # Define training parameters for each model
    train_params_list = [
        {   
            'batch_size': 32, # 32, 64
            'epochs': 10, #5, 20, 50, 100
            'learning_rate': 0.0001, #0.0005, 0.0001, 0.005, 0.001
            'val_split': 0.2,
            'test_split': 0.2,
            'random_state': 42,
            'epsilon': 0.1,
            'init_weights': None,
        },
    ]  

    # Combine model configurations with their respective training parameters
    models_configs = [{'model_config': mc, 'train_params': tp} for mc, tp in zip(model_configs, train_params_list)]

    # Train, evaluate, and save models
    for config in models_configs:
        results = mu.train_evaluate_and_save_models([config['model_config']], X, y, config['train_params'])

    print("Training, evaluation, and saving complete!")

if __name__ == "__main__":
    main()