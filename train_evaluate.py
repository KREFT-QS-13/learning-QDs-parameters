import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import argparse 
import os
import json
from datetime import datetime

import csv
import h5py

import utilities.config as c
import utilities.model_utils as mu
from models.transfer_CNN import TransferLearningCNN
from models.vanilla_CNN import VanillaCNN
from utilities.utils import ensure_dir_exists


def divide_dataset(batch_size, val_split, test_split, random_state):
     # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_split, random_state=random_state)

    # Create DataLoader objects
    train_dataset = TensorDataset(torch.FloatTensor(X_train).to(c.DEVICE), torch.FloatTensor(y_train).to(c.DEVICE))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(X_val).to(c.DEVICE), torch.FloatTensor(y_val).to(c.DEVICE))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.FloatTensor(X_test).to(c.DEVICE), torch.FloatTensor(y_test).to(c.DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, X, y, batch_size=32, epochs=100, learning_rate=0.001, val_split=0.2, test_split=0.1, random_state=42, epsilon=1.0, init_weights=None):
    '''
        Train a model on the given data and hyperparameters.
    '''
    print(f"\nUsing device: {c.DEVICE}")

    # Move model to GPU
    model = model.to(c.DEVICE)

    train_loader, val_loader, test_loader = divide_dataset(batch_size, val_split, test_split, random_state)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'vec_local_train_accuracy': [],
        'vec_local_val_accuracy': [],
        'train_mse': [],
        'val_mse': [],
        'L2 norm': []
    }

    # If we're continuing training, load the previous history
    if init_weights:
        history_path = os.path.join(os.path.dirname(init_weights), 'results.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                prev_history = json.load(f)['history']
            for key in history:
                history[key] = prev_history.get(key, [])

            print(f"Last epoch: Tr. Loss: {history['train_losses'][-1]:.5f}, Val. Loss: {history['val_losses'][-1]:.5f}\n", 
                f"{'':<11}Tr. MSE: {history['train_mse'][-1]:.3f}, Val. MSE: {history['val_mse'][-1]:.3f}")
              

    for epoch in range(epochs): 
        model.train()
        
        train_loss = 0.0
        global_correct_predictions = 0
        local_correct_predictions = 0
        vec_local_correct_predictions = None
        total_predictions = 0
        all_train_outputs = []
        all_train_targets = []

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            predicted_values = outputs.detach().cpu().numpy()
            true_values = targets.cpu().numpy()

            correct = np.isclose(predicted_values, true_values, atol=epsilon)
            if vec_local_correct_predictions is None:
                vec_local_correct_predictions = np.sum(correct, axis=0)
            else:
                vec_local_correct_predictions += np.sum(correct, axis=0)
            
            global_correct_predictions += np.sum(np.all(correct, axis=1))
            local_correct_predictions += np.sum(correct)/len(true_values)

            total_predictions += len(targets)
            all_train_outputs.append(predicted_values)
            all_train_targets.append(true_values)

        avg_train_loss = train_loss / len(train_loader)
        
        global_train_accuracy = global_correct_predictions / total_predictions if total_predictions > 0 else 0
        local_train_accuracy = local_correct_predictions / total_predictions if total_predictions > 0 else 0
        vec_local_train_accuracy = vec_local_correct_predictions / total_predictions if total_predictions > 0 else np.zeros_like(vec_local_correct_predictions)

        
        all_train_outputs = np.concatenate(all_train_outputs)
        all_train_targets = np.concatenate(all_train_targets)
        train_mse = mean_squared_error(all_train_targets, all_train_outputs)

        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append([global_train_accuracy, local_train_accuracy])
        history['vec_local_train_accuracy'].append(vec_local_train_accuracy)
        history['train_mse'].append(train_mse)

        # Validation step:
        val_loss, global_val_accuracy, local_val_accuracy, val_mse, _, val_vec_local_acc = evaluate_model(model, val_loader, criterion, epsilon)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append([global_val_accuracy, local_val_accuracy])
        history['vec_local_val_accuracy'].append(val_vec_local_acc)
        history['val_mse'].append(val_mse)

        print(f"Epoch {epoch+1}/{epochs}: Tr. Loss: {avg_train_loss:.5f}, Val. Loss: {val_loss:.5f}")
        print(f"{'':<11} Tr. Acc.: {100*global_train_accuracy:.2f}% ({100*local_train_accuracy:.2f}%), "
              f"Val. Acc.: {100*global_val_accuracy:.2f}% ({100*local_val_accuracy:.2f}%)")
        print(f"{'':<11} Vec. Tr. Local Acc.: {np.round(100*vec_local_train_accuracy, 2)}%")
        print(f"{'':<11} Vec. Val. Local Acc.: {np.round(100*val_vec_local_acc, 2)}%")
        print(f"{'':<11} Tr. MSE: {train_mse:.5f}, Val. MSE: {val_mse:.5f}")

    return model, history, test_loader

def evaluate_model(model, dataloader, criterion=nn.MSELoss(), epsilon=1.0):
    model.eval()

    global_correct_predictions = 0
    local_correct_predictions = 0
    vec_local_correct_predictions = None
    total_predictions = 0
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    predictions = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            predicted_values = outputs.cpu().numpy()
            true_values = targets.cpu().numpy()

            predictions.append([inputs.cpu().numpy(), true_values, predicted_values])

            correct = np.isclose(predicted_values, true_values, atol=epsilon)
            if vec_local_correct_predictions is None:
                vec_local_correct_predictions = np.sum(correct, axis=0)
            else:
                vec_local_correct_predictions += np.sum(correct, axis=0)
            
            global_correct_predictions += np.sum(np.all(correct, axis=1))
            local_correct_predictions += np.sum(correct)/len(true_values)


            total_predictions += len(targets)
            all_outputs.append(predicted_values)
            all_targets.append(true_values)

    avg_loss = total_loss / len(dataloader)
    
    global_avg_accuracy = global_correct_predictions / total_predictions if total_predictions > 0 else 0
    local_avg_accuracy = local_correct_predictions / total_predictions if total_predictions > 0 else 0
    vec_local_avg_accuracy = vec_local_correct_predictions / total_predictions if total_predictions > 0 else np.zeros_like(vec_local_correct_predictions)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    mse = mean_squared_error(all_targets, all_outputs)

    return avg_loss, global_avg_accuracy, local_avg_accuracy, mse, predictions, vec_local_avg_accuracy

def collect_performance_metrics(model, test_loader):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(c.DEVICE), targets.to(c.DEVICE)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    r2 = r2_score(all_targets, all_outputs)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }
    return metrics


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def save_results_and_history(result, history,  predictions, save_dir):
    ensure_dir_exists(save_dir)
    # Save as HDF5
    with h5py.File(os.path.join(save_dir, 'results_and_history.h5'), 'w') as f:
        f.create_dataset('test_loss', data=result['test_loss'])
        f.create_dataset('test_accuracy_global', data=result['global_test_accuracy'])
        f.create_dataset('test_accuracy_local', data=result['local_test_accuracy'])

        for k, v in result['metrics'].items():
            f.create_dataset(f'metrics/{k}', data=v)
        for k, v in history.items():
            f.create_dataset(f'history/{k}', data=v)
        
    inputs_array = np.concatenate([p[0] for p in predictions])
    targets_array = np.concatenate([p[1] for p in predictions])
    outputs_array = np.concatenate([p[2] for p in predictions])

    with h5py.File(os.path.join(save_dir, 'predictions.h5'), 'w') as hf:
        hf.create_dataset('inputs', data=inputs_array)
        hf.create_dataset('targets', data=targets_array)
        hf.create_dataset('outputs', data=outputs_array)

    # Save as CSV
    with open(os.path.join(save_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Loss', result['test_loss']])
        writer.writerow(['Test Accuracy Global', result['global_test_accuracy']])
        writer.writerow(['Test Accuracy Local', result['local_test_accuracy']])
        for k, v in result['metrics'].items():
            writer.writerow([k, v])

def save_model(model, save_dir, model_name):
    ensure_dir_exists(save_dir)
    # Save in PyTorch's native format
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}.pth'))

def train_evaluate_and_save_models(model_configs, X, y, train_params, save_dir='Results'):
    """
    Train, evaluate, and save multiple models based on the given configurations.
    
    Args:
        model_configs (list): List of model configurations.
        X (np.array): Input data.
        y (np.array): Target data.
        train_params (dict): Parameters for training.
        save_dir (str): Directory to save results.
    
    Returns:
        list: Results including model, history, and evaluation metrics for each model.
    """
    results = []
    for config in model_configs:
        model = config['model'](**config['params'])
        model_name = config['params']['name']
        base_model = config['params'].get('base_model', 'default')
        
        # Create a unique directory for this run under the base model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = os.path.join(save_dir, base_model, f"{model_name}_{timestamp}")
        ensure_dir_exists(model_save_dir)
        
        # Load initial weights if specified
        if train_params.get('init_weights'):
            init_weights_path = train_params['init_weights']
            if os.path.exists(init_weights_path):
                model = mu.load_model_weights(model, init_weights_path)
                print(f"Loaded initial weights from {init_weights_path}")
            else:
                print(f"Warning: Initial weights file not found at {init_weights_path}. Starting with random weights.")
        
        # Train the model
        trained_model, history, test_loader = train_model(model, X, y, **train_params)
        
        # Evaluate the model
        test_loss, global_test_accuracy, local_test_accuracy, test_mse, predictions, vec_local_test_accuracy = evaluate_model(trained_model, test_loader, epsilon=train_params.get('epsilon', 1.0))
        
        # Collect performance metrics on the test set
        metrics = collect_performance_metrics(trained_model, test_loader)
        
        print(f"Evaluation: Test Accuracy (Global): {global_test_accuracy:.2f}%, Test Accuracy (Local): {local_test_accuracy:.2f}%")
        print(f"Evaluation: Test Loss: {test_loss:.5f}, Test MSE: {test_mse:.5f}")
        print(f"Evaluation: Vec. Test Local Acc.: {np.round(100*vec_local_test_accuracy, 2)}%")
        print(f"Evaluation: MSE: {metrics['MSE']:.2f}, MAE: {metrics['MAE']:.2f}, R2: {metrics['R2']:.2f}\n")

        # Extract targets and outputs from predictions
        targets = np.concatenate([p[1] for p in predictions])
        outputs = np.concatenate([p[2] for p in predictions])
        
        # Create and save the L2 norm polar plot
        plot_l2_norm_polar(targets, outputs, model_save_dir)
        
        # Save the model
        model_path = os.path.join(model_save_dir, f"{model_name}.pth")
        mu.save_model_weights(trained_model, model_path)
        
        # Save the history and metrics
        result = {
            'config': {
                'model_name': config['model'].__name__,
                'params': config['params']
            },
            'input_shape': X.shape[1:],
            'output_shape': y.shape[1:],
            'dataset_size': len(X),
            'train_params': {k: v for k, v in train_params.items()},
            'history': {k: v for k, v in history.items() if k != 'L2 norm'},
            'test_loss': float(test_loss),
            'global_test_accuracy': float(global_test_accuracy),
            'local_test_accuracy': float(local_test_accuracy),
            'metrics': {k: float(v) for k, v in metrics.items()},
        }
        
        with open(os.path.join(model_save_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
        
        plot_learning_curves(history, result, model_save_dir)
        
        results.append(result)
        
        mu.save_results_to_csv([result])
        
        save_results_and_history(result, history, predictions, model_save_dir)
        save_model(trained_model, model_save_dir, model_name)
        
    return results

def plot_learning_curves(history, result, save_dir):
    ensure_dir_exists(save_dir)
    plt.figure(figsize=(20, 12))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss vs. Epochs ({result["train_params"]["batch_size"]}, {result["train_params"]["epochs"]}, {result["train_params"]["learning_rate"]})')
    plt.legend(fontsize=12)
    
    # MSE plot
    plt.subplot(2, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE vs. Epochs')
    plt.legend(fontsize=12)

    # Global Accuracy plot
    plt.subplot(2, 2, 3)
    plt.plot([acc[0] for acc in history['train_accuracies']], label='Train Acc.')
    plt.plot([acc[0] for acc in history['val_accuracies']], label='Validation Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Global Accuracy')
    plt.title('Global Accuracy vs. Epochs')
    plt.legend(fontsize=12)
    
    # Local Accuracy plot
    plt.subplot(2, 2, 4)
    plt.plot([acc[1] for acc in history['train_accuracies']], label='Train Acc.')
    plt.plot([acc[1] for acc in history['val_accuracies']], label='Validation Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Local Accuracy')
    plt.title('Local Accuracy vs. Epochs')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_l2_norm_polar(targets, outputs, save_dir, num_points=None):
    ensure_dir_exists(save_dir)
    '''
    Create a plot in polar coordinates with points as distances between
    the origin and the L2 norm of the difference of targets and outputs.
    The plot will have concentric circles at integer radii and no angle labels.
    Colors and shapes represent distance groups from the origin.

    Args:
        targets (np.array): The true values.
        outputs (np.array): The predicted values from the model.
        save_dir (str): Directory to save the plot.
        num_points (int, optional): Number of points to plot. If None, all points are plotted.

    Returns:
        None
    '''
    # Calculate L2 norms
    l2_norms = np.linalg.norm(targets - outputs, axis=1)
    
    # If num_points is specified and less than the total number of points, randomly sample
    if num_points is not None and num_points < len(l2_norms):
        indices = np.random.choice(len(l2_norms), num_points, replace=False)
        l2_norms = l2_norms[indices]
    else:
        num_points = len(l2_norms)
    
    # Create angles for each point (evenly spaced)
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 14), subplot_kw=dict(projection='polar'))
    
    # Define color groups and shapes
    color_groups = np.minimum(np.floor(l2_norms), 5)
    shapes = ['o', 's', '^', 'D', 'p', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    # Calculate percentages for each L2 range
    percentages = [np.sum(color_groups == i) / len(color_groups) * 100 for i in range(6)]
    
    # if np.max(l2_norms) > 10:
    #     increment = 1
    #     base = 5
    # elif np.max(l2_norms) > 5:
    #     increment = 0.5
    #     base = 5
    # elif np.max(l2_norms) > 1:
    #     increment = 0.25
    #     base = 0


    # Plot the points
    for i in range(6):
        mask = color_groups == i
        if i < 5:
            label = f'{i:.0f} ≤ L2 < {i+1:.0f} ({percentages[i]:.1f}%)'
        else:
            label = f'L2 ≥ 5 ({percentages[i]:.1f}%)'
        ax.scatter(theta[mask], l2_norms[mask], c=[colors[i]], marker=shapes[i], label=label, alpha=0.8)
    
    # Set the rmax to be the ceiling of the max L2 norm
    rmax = np.ceil(np.max(l2_norms))
    ax.set_rmax(rmax)
    
    # Set the rticks to be integer values from 1 to rmax
    ax.set_rticks(np.concatenate((np.arange(1, 5), np.arange(5, rmax + 1, 5))))
    
    # Remove the angle labels
    ax.set_xticklabels([])
    
    # Add labels and title
    ax.set_title(f"L2 Norm of Target-Output Difference ({num_points} points)", fontsize=22)
    
    # Add legend for shapes and colors
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="L2 Norm Groups", fontsize=18)
    plt.setp(legend.get_title(), fontsize=16)
    
    # Adjust layout manually
    plt.subplots_adjust(right=0.75, bottom=0.05, top=0.95)
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'l2_norm_polar_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"L2 norm polar plot saved in {save_dir}")

# Example usage:
if __name__ == "__main__":
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
    # X, y = mu.prepare_data()
    X, y = mu.prepare_data(all_batches=False, batches=np.arange(1,5)) # for testing
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
            'batch_size': 512, # 32, 64
            'epochs': 100, #5, 20, 50, 100
            'learning_rate': 0.001, #0.0005, 0.0001, 0.005, 0.001
            'val_split': 0.1,
            'test_split': 0.1,
            'random_state': 42,
            'epsilon': 1,
            'init_weights': None,
        },
    ]  

    # Combine model configurations with their respective training parameters
    models_configs = [{'model_config': mc, 'train_params': tp} for mc, tp in zip(model_configs, train_params_list)]

    # Train, evaluate, and save models
    for config in models_configs:
        results = train_evaluate_and_save_models([config['model_config']], X, y, config['train_params'])

    print("Training, evaluation, and saving complete!")