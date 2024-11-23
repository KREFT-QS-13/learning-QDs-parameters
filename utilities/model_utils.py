import sys
import h5py, os, csv, json
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

import matplotlib.pyplot as plt

import utilities.config as c
import utilities.utils as u
sys.path.append('./qdarts')
from qdarts.experiment import Experiment
from qdarts.plotting import plot_polytopes

from models.transfer_CNN import ResNet
from models.vanilla_CNN import VanillaCNN


# ----------------------------- LOAD DATA
def load_datapoints(config_tuple, param_names:list, all_batches=True, batches:list=None):
    """
    Args:
        param_names - the list of the parameters' names to load from .h5 file
        all_batches - if True the function loads all available batches
        batches - if all_batches=False this should pass the list of batch numbers to load
    
    Returns:
        A dictionary where keys are param_names and values are lists of elements from all batches
    """
    if all_batches and batches is None:
        batches_nums = np.arange(1, u.count_directories_in_folder(config_tuple)+1)
    elif all_batches==False and batches is not None:
        if all(isinstance(b, (int, np.integer)) and b > 0 for b in batches):
            max_batch = u.count_directories_in_folder(config_tuple)
            if all(b <= max_batch for b in batches):
                batches_nums = batches
            else:
                raise ValueError(f"Some batch numbers are greater than the total number of directories ({max_batch})")
        else:
            raise ValueError("Batches must be a list of positive integers")
    else:
        raise ValueError("Batches not defined properly, both all_batches and batches activated.")
      
    # all_groups_data = {param:[] for param in param_names}
    all_groups_data = []
    print(f"Loading batches: {len(batches_nums)} from {u.get_path_hfd5(batches_nums[0], config_tuple)}")
    for b in batches_nums:
        with h5py.File(u.get_path_hfd5(b, config_tuple), 'r') as f:
            def process_group(name, obj):
                if isinstance(obj, h5py.Group):
                    group_data = []
                    for param in param_names:
                        if param in obj:
                            group_data.append(obj[param][()])
                            # all_groups_data[param].append(obj[param][()])
                        else:
                            raise ValueError(f"There is no group/data name {param} in the file {u.get_path_hfd5(b, config_tuple)}.")
                    
                    all_groups_data.append(group_data)

            f.visititems(process_group)

    return all_groups_data


def filter_dataset(dps:list):
    min_value = 4.5
    filtered_dps = []
    seen = set()

    for idx, x in enumerate(dps):
        C_DD, C_DG = x[1], x[2]
        K = C_DD.shape[0]
        # Check if all diagonal elements of C_DD are greater than or equal to min_value
        if all(C_DD[i][i] >= min_value for i in range(K)):
            # Create a hashable representation of the datapoint
            hashable_rep = (tuple(C_DD.flatten()), tuple(C_DG.flatten()))
            
            # If this representation hasn't been seen before, add it to the filtered dataset
            if hashable_rep not in seen:
                seen.add(hashable_rep)
                filtered_dps.append(x)
    
    print(f"Original dataset size: {len(dps)}")
    print(f"Filtered dataset size: {len(filtered_dps)}")
    print(f"Removed {len(dps) - len(filtered_dps)} datapoints")
    
    return filtered_dps

def preprocess_csd(csd_array:np.ndarray):
    # Check if the input
    if not isinstance(csd_array, np.ndarray):
        raise TypeError(f"Input must be a numpy.ndarray, {type(csd_array)}.")
    elif csd_array.shape != (4,c.RESOLUTION,c.RESOLUTION):
        raise TypeError(f"CSD image must be of a shape 4x{c.RESOLUTION}x{c.RESOLUTION}")

    csd_array = np.transpose(csd_array, (1, 2, 0))
    if csd_array.shape[2] == 4:
        csd_array = csd_array[:, :, :3]  # Keep only the RGB channels

    # Convert the NumPy array to a PyTorch tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with a single channel
        transforms.ToTensor(),  # Converts grayscale image to tensor with shape (1, H, W) in [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Apply the transformation
    csd_tensor = transform(csd_array)

    return csd_tensor

def preprocess_capacitance_matrices(c_dd:np.ndarray, c_dg:np.ndarray):
    K = c_dd.shape[0]
    if c.MODE == 1:
        c_dd = c_dd[np.triu_indices(n=K)]
        return np.concatenate((c_dd, c_dg.reshape(K**2)), axis=None)
    elif c.MODE  == 2:
        return np.concatenate((np.diag(c_dd), np.diag(c_dg)), axis=None)
    elif c.MODE  == 3:
        return np.diag(c_dd)
    else:
        raise ValueError(f"Mode must be 1 (all params), 2(both diags), 3(diag C_DD), {c.MODE} is not a valid mode.")

def reconstruct_capacitance_matrices(output:np.ndarray, K:int):
    if c.MODE == 1:
        c_dd = np.zeros((K, K))
        c_dd[np.triu_indices(n=K)] = output[:K*(K+1)//2]
        c_dd = c_dd + c_dd.T 
        c_dd[np.diag_indices_from(c_dd)] = c_dd[np.diag_indices_from(c_dd)]/2
        c_dg = output[K*(K+1)//2:].reshape(K, K)
    else:
        raise ValueError(f"For modes different than 1, the function is not implemented (ambiguous solution).")
    # elif c.MODE == 2:
    #     c_dd = np.diag(output[:c.K])
    #     c_dg = np.diag(output[c.K:])
    # elif c.MODE == 3:
    #     c_dd = np.diag(output)
    #     c_dg = np.zeros((c.K, c.K))
    # else:
    #     raise ValueError(f"Mode must be 1,2,3. {c.MODE} is not a valid mode.")

    return c_dd, c_dg   

def preprocess_data(dps:list, filtered:bool=True):
    """
    Args:
        dps - the list of the loaded parameters' in a format of [['csd','C_DD', 'C_DG', any other ... ], [...], ... [...]]
        param_names - the list of the parameters' names to load from .h5 file
    Returns:
        Returns the list of the preprocessed data
    """
    if filtered:
        dps = filter_dataset(dps)
    
    # Get only csd and C_DD, C_DG <-> input and output
    dps = [x[:3] for x in dps]
    X, Y = list(), list()

    for x in dps:
        X.append(preprocess_csd(x[0]))
        Y.append(preprocess_capacitance_matrices(x[1], x[2]))

    return np.array(X), np.array(Y)

def prepare_data(config_tuple, param_names:list=['csd', 'C_DD', 'C_DG'], all_batches=True, batches:list=None, datasize_cut:int=None):
    datapoints = load_datapoints(config_tuple, param_names, all_batches, batches)
    X, Y = preprocess_data(datapoints)

    if datasize_cut is not None and datasize_cut > len(X):
        print(f"Datasize is greater than the number of datapoints available ({len(X)}). Returning all datapoints.")
        return torch.FloatTensor(X), torch.FloatTensor(Y)
    else:
        return torch.FloatTensor(X[:datasize_cut]), torch.FloatTensor(Y[:datasize_cut])

def tensor_to_image(tensor, unnormalize=True):
    """
    Convert a PyTorch tensor to a displayable image.

    Args:
        tensor (torch.Tensor): The input tensor with shape (1, H, W) for grayscale or (C, H, W) for color.
        unnormalize (bool): If True, reverse the normalization (default is True).

    Returns:
        numpy.ndarray: The reconstructed image as a NumPy array with pixel values in [0, 255].
    """

    if unnormalize:
        # Assuming the normalization was mean=0.5 and std=0.5
        tensor = tensor * 0.5 + 0.5  # Reverse normalization from [-1, 1] to [0, 1]

    # Clip the values to be in the range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to a NumPy array
    if tensor.shape[0] == 1:  # Grayscale image
        image_array = tensor.squeeze(0).numpy()  # Remove the channel dimension for grayscale
    else:  # Color image
        image_array = tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

    # Scale to [0, 255] and convert to uint8
    image_array = (image_array * 255).astype(np.uint8)

    return image_array

def show_image_from_tensor(tensor, unnormalize=False):
    """
    Display an image from a PyTorch tensor.

    Args:
        tensor (torch.Tensor): The input tensor with shape (1, H, W) for grayscale or (C, H, W) for color.
        unnormalize (bool): If True, reverse the normalization (default is True).
    """
    image_array = tensor_to_image(tensor, unnormalize)

    plt.figure(figsize=(c.RESOLUTION/c.DPI, c.RESOLUTION/c.DPI), dpi=c.DPI, layout='tight')
    
    plt.imshow(image_array, cmap='gray')
    
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    plt.show()


# ----------------------------- TRAINING

def divide_dataset(X, y, batch_size, val_split, test_split, random_state):
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

def train_model(model, X, y, batch_size=32, epochs=100, learning_rate=0.001, val_split=0.2, 
                test_split=0.1, random_state=42, epsilon=1.0, init_weights=None):
    '''
        Train a model on the given data and hyperparameters.
    '''
    print(f"\nUsing device: {c.DEVICE}")

    # Move model to GPU
    model = model.to(c.DEVICE)

    train_loader, val_loader, test_loader = divide_dataset(X, y, batch_size, val_split, test_split, random_state)

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
        base_model = config['params'].get('base_model', 'CustomCNN')
        
        # Create a unique directory for this run under the base model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_dir = os.path.join(save_dir, base_model, f"{model_name}_{timestamp}")
        u.ensure_dir_exists(model_save_dir)
        
        # Load initial weights if specified
        if train_params.get('init_weights'):
            init_weights_path = train_params['init_weights']
            if os.path.exists(init_weights_path):
                model = load_model_weights(model, init_weights_path)
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
        print(f"Evaluation: MSE: {metrics['MSE']:.6f}, MAE: {metrics['MAE']:.6f}, R2: {metrics['R2']:.4f}\n")

        # Extract targets and outputs from predictions
        targets = np.concatenate([p[1] for p in predictions])
        outputs = np.concatenate([p[2] for p in predictions])
        
        # Create and save the L2 norm polar plot
        plot_l2_norm_polar(targets, outputs, model_save_dir, train_params['epsilon'])
        plot_l2_norm_polar(targets, outputs, model_save_dir, train_params['epsilon'], num_groups=5)
        
        # Save the model
        model_path = os.path.join(model_save_dir, f"{model_name}.pth")
        save_model_weights(trained_model, model_path)
        
        # Save the history and metrics
        result = {
            'config': {
                'model_name': config['model'].__name__,
                'params': config['params'],
                'custom_head': config['params'].get('custom_head', None),
                'dropout': config['params'].get('dropout', None)
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
        
        save_results_to_csv([result])
        
        save_results_and_history(result, history, predictions, model_save_dir)
        save_model(trained_model, model_save_dir, model_name)
        
    return results

# ----------------------------- Plotting
def plot_learning_curves(history, result, save_dir):
    u.ensure_dir_exists(save_dir)
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

def plot_l2_norm_polar(targets, outputs, save_dir, epsilon, num_groups=6, num_points=None):
    """
    Create two plots in polar coordinates with points as distances between
    the origin and the L2 norm of the difference of targets and outputs.
    The first plot will have concentric circles at integer radii and no angle labels.
    Colors and shapes represent distance groups from the origin based on epsilon.
    The second plot will have the intrested 5 groups.

    Args:
        targets (np.array): The true values.
        outputs (np.array): The predicted values from the model.
        save_dir (str): Directory to save the plot.
        epsilon (float): The epsilon value used for grouping.
        num_points (int, optional): Number of points to plot. If None, all points are plotted.

    Returns:
        None
    """
    u.ensure_dir_exists(save_dir)
    
    # Calculate L2 norms
    l2_norms = np.linalg.norm(targets - outputs, axis=1)
    
    # Sample points if specified
    if num_points is not None and num_points < len(l2_norms):
        indices = np.random.choice(len(l2_norms), num_points, replace=False)
        l2_norms = l2_norms[indices]
    else:
        num_points = len(l2_norms)
    
    # Create angles for each point
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Plot 1: Custom ranges or epsilon-based
    fig1, ax1 = plt.subplots(figsize=(16, 14), subplot_kw=dict(projection='polar'))
    
    # Define color groups and shapes based on epsilon
    color_groups = np.minimum(np.floor(l2_norms / epsilon), 5)  # 6 groups (0 to 5)
    shapes = ['o', 's', '^', 'D', 'p', '*']
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    
    # Calculate percentages for each L2 range
    percentages = [np.sum(color_groups == i) / len(color_groups) * 100 for i in range(6)]
    
    # Plot the points
    for i in range(num_groups):
        mask = color_groups == i
        if i < 5:
            label = f'{i*epsilon:.2f} ≤ L2 < {(i+1)*epsilon:.2f} ({percentages[i]:.1f}%)'
        else:
            label = f'L2 ≥ {5*epsilon:.2f} ({percentages[i]:.1f}%)'
        ax1.scatter(theta[mask], l2_norms[mask], c=[colors[i]], marker=shapes[i], label=label, alpha=0.8)
    
    # Set the rmax to be the ceiling of the max L2 norm
    rmax = np.ceil(np.max(l2_norms))
    ax1.set_rmax(rmax)
    
    # Set the rticks to be multiples of epsilon
    if num_groups > 5:
        rticks = np.concatenate([
            np.arange(0, min(5*epsilon, rmax), epsilon),
            np.arange(5*epsilon, rmax + epsilon, max(epsilon, 1))
        ])
        # Set the rmax to be the ceiling of the max L2 norm
        rmax = np.ceil(np.max(l2_norms))
        ax1.set_rmax(rmax)
    else:
        rticks = np.arange(0, (num_groups+1)*epsilon, epsilon)
        ax1.set_rmax(rticks[-1])

    rticks = np.unique(rticks)  # Remove any duplicates
    ax1.set_rticks(rticks)
    
    # Remove the angle labels
    ax1.set_xticklabels([])
    
    # Add labels and title
    ax1.set_title(f"L2 Norm of Target-Output Difference\n(ε={epsilon:.2f}, {num_points} points)", fontsize=22)
    
    # Add legend for shapes and colors
    legend = ax1.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="L2 Norm Groups", fontsize=18)
    plt.setp(legend.get_title(), fontsize=16)
    
    # Adjust layout manually
    plt.subplots_adjust(right=0.75, bottom=0.05, top=0.95)
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'l2_norm_polar_plot_{num_groups}_groups.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"L2 norm polar plot saved in {save_dir}")


# ----------------------------- SAVES
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
    u.ensure_dir_exists(save_dir)
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
    u.ensure_dir_exists(save_dir)
    # Save in PyTorch's native format
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}.pth'))

def save_model_weights(model, path):
    """
    Save the model weights to a file.
    
    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path to save the model weights.
    """
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

def load_model_weights(model, path):
    """
    Load the model weights from a file.
    
    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): The path to load the model weights from.
    
    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        print(f"Model weights loaded from {path}")
    else:
        print(f"No weights file found at {path}")
    return model

def save_results_to_csv(results, filename='Results/model_results.csv'):
    """
    Save or update the results from train_evaluate_and_save_models function to a CSV file.
    
    Args:
        results (list): List of dictionaries containing model results.
        filename (str): Name of the CSV file to save/update.
    """
    u.ensure_dir_exists(os.path.dirname(filename))
    results_data = []
    for result in results:
        input_shape = result['input_shape']
        output_shape = result['output_shape']
        dataset_size = result['dataset_size']
        val_split = result['train_params']['val_split']
        test_split = result['train_params']['test_split']
        seed = result['train_params']['random_state']
        model_name = result['config']['params']['name']
        base_model = result['config']['params'].get('base_model', 'N/A')
        init_weights = True if result['train_params']['init_weights'] is not None else False  
        epsilon = result['train_params']['epsilon']

        train_params = result['train_params']
        results_data.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            'base_model': base_model,
            'input_shape': list(input_shape),
            'output_shape': list(output_shape),
            'dataset_size': dataset_size,
            'val_split': val_split,
            'test_split': test_split,
            'seed': seed,
            'init_weights': init_weights,
            'custom_head':  result['config']['params'].get('custom_head', '[512, 256]'),
            'dropout': result['config']['params'].get('dropout', 'N/A'),
            'batch_size':  train_params.get('batch_size', 'N/A'),
            'epochs':  train_params.get('epochs', 'N/A'),
            'learning_rate':  train_params.get('learning_rate', 'N/A'),
            'epsilon': epsilon,
            'test_accuracy_global': result['global_test_accuracy'],
            'test_accuracy_local':result['local_test_accuracy'],
            'MSE': result['metrics']['MSE'],
            'MAE': result['metrics']['MAE'],
            'R2': result['metrics']['R2']

        })
    
    df = pd.DataFrame(results_data)
    
    if os.path.exists(filename):
        # If file exists, append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If file doesn't exist, write the header
        df.to_csv(filename, index=False)
    
    print(f"Results saved/updated in {filename}")


# ----------------------------- EVALUATION
def explain_output(model_path, input_tensor):
    """
    Load a model and create explanatory visualizations (saliency map and Grad-CAM) for a given input.
    For regression, we'll visualize the gradients with respect to the mean of all outputs.
    
    Args:
        model_path (str): Path to the saved model weights
        input_tensor (torch.Tensor): Input image tensor of shape (1, 1, H, W)
    
    Returns:
        tuple: (saliency_overlay, gradcam_overlay, prediction, reconstructed_matrices)
            - saliency_overlay (PIL.Image): Saliency map overlaid on input image
            - gradcam_overlay (PIL.Image): Grad-CAM visualization overlaid on input image
            - prediction (numpy.ndarray): Model's output prediction
            - reconstructed_matrices (tuple): (C_DD, C_DG) reconstructed matrices
    """
    K = input_tensor.shape[0]
    # Ensure input is on the correct device
    input_tensor = input_tensor.to(c.DEVICE)
    
    # Load model architecture and weights
    if 'resnet' in model_path.lower():
        model = ResNet(name="ResNet_cnn")
    else:
        model = VanillaCNN(name="vanilla_cnn")
    
    # Load state dict with better error handling
    try:
        state_dict = torch.load(model_path, map_location=c.DEVICE)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # If loading fails, try loading with weights_only=True
        try:
            state_dict = torch.load(model_path, map_location=c.DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e2:
            print(f"Error loading model weights: {e2}")
            raise
    
    model = model.to(c.DEVICE)
    model.eval()
    
    # Get model prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy()[0]
    
    # Generate saliency map
    saliency = saliency_map(model, input_tensor.clone())
    
    # Generate Grad-CAM visualization
    gradcam = grad_cam(model, input_tensor.clone())
    
    # Convert input tensor to range [0, 1] for visualization
    input_image = input_tensor.squeeze().cpu()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    input_image = input_image.unsqueeze(0).repeat(3, 1, 1)  # Convert to RGB
    
    # Create overlays
    saliency_overlay = overlay_heatmap_on_image(input_image, saliency, alpha=0.7)
    gradcam_overlay = overlay_heatmap_on_image(input_image, gradcam, alpha=0.7)
    
    # Reconstruct C_DD and C_DG matrices from prediction
    reconstructed_matrices = reconstruct_capacitance_matrices(prediction, K)
    
    return saliency_overlay, gradcam_overlay, prediction, reconstructed_matrices

def saliency_map(model, input_tensor):
    """
    Generate a saliency map for the given input in a regression task.
    We'll use the mean of all outputs as the target for visualization.
    
    Args:
        model (nn.Module): The neural network model
        input_tensor (torch.Tensor): Input tensor of shape (1, 1, H, W)
    
    Returns:
        numpy.ndarray: Saliency map of shape (H, W)
    """
    input_tensor.requires_grad_()
    
    # Forward pass
    output = model(input_tensor)
    # Use mean of all outputs for regression visualization
    target = output.mean()
    
    # Backward pass
    model.zero_grad()
    target.backward()
    
    # Get gradients and convert to saliency map
    saliency = input_tensor.grad.abs().squeeze().cpu().numpy()
    
    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency

def grad_cam(model, input_tensor):
    """
    Generate a Grad-CAM visualization for the given input in a regression task.
    We'll use the mean of all outputs for visualization.
    
    Args:
        model (nn.Module): The neural network model
        input_tensor (torch.Tensor): Input tensor of shape (1, 1, H, W)
    
    Returns:
        numpy.ndarray: Grad-CAM heatmap of shape (H, W)
    """
    # Get the last convolutional layer
    if hasattr(model, 'base_model'):  # ResNet
        target_layer = model.base_model.layer4[-1]
    else:  # VanillaCNN
        target_layer = [m for m in model.network.modules() if isinstance(m, nn.Conv2d)][-1]
    
    # Register hooks to get gradients and activations
    gradients = []
    activations = []
    
    def save_gradient(grad):
        gradients.append(grad)
    
    def save_activation(module, input, output):
        activations.append(output)
    
    # Register hooks
    handle_activation = target_layer.register_forward_hook(save_activation)
    handle_gradient = target_layer.register_backward_hook(
        lambda module, grad_input, grad_output: save_gradient(grad_output[0])
    )
    
    # Forward pass
    output = model(input_tensor)
    target = output.mean()  # Use mean of all outputs for regression visualization
    
    # Backward pass
    model.zero_grad()
    target.backward()
    
    # Remove hooks
    handle_activation.remove()
    handle_gradient.remove()
    
    # Calculate Grad-CAM
    gradients = gradients[0].cpu().data.numpy()[0]  # [C, H, W]
    activations = activations[0].cpu().data.numpy()[0]  # [C, H, W]
    
    weights = np.mean(gradients, axis=(1, 2))  # [C]
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)  # [H, W]
    
    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Resize to input size
    cam = Image.fromarray((cam * 255).astype(np.uint8))
    cam = cam.resize((input_tensor.shape[3], input_tensor.shape[2]), Image.BICUBIC)
    cam = np.array(cam) / 255.0
    
    return cam

def overlay_heatmap_on_image(image: torch.Tensor, heatmap: np.ndarray, alpha: float = 0.5, colormap: str = 'jet') -> Image:
    """
    Overlays a heatmap on an original image.

    Args:
        image (torch.Tensor): Original image tensor of shape (3, H, W) in the range [0, 1].
        heatmap (np.ndarray): Heatmap array of shape (H, W) with values in range [0, 1].
        alpha (float): Transparency factor for the heatmap overlay. 0 is fully transparent, 1 is fully opaque.
        colormap (str): Colormap to use for heatmap (e.g., 'jet').

    Returns:
        Image: PIL Image with the heatmap overlay.
    """
    # Convert the image tensor to a PIL image
    image_pil = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    # Normalize heatmap if it is not already in range [0, 1]
    if heatmap.max() > 1:
        heatmap = heatmap / heatmap.max()

    # Convert heatmap to a color map using PIL
    heatmap_colored = Image.fromarray((plt.cm.get_cmap(colormap)(heatmap)[:, :, :3] * 255).astype(np.uint8))
    heatmap_colored = heatmap_colored.resize(image_pil.size, Image.LANCZOS)

    # Blend the original image and the heatmap
    heatmap_overlay = Image.blend(image_pil, heatmap_colored, alpha)

    return heatmap_overlay

# TODO: Rewrite this function to work without need of c.K, but with config tuple
def generate_csd_from_prediction(config_tuple, prediction):
    return NotImplementedError()
#     C_DD, C_DG = reconstruct_capacitance_matrices(prediction, config_tuple[0])
#     capacitance_config = {
#         "C_DD" : C_DD,  #dot-dot capacitance matrix
#         "C_Dg" : C_DG,  #dot-gate capacitance matrix
#         "ks" : None,       
#     }

#     cuts = u.get_cut(config_tuple[0])
#     # x_vol = np.linspace(-c.V_G, c.V_G, c.RESOLUTION)
#     x_vol = np.linspace(0, 0.05, c.RESOLUTION)
#     y_vol = np.linspace(0, 0.05, c.RESOLUTION)

#     xks, yks, csd_dataks, polytopesks, _, _ =  Experiment(capacitance_config).generate_CSD(
#                                                 x_voltages = x_vol,  #V
#                                                 y_voltages = y_vol,  #V
#                                                 plane_axes = cuts,
#                                                 compute_polytopes = True,
#                                                 use_virtual_gates = False)   
    
#     pred_csd = u.plot_CSD(xks, yks, csd_dataks, polytopesks)

#     return pred_csd
