import sys
import h5py, os, csv, json
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
        if not all(isinstance(b, int) for b in batches):
            batches = [int(b) for b in batches]

        if all(b>0 for b in batches):
            max_batch = u.count_directories_in_folder(config_tuple)
            if all(b <= max_batch for b in batches):
                batches_nums = batches
            else:
                raise ValueError(f"Some batch numbers are greater than the total number of directories ({max_batch}).")
        else:
            raise ValueError("Batches must be a list of positive integers.")
    else:
        raise ValueError("Batches not defined properly, both all_batches and batches activated.")
      
    data_dict = {param:[] for param in param_names}
    # [list(values) for values in zip(*dictionary.values())]
    print(f"Loading data from {u.get_path_hfd5(batches_nums[0], config_tuple)}")
    for b in batches_nums:
        with h5py.File(u.get_path_hfd5(b, config_tuple), 'r') as f:
            groups = list(f.keys())
            for gn in groups:
                group = f[gn]
                for param in param_names:
                    if isinstance(group[param], h5py.Group):
                        # For parameters that are groups (like 'cuts', 'csd', etc.)
                        data_list = []
                        for item in group[param].keys():
                            data = np.array(group[param][item][()]).squeeze()
                            data_list.append(data)
                        data_dict[param].append(data_list)
                    else:
                        # For direct datasets
                        data_dict[param].append(np.array(group[param][()]).squeeze())

    for param in param_names:
        data_dict[param] = np.array(data_dict[param]).squeeze()

    return data_dict

def filter_dataset(dps:list):
    min_value = 4.0
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

def convert_csd_gradient_to_csd_img(csd_gradient: np.ndarray):
    fig, ax = plt.subplots(figsize=(c.RESOLUTION/c.DPI, c.RESOLUTION/c.DPI), dpi=c.DPI)
    ax.pcolormesh(csd_gradient.squeeze())
    plt.axis('off')
    plt.tight_layout(pad=0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert the canvas to a NumPy array
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(canvas.get_width_height()[::-1] + (4,))

    plt.close(fig)   

    return image

def preprocess_csd(csd_array: np.ndarray, input_type: str = 'csd'):
    """
    Preprocess CSD or gradient array for model input.
    
    Args:
        csd_array (np.ndarray): Input array of shape (4,RESOLUTION,RESOLUTION) or (1,RESOLUTION,RESOLUTION)
        input_type (str): Type of input - 'csd' or 'gradient'
    
    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, RESOLUTION, RESOLUTION)
    """
    # Check input
    if not isinstance(csd_array, np.ndarray):
        raise TypeError(f"Input must be a numpy.ndarray, got {type(csd_array)}")
    
    # if input_type == 'csd_gradient':
    #     csd_array = convert_csd_gradient_to_csd_img(csd_array)
    
    # Handle different input shapes    
    if len(csd_array.shape) == 3:  # (C, H, W)
        if csd_array.shape[0] == 4:  # RGBA
            csd_array = np.transpose(csd_array, (1, 2, 0))
            csd_array = csd_array[:, :, :3]  # Keep only RGB
        elif csd_array.shape[0] == 1:  # Already single channel
            return torch.FloatTensor(csd_array)
    elif len(csd_array.shape) == 2:  # (H, W)
        csd_array = csd_array[None, :, :]  # Add channel dimension
        return torch.FloatTensor(csd_array)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert NumPy array to PIL Image
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with a single channel
        transforms.ToTensor(),  # Converts grayscale image to tensor with shape (1, H, W) in [0, 1]
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    return transform(csd_array)

def get_maxwell_capacitance_matrices(C_DD:np.ndarray, C_DG:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        Get the Maxwell capacitance matrix.
    """
    K = len(C_DD)   
    maxwell_c_dd = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i==j:
                maxwell_c_dd[i,j] = np.sum(C_DD[i,:]+ C_DG[i,:]) 
            else:
                maxwell_c_dd[i,j] = -C_DD[i,j]
    
    maxwell_c_dd = np.round(maxwell_c_dd, 6)
    maxwell_c_dg = np.round(C_DG, 6)

    return maxwell_c_dd, maxwell_c_dg

def preprocess_capacitance_matrices(c_dd:np.ndarray, c_dg:np.ndarray, maxwell:bool=True):
    
    if maxwell:
        c_dd, c_dg = get_maxwell_capacitance_matrices(c_dd, c_dg)
        c_dd = np.linalg.inv(c_dd)

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

def reconstruct_capacitance_matrices(config_tuple, output:np.ndarray):
    K, _, _ = config_tuple

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

def preprocess_data(dps:dict, filtered:bool=True, input_type:str='csd', maxwell:bool=True):
    """
    Args:
        dps - the dictionary of the loaded parameters' in a format of [['csd','C_DD', 'C_DG', any other ... ], [...], ... [...]]
        param_names - the list of the parameters' names to load from .h5 file
    Returns:
        Returns the list of the preprocessed data
    """
    # if filtered:
    #     dps = filter_dataset(dps)
    
    # Get only csd and C_DD, C_DG <-> input and output
    params_name = dps.keys()
    
    X, Y = list(), list()

    print(f"Preprocessing data...")
    for x in dps[input_type]:
        X.append(preprocess_csd(x, input_type))

    for c_dd, c_dg in zip(dps['C_DD'], dps['C_DG']):
        Y.append(preprocess_capacitance_matrices(c_dd, c_dg, maxwell))

    print(f"Data preprocessed.")
    
    return np.array(X), np.array(Y)

def prepare_data(config_tuple:tuple, param_names:list=['csd', 'C_DD', 'C_DG'], all_batches:bool=True, batches:list=None, datasize_cut:int=None, maxwell:bool=True):
    datapoints = load_datapoints(config_tuple, param_names, all_batches, batches)
    X, Y = preprocess_data(datapoints, input_type=param_names[0], maxwell=maxwell)

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

def calculate_local_global_accuracy(y_true: np.array, y_pred: np.array, epsilon: float):
    """
    Calculate local and global accuracy metrics with a given error margin (epsilon).
    
    Args:
        y_true (numpy.ndarray): True values, shape (N, d).
        y_pred (numpy.ndarray): Predicted values, shape (N, d).
        epsilon (float): Error margin.
    
    Returns:
        local_accuracy (np.ndarray): Percentage of correctly predicted dimensions.
        global_accuracy (float): Percentage of fully correct predicted vectors.
    """
    # Calculate absolute differences
    abs_diff = np.abs(y_true - y_pred)
    
    # Local accuracy: proportion of dimensions within epsilon
    local_accuracy = np.round(np.mean((abs_diff < epsilon).astype(float), axis=0),4)

    # Global accuracy: proportion of samples where all dimensions are within epsilon
    global_accuracy =  np.round(np.mean((np.max(abs_diff, axis=1) <= epsilon).astype(float)), 4)

    return  global_accuracy * 100, local_accuracy * 100

def concordance_correlation_coef(targets: np.array, outputs: np.array):
    """
    Calculate the concordance correlation coefficient (CCC) between true and predicted values.

    Args:
        targets (np.array): True values, shape (N, d).
        outputs (np.array): Predicted values, shape (N, d).

    Returns:
        ccc (np.array): Concordance correlation coefficient for each dimension.
    """
    mean_true = np.mean(targets, axis=0)
    mean_pred = np.mean(outputs, axis=0)
    var_true = np.var(targets, axis=0)
    var_pred = np.var(outputs, axis=0)
    covariance = np.mean((targets - mean_true) * (outputs - mean_pred), axis=0)

    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator

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

#TODO: check if this is correct
def physics_informed_regularization_torch(config_tuple, outputs, targets, reduction='mean'):
    """
    PyTorch version of physics-informed regularization that works with batches and enables autograd.
    
    Args:
        config_tuple: Configuration tuple containing (K, R, S) values
        outputs: Predicted values tensor from the model (batch_size, output_dim)
        targets: Target values tensor (batch_size, output_dim)
        
    Returns:
        Regularization loss tensor
    """
    K, _, S = config_tuple
    # if S == 0:
    #     return torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

    batch_size = outputs.shape[0]
    
    # Reconstruct C_DD and C_DG matrices for both outputs and targets
    c_dd_hat = torch.zeros((batch_size, K, K), device=outputs.device, dtype=outputs.dtype)
    c_dd_hat[:, torch.triu_indices(K, K)[0], torch.triu_indices(K, K)[1]] = outputs[:, :K*(K+1)//2]
    c_dd_hat = c_dd_hat + torch.triu(c_dd_hat, 1).transpose(-2, -1)
    
    c_dg_hat = outputs[:, K*(K+1)//2:].reshape(batch_size, K, K)
    
    # Same for targets
    c_dd = torch.zeros((batch_size, K, K), device=targets.device, dtype=targets.dtype)
    c_dd[:, torch.triu_indices(K, K)[0], torch.triu_indices(K, K)[1]] = targets[:, :K*(K+1)//2]
    c_dd = c_dd + torch.triu(c_dd, 1).transpose(-2, -1)
    
    c_dg = targets[:, K*(K+1)//2:].reshape(batch_size, K, K)
    
    # Calculate true self capacitances
    true_self_capacitances = (torch.diagonal(c_dd, dim1=1, dim2=2) - 
                             torch.sum(c_dg, dim=2) - 
                             (torch.sum(c_dd, dim=2) - torch.diagonal(c_dd, dim1=1, dim2=2)))
    
    # Calculate regularization expression
    reg_expression = (torch.diagonal(c_dd_hat, dim1=1, dim2=2) - 
                     true_self_capacitances - 
                     torch.sum(c_dg_hat, dim=2) - 
                     (torch.sum(c_dd_hat, dim=2) - torch.diagonal(c_dd_hat, dim1=1, dim2=2)))
    # TODO: redefine reg_expression just for diagonal terms in hat matrices the rest from the 
    # More specific regularization expression: just for diagonal terms in hat matrices the rest from the true values
    # reg_expression = (torch.diagonal(c_dd_hat, dim1=1, dim2=2) - 
    #                  true_self_capacitances - 
    #                  torch.diag(c_dg, dim1=1, dim2=2) + torch.sum(c_dg, dim=2) - 
    #                  (torch.sum(c_dd_hat, dim=2) - torch.diagonal(c_dd_hat, dim1=1, dim2=2)))
    
    if reduction == 'mean':
        return torch.mean(torch.norm(reg_expression, dim=1)**2)
    else:
        return torch.norm(reg_expression, dim=1)**2

def calculate_loss(config_tuple, criterion, regularization_coeff, outputs, targets, reduction='mean'):
    loss = criterion(outputs, targets)
    if regularization_coeff > 0:
        physics_informed_loss = physics_informed_regularization_torch(config_tuple, outputs, targets, reduction=reduction)
        loss += regularization_coeff * physics_informed_loss
    return loss

# def physics_informed_regularization(config_tuple, output, target):
#     # output = output.detach().cpu().numpy()
#     # target = target.detach().cpu().numpy()

#     # print(f"output: {type(output)}, {output.shape}, target: {type(target)}, {target.shape}")

#     _, _, S = config_tuple
#     if S==0:
#         print("S=0 is not a valid mode for physics-informed regularization.")
#         return 0

#     output_matrices = reconstruct_capacitance_matrices(config_tuple, output)
#     target_matrices = reconstruct_capacitance_matrices(config_tuple, target)

#     c_dd_hat, c_dg_hat = output_matrices
#     c_dd, c_dg = target_matrices

#     true_self_capacitances = np.diag(c_dd) - np.sum(c_dg, axis=1) - (np.sum(c_dd, axis=1) - np.diag(c_dd)) 
#     sum_preds_c_dg = np.sum(c_dg_hat, axis=1) # sum over rows in dot-gate for given dot
#     sum_preds_c_dd = (np.sum(c_dd_hat, axis=1) - np.diag(c_dd_hat))  # aka c_m

#     reg_expression = np.diag(c_dd_hat) - true_self_capacitances - sum_preds_c_dg - sum_preds_c_dd

#     return np.linalg.norm(reg_expression)**2

def train_model(config_tuple, model, X, y, batch_size=32, epochs=100, learning_rate=0.001, val_split=0.2, 
                test_split=0.1, random_state=42, epsilon=1.0, init_weights=None, 
                criterion=nn.MSELoss(), regularization_coeff=0.0, load_conv_only=None):
    '''
        Train a model on the given data and hyperparameters.
    '''
    print(f"\nUsing device: {c.DEVICE}")

    # Move model to GPU
    model = model.to(c.DEVICE)

    train_loader, val_loader, test_loader = divide_dataset(X, y, batch_size, val_split, test_split, random_state)

    # Define loss function and optimizer
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
        output_dim = None
        all_train_outputs = []
        all_train_targets = []

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = calculate_loss(config_tuple, criterion, regularization_coeff, outputs, targets)
            loss.backward()
            optimizer.step()
            output_dim = outputs.shape[1]
            
            train_loss += loss.item()

            predicted_values = outputs.detach().cpu().numpy()
            true_values = targets.detach().cpu().numpy()

            all_train_outputs.append(predicted_values)
            all_train_targets.append(true_values)

        avg_train_loss = train_loss / len(train_loader)
        
        all_train_targets = np.array(all_train_targets).reshape(-1, output_dim)
        all_train_outputs = np.array(all_train_outputs).reshape(-1, output_dim)
    
        global_train_acc, vec_local_train_acc = calculate_local_global_accuracy(all_train_targets, all_train_outputs, epsilon)
        local_train_acc = np.round(np.mean(vec_local_train_acc), 4)

        train_mse = mean_squared_error(all_train_targets, all_train_outputs)

        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append([global_train_acc, local_train_acc])
        history['vec_local_train_accuracy'].append(vec_local_train_acc)
        history['train_mse'].append(train_mse)

        # Validation step:
        val_loss, global_val_acc, local_val_acc, val_mse, _, vec_local_val_acc = evaluate_model(config_tuple, model, val_loader, criterion, epsilon, regularization_coeff)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append([global_val_acc, local_val_acc])
        history['vec_local_val_accuracy'].append(vec_local_val_acc)
        history['val_mse'].append(val_mse)

        print(f"Epoch {epoch+1}/{epochs}: Tr. Loss: {avg_train_loss:.5f}, Val. Loss: {val_loss:.5f}")
        print(f"{'':<11} Tr. Acc.: {global_train_acc}% ({local_train_acc}%), "
              f"Val. Acc.: {global_val_acc}% ({local_val_acc}%)")
        print(f"{'':<11} Vec. Tr. Local Acc.: {vec_local_train_acc}%")
        print(f"{'':<11} Vec. Val. Local Acc.: {vec_local_val_acc}%")
        print(f"{'':<11} Tr. MSE: {train_mse:.5f}, Val. MSE: {val_mse:.5f}")

    return model, history, test_loader

def evaluate_model(config_tuple, model, dataloader, criterion=nn.MSELoss(), epsilon=1.0,  regularization_coeff=0.0):
    model.eval()

    total_loss = 0.0
    output_dim = None
    all_outputs = []
    all_targets = []

    predictions = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = calculate_loss(config_tuple, criterion, regularization_coeff, outputs, targets)
            total_loss += loss.item()
            
            output_dim = outputs.shape[1]
            predicted_values = outputs.cpu().numpy()
            true_values = targets.cpu().numpy()

            predictions.append([inputs.cpu().numpy(), true_values, predicted_values])

            all_outputs.append(predicted_values)
            all_targets.append(true_values)

    avg_loss = total_loss / len(dataloader)

    all_targets = np.array(all_targets).reshape(-1, output_dim)
    all_outputs = np.array(all_outputs).reshape(-1, output_dim)
    
    global_acc, vec_local_acc = calculate_local_global_accuracy(all_targets, all_outputs, epsilon)
    local_acc = np.round(np.mean(vec_local_acc), 4)
        
    mse = mean_squared_error(all_targets, all_outputs)

    return avg_loss, global_acc, local_acc, mse, predictions, vec_local_acc

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
    mape = mean_absolute_percentage_error(all_targets, all_outputs)
    ccc = concordance_correlation_coef(all_targets, all_outputs)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        # 'MAPE': mape,
        # 'CCC': ccc
    }
    return metrics

def train_evaluate_and_save_models(config_tuple, model_configs, X, y, param_names, train_params, save_dir=None):
    """Train, evaluate, and save multiple models based on the given configurations."""
    if save_dir is None:
        save_dir = os.path.join(c.PATH_0, c.PATH_TO_RESULTS)
    
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
            init_weights_path = os.path.normpath(train_params['init_weights']).replace('\\', '/')
            if os.path.exists(init_weights_path):
                if train_params.get('load_conv_only', False):
                    model = load_conv_weights(model, init_weights_path)
                    print(f"Loaded convolutional weights from {init_weights_path}")
                else:
                    model = load_model_weights(model, init_weights_path)
                    print(f"Loaded all weights from {init_weights_path}")
            else:
                print(f"Warning: Initial weights file not found at {init_weights_path}")
                train_params['init_weights'] = None
                return 0
        
        # Model's parameters
        print("\n\n--------- START TRAINING ---------")
        print(f"Model: {model_name}")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model's parameters: {num_params}, Number of trainable parameters: {num_trainable_params}")

        # Train the model
        trained_model, history, test_loader = train_model(config_tuple, model, X, y, **train_params)
        
        # Evaluate the model
        test_loss, global_test_accuracy, local_test_accuracy, test_mse, predictions, vec_local_test_accuracy = evaluate_model(config_tuple, trained_model, test_loader, epsilon=train_params.get('epsilon', 1.0))
        
        # Collect performance metrics on the test set
        metrics = collect_performance_metrics(trained_model, test_loader)
        
        print(f"Evaluation: Test Accuracy (Global): {global_test_accuracy}%, Test Accuracy (Local): {local_test_accuracy}%")
        print(f"Evaluation: Test Loss: {test_loss:.5f}, Test MSE: {test_mse:.5f}")
        print(f"Evaluation: Vec. Test Local Acc.: {vec_local_test_accuracy}%")
        print(f"Evaluation: MSE: {metrics['MSE']:.6f}, MAE: {metrics['MAE']:.6f}, R2: {metrics['R2']:.6f}\n")

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
            'param_names': param_names,
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
        
        save_results_to_csv([result], filename=os.path.join(save_dir, 'model_results.csv'))
        
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

# TODO: Add save_outliers
def plot_l2_norm_polar(targets, outputs, save_dir, epsilon, num_groups=6, num_points=None, save_outliers=True):
    """
    Create two plots in polar coordinates with points as distances between
    the origin and the L2 norm of the difference of targets and outputs.
    The first plot will have concentric circles at integer radii and no angle labels.
    Colors and shapes represent distance groups from the origin based on epsilon.
    The second plot will have the intrested 5 groups.
    Optionally save outliers to HDF5 file.

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
            np.arange(5*epsilon, rmax + epsilon, max(epsilon, 2))
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
        if isinstance(obj, torch.nn.modules.loss._Loss):
            return obj.__class__.__name__
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
    model.load_state_dict(torch.load(path, weights_only=True))
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
        maxwell_mode = result['param_names'][-1]
        base_model = result['config']['params'].get('base_model', 'N/A')
        init_weights = True if result['train_params']['init_weights'] is not None else False  
        epsilon = result['train_params']['epsilon']

        train_params = result['train_params']
        results_data.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            'base_model': base_model,
            'maxwell_mode': maxwell_mode,
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
            'regularization_coeff': train_params.get('regularization_coeff', '0'),
            'criterion': train_params.get('criterion', nn.MSELoss()),
            'epsilon': epsilon,
            'test_accuracy_global': result['global_test_accuracy'],
            'test_accuracy_local':result['local_test_accuracy'],
            'MSE': result['metrics']['MSE'],
            'MAE': result['metrics']['MAE'],
            'R2': result['metrics']['R2'],
            'MAPE': result['metrics']['MAPE'],
            'CCC': result['metrics']['CCC'] # Concordance Correlation Coefficient

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
def explain_output(config_tuple, model_path, model_name, input_tensor):
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
    K, N, S = config_tuple
    # Ensure input is on the correct device
    input_tensor = input_tensor.to(c.DEVICE)
    
    # Load model architecture and weights
    # if 'resnet' in model_path.lower():
    #     model = ResNet(name="ResNet_cnn")
    # else:
    #     model = VanillaCNN(name="vanilla_cnn")
    
    model = ResNet(base_model=model_name, config_tuple=config_tuple,  custom_head=[2048, 1024], dropout=0.2)
    
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
    
    print(f"Type of input tensor: {type(input_tensor)}, shape: {input_tensor.shape}")

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
    reconstructed_matrices = reconstruct_capacitance_matrices(config_tuple, prediction)
    
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
    # return NotImplementedError()
    C_DD, C_DG = reconstruct_capacitance_matrices(output=prediction, config_tuple=config_tuple)
    capacitance_config = {
        "C_DD" : C_DD,  #dot-dot capacitance matrix
        "C_Dg" : C_DG,  #dot-gate capacitance matrix
        "ks" : None,       
    }

    cuts = u.get_cut(config_tuple)
    # x_vol = np.linspace(-c.V_G, c.V_G, c.RESOLUTION)
    x_vol = np.linspace(0, 0.05, c.RESOLUTION)
    y_vol = np.linspace(0, 0.05, c.RESOLUTION)

    xks, yks, csd_dataks, polytopesks, _, _ =  Experiment(capacitance_config).generate_CSD(
                                                x_voltages = x_vol,  #V
                                                y_voltages = y_vol,  #V
                                                plane_axes = cuts,
                                                compute_polytopes = True,
                                                use_virtual_gates = False)   
    
    pred_csd = u.plot_CSD(xks, yks, csd_dataks, polytopesks)

    return pred_csd

def load_conv_weights(model, path):
    """
    Load only the convolutional layers' weights from a saved ResNet model.
    
    Args:
        model (torch.nn.Module): The model to load weights into
        path (str): Path to the saved model weights
    
    Returns:
        torch.nn.Module: Model with loaded convolutional weights
    """
    try:
        # Load state dict
        state_dict = torch.load(path, map_location=c.DEVICE, weights_only=True)
        
        # Create new state dict with only conv layers
        new_state_dict = {}
        for name, param in state_dict.items():
            # Include conv layers, batch norm, and their related parameters
            if any(x in name for x in ['conv', 'bn', 'downsample']):
                if 'base_model' in name:
                    # For ResNet models, keep the base_model prefix
                    new_state_dict[name] = param
                else:
                    # For custom models, might need to add base_model prefix
                    new_name = f'base_model.{name}' if 'base_model' not in name else name
                    new_state_dict[new_name] = param
        
        # Load the filtered state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        print("Successfully loaded convolutional weights")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # Initialize the remaining layers with Xavier initialization
        for name, param in model.named_parameters():
            if name not in new_state_dict:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
    except Exception as e:
        print(f"Error loading convolutional weights: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise
    
    return model