import utilities.config as c
import utilities.utils as u

import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import torch
from torchvision import transforms

from utilities.utils import ensure_dir_exists

def load_datapoints(param_names:list, all_batches=True, batches:list=None):
    """
    Args:
        param_names - the list of the parameters' names to load from .h5 file
        all_batches - if True the function loads all available batches
        batches - if all_batches=False this should pass the list of batch numbers to load
    
    Returns:
        A dictionary where keys are param_names and values are lists of elements from all batches
    """
    if all_batches and batches is None:
        batches_nums = np.arange(1, u.count_directories_in_folder()+1)
    elif all_batches==False and batches is not None:
        if all(isinstance(b, (int, np.integer)) and b > 0 for b in batches):
            max_batch = u.count_directories_in_folder()
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

    for b in batches_nums:
        with h5py.File(u.get_path_hfd5(b), 'r') as f:
            def process_group(name, obj):
                if isinstance(obj, h5py.Group):
                    group_data = []
                    for param in param_names:
                        if param in obj:
                            group_data.append(obj[param][()])
                            # all_groups_data[param].append(obj[param][()])
                        else:
                            raise ValueError(f"There is no group/data name {param} in the file {u.get_path_hfd5(b)}.")
                    
                    all_groups_data.append(group_data)

            f.visititems(process_group)

    return all_groups_data


def filter_dataset(dps:list):
    min_value = 4.5

    for idx, x in enumerate(dps):
        C_DD, C_DG = x[1], x[2]
        if all(C_DD[i][i] < min_value for i in range(c.K)):
            del dps[idx]
    
    return dps

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
    if c.MODE == 1:
        c_dd = c_dd[np.triu_indices(n=c.K)]
        return np.concatenate((c_dd, c_dg.reshape(c.K**2)), axis=None)
    elif c.MODE  == 2:
        return np.concatenate((np.diag(c_dd), np.diag(c_dg)), axis=None)
    elif c.MODE  == 3:
        return np.diag(c_dd)
    else:
        raise ValueError(f"Mode must be 1 (all params), 2(both diags), 3(diag C_DD), {c.MODE} is not a valid mode.")

def reconstruct_capacitance_matrices(output:np.ndarray):
    if c.MODE == 1:
        c_dd = np.zeros((c.K, c.K))
        c_dd[np.triu_indices(n=c.K)] = output[:c.K*(c.K+1)//2]
        c_dd = c_dd + c_dd.T 
        c_dd[np.diag_indices_from(c_dd)] = c_dd[np.diag_indices_from(c_dd)]/2
        c_dg = output[c.K*(c.K+1)//2:].reshape(c.K, c.K)
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

def prepare_data(param_names:list=['csd', 'C_DD', 'C_DG'], all_batches=True, batches:list=None):
    datapoints = load_datapoints(param_names, all_batches, batches)
    X, Y = preprocess_data(datapoints)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

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


# ----------------------------- Change

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
    ensure_dir_exists(os.path.dirname(filename))
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
        batch_size = result['config']['params'].get('batch_size', 'N/A')
        num_epochs = result['config']['params'].get('epochs', 'N/A')
        learning_rate = result['config']['params'].get('learning_rate', 'N/A')
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
            'batch_size':  train_params.get('learning_rate', 'N/A'),
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
