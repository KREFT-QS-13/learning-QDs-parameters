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

import src.utilities.utils as u
sys.path.append('./qdarts')
from qdarts.experiment import Experiment
from qdarts.plotting import plot_polytopes

from src.models.img_encoder import ResNet, CNN
from src.models.multi_branch import MultiBranchArchitecture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- TRAINING
def divide_dataset(imgs, context, outputs, batch_size, val_split, test_split, random_state, device=DEVICE, use_normalization=False):
    """
    Split dataset into train/val/test sets and create DataLoaders for MultiBranchArchitecture.
    
    Args:
        imgs: torch.Tensor of shape (N, num_branches, 1, H, W) - image tensors
        context: torch.Tensor of shape (N, context_vector_size) - context vectors
        outputs: torch.Tensor of shape (N, output_size) - target outputs
        batch_size: Batch size for DataLoaders
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random state for reproducibility
        device: Device to move tensors to (defaults to DEVICE)
        use_normalization: Whether to use Z-score normalization
    
    Returns:
        train_loader, val_loader, test_loader: DataLoaders returning (imgs, context, outputs) tuples
        norm_dict: Dictionary containing the mean and std of the outputs and context for each split
    """
    # Ensure inputs are torch tensors (should already be, but check for safety)
    if not isinstance(imgs, torch.Tensor):
        raise TypeError(f"imgs must be a torch.Tensor, got {type(imgs)}")
    if not isinstance(context, torch.Tensor):
        raise TypeError(f"context must be a torch.Tensor, got {type(context)}")
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"outputs must be a torch.Tensor, got {type(outputs)}")
    
    # Get number of samples
    num_samples = imgs.shape[0]
    indices = np.arange(num_samples)
    
    # Split indices into train, val, test
    train_indices, test_indices = train_test_split(indices, test_size=test_split, random_state=random_state)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_split, random_state=random_state)

    print(f"\nTrain dataset length: {len(train_indices)} out of {num_samples}. ({len(train_indices)/num_samples*100:.2f}%).")
    print(f"Val dataset length: {len(val_indices)} out of {num_samples}. ({len(val_indices)/num_samples*100:.2f}%).")
    print(f"Test dataset length: {len(test_indices)} out of {num_samples}. ({len(test_indices)/num_samples*100:.2f}%)\n.")

    # Split the data
    imgs_train = imgs[train_indices].to(device)
    imgs_val = imgs[val_indices].to(device)
    imgs_test = imgs[test_indices].to(device)
    
    norm_dict = None
    if use_normalization:
        outputs_train, outputs_mean, outputs_std = u.Z_score_transformation(outputs[train_indices].numpy())
        outputs_val, outputs_mean, outputs_std = u.Z_score_transformation(outputs[val_indices].numpy())
        outputs_test, outputs_mean, outputs_std = u.Z_score_transformation(outputs[test_indices].numpy())
        
        context_train, context_mean, context_std = u.Z_score_transformation(context[train_indices].numpy())
        context_val, context_mean, context_std = u.Z_score_transformation(context[val_indices].numpy())
        context_test, context_mean, context_std = u.Z_score_transformation(context[test_indices].numpy())

        norm_dict = {
            'train': [(outputs_mean, outputs_std), (context_mean, context_std)],
            'val': [(outputs_mean, outputs_std), (context_mean, context_std)],
            'test': [(outputs_mean, outputs_std), (context_mean, context_std)]
        }

        outputs_train = torch.tensor(outputs_train, dtype=torch.float32).to(device)
        outputs_val = torch.tensor(outputs_val, dtype=torch.float32).to(device)
        outputs_test = torch.tensor(outputs_test, dtype=torch.float32).to(device)

        context_train = torch.tensor(context_train, dtype=torch.float32).to(device)
        context_val = torch.tensor(context_val, dtype=torch.float32).to(device)
        context_test = torch.tensor(context_test, dtype=torch.float32).to(device)
    else:
        outputs_train = outputs[train_indices].to(device)
        outputs_val = outputs[val_indices].to(device)
        outputs_test = outputs[test_indices].to(device)

        context_train = context[train_indices].to(device)
        context_val = context[val_indices].to(device)
        context_test = context[test_indices].to(device)

    
    # Create DataLoader objects
    # Each DataLoader returns (imgs, context, outputs) tuples
    train_dataset = TensorDataset(imgs_train, context_train, outputs_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(imgs_val, context_val, outputs_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(imgs_test, context_test, outputs_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm_dict

def calculate_loss(criterion, outputs, targets, reg_coeff_diag=0.1, reg_coeff_off=1.0):
    """
    Calculate loss with different coefficients for diagonal and off-diagonal elements.
    Assumes outputs represent a flattened square matrix (e.g., 3x3 = 9 elements).
    If output_dim is not a perfect square, treats all elements with reg_coeff_off.
    
    Args:
        criterion: Loss function (e.g., nn.MSELoss())
        outputs: Predicted values, shape (batch_size, output_dim)
        targets: True values, shape (batch_size, output_dim)
        reg_coeff_diag: Coefficient for diagonal elements (default: 0.1)
        reg_coeff_off: Coefficient for off-diagonal elements (default: 1.0)
    
    Returns:
        Combined loss with different weights for diagonal and off-diagonal elements
    """
    outputs_dim = outputs.shape[1]
    
    # Check if output_dim is a perfect square (represents a square matrix)
    matrix_size = int(np.sqrt(outputs_dim))
    is_square_matrix = (matrix_size * matrix_size == outputs_dim)
    
    if is_square_matrix and matrix_size > 1:
        # Reshape to (batch_size, matrix_size, matrix_size)
        outputs_reshaped = outputs.view(-1, matrix_size, matrix_size)
        targets_reshaped = targets.view(-1, matrix_size, matrix_size)
        
        # Extract diagonal elements: (batch_size, matrix_size)
        outputs_diag = torch.diagonal(outputs_reshaped, dim1=1, dim2=2)
        targets_diag = torch.diagonal(targets_reshaped, dim1=1, dim2=2)
        
        # Extract off-diagonal elements
        # Create mask for off-diagonal elements
        mask = ~torch.eye(matrix_size, dtype=torch.bool, device=outputs.device)
        outputs_off = outputs_reshaped[:, mask]
        targets_off = targets_reshaped[:, mask]
        
        # Calculate losses
        loss_diag = reg_coeff_diag * criterion(outputs_diag, targets_diag)
        loss_off = reg_coeff_off * criterion(outputs_off, targets_off)
        loss = loss_diag + loss_off
    else:
        # If not a square matrix, apply reg_coeff_off to all elements
        # or treat as if all are off-diagonal
        loss = reg_coeff_off * criterion(outputs, targets)
    
    return loss

def train_evaluate_and_save_models(model:MultiBranchArchitecture, imgs:torch.Tensor, context:torch.Tensor, outputs:torch.Tensor, save_dir:str, batch_size:int, epochs:int, learning_rate:float, val_split:float, test_split:float, random_state:int, epsilon:float, reg_coeff_diag:float, reg_coeff_off:float, use_normalization:bool):
    """Train, evaluate, and save multiple models based on the given configurations."""
    print("\n\n--------- START TRAINING ---------")
    trained_model, history, test_loader, norm_dict = train_model(model=model, 
                                                                 imgs=imgs, 
                                                                 context=context, 
                                                                 outputs=outputs, 
                                                                 batch_size=batch_size, 
                                                                 epochs=epochs, 
                                                                 learning_rate=learning_rate, 
                                                                 val_split=val_split, 
                                                                 test_split=test_split, 
                                                                 random_state=random_state, 
                                                                 epsilon=epsilon, 
                                                                 reg_coeff_diag=reg_coeff_diag, 
                                                                 reg_coeff_off=reg_coeff_off, 
                                                                 use_normalization=use_normalization)
        
    # Final test of the model
    test_loss, global_test_accuracy, local_test_accuracy, test_mse, predictions, vec_local_test_accuracy = evaluate_model(trained_model, test_loader, epsilon=epsilon)
        
    # Collect performance metrics on the test set
    metrics = collect_performance_metrics(trained_model, test_loader)
        
    print(f"Evaluation: Test Accuracy (Global): {global_test_accuracy}%, Test Accuracy (Local): {local_test_accuracy}%")
    print(f"Evaluation: Test Loss: {test_loss:.5f}, Test MSE: {test_mse:.5f}")
    print(f"Evaluation: Vec. Test Local Acc.: {vec_local_test_accuracy}%")
    print(f"Evaluation: MSE: {metrics['MSE']:.6f}, MAE: {metrics['MAE']:.6f}, R2: {metrics['R2']:.6f}\n")

    # Extract targets and outputs from predictions
    targets = np.concatenate([p[1] for p in predictions])
    outputs = np.concatenate([p[2] for p in predictions])
        
    # # Create and save the L2 norm polar plot (TODO: Add back, add just to new architecture)
    # plot_l2_norm_polar(targets, outputs, save_dir, epsilon)
    # plot_l2_norm_polar(targets, outputs, save_dir, train_params['epsilon'], num_groups=5)
        
    # Save the model
    save_dir = os.path.join(save_dir, f'{model.name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    u.ensure_dir_exists(save_dir)
    model_path = os.path.join(save_dir, f"{model.name}.pth")
    save_model_weights(trained_model, model_path)
        
    # Extract model configuration information
    # Try to get branch model type from the first branch
    branch_model = 'N/A'
    base_model = 'N/A'
    dropout = 'N/A'
    custom_head = 'N/A'
    
    try:
        first_branch = model.branches[0]
        # Check if it's a ResNet branch
        if hasattr(first_branch, 'base_model'):
            # Try to infer model name from the base_model object type
            base_model_obj = first_branch.base_model
            model_type = type(base_model_obj).__name__.lower()
            if 'resnet' in model_type:
                # Extract number from model type (e.g., 'resnet' -> try to get from structure)
                branch_model = model_type
                base_model = model_type
            else:
                branch_model = str(model_type)
                base_model = str(model_type)
        # Check if it's a CNN branch
        elif hasattr(first_branch, 'conv_layers'):
            branch_model = 'cnn'
            base_model = 'cnn'
        
        # Try to get dropout from prediction heads or branch
        if hasattr(model, 'diagonal_head'):
            for layer in model.diagonal_head:
                if isinstance(layer, nn.Dropout):
                    dropout = float(layer.p)
                    break
        # Also check branch prediction head
        if dropout == 'N/A' and hasattr(first_branch, 'custom_prediction_head'):
            for layer in first_branch.custom_prediction_head:
                if isinstance(layer, nn.Dropout):
                    dropout = float(layer.p)
                    break
        
        # Try to get custom head structure from final prediction head (use diagonal head as reference)
        if hasattr(model, 'diagonal_head'):
            head_layers = []
            for layer in model.diagonal_head:
                if isinstance(layer, nn.Linear):
                    head_layers.append(int(layer.out_features))
            if head_layers:
                custom_head = str(head_layers[:-1])  # Exclude final output layer
    except Exception as e:
        # Use defaults if extraction fails
        pass
    
    # Extract data shapes
    input_shape = list(imgs.shape[1:]) if imgs is not None else [0]  # Exclude batch dimension
    output_shape = list(outputs.shape[1:]) if outputs is not None else [0]  # Exclude batch dimension
    dataset_size = len(imgs) if imgs is not None else 0
    
    # Create param_names placeholder (used for mode calculation)
    # This would typically be parameter names from the model, using output shape as proxy
    # Need at least 2 elements for mode calculation: MW-param_names[-2]-param_names[-1]
    if output_shape[0] > 0 and output_shape[0] >= 2:
        param_names = list(range(output_shape[0]))
    else:
        param_names = [0, 1]  # Default to at least 2 elements
    
    # Save the history and metrics
    result = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'dataset_size': dataset_size,
            'param_names': param_names,
            'config': {
                'params': {
                    'name': model.name,
                    'base_model': base_model,
                    'branch_model': branch_model,
                    'custom_head': custom_head,
                    'dropout': dropout
                }
            },
            'train_params': {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'val_split': val_split,
                'test_split': test_split,
                'random_state': random_state,
                'epsilon': epsilon,
                'reg_coeff_diag': reg_coeff_diag,
                'reg_coeff_off': reg_coeff_off,
                'init_weights': None, # Add init_weights field
                'use_normalization': use_normalization,
                'norm_dict': norm_dict
            },
            'history': {k: v for k, v in history.items() if k != 'L2 norm'},
            'test_loss': float(test_loss),
            'global_test_accuracy': float(global_test_accuracy),
            'local_test_accuracy': float(local_test_accuracy),
            'metrics': {k: float(v) for k, v in metrics.items()},
    }
        
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=4, cls=NumpyEncoder)
        
    plot_learning_curves(history, result, save_dir)
                
    save_results_to_csv([result], filename=os.path.join(save_dir, 'model_results.csv'))
        
    save_results_and_history(result, history, predictions, save_dir)
    save_model(trained_model, save_dir, model.name)
        
    return result


def train_model(model, imgs, context, outputs, batch_size=32, epochs=100, learning_rate=0.001, val_split=0.2, 
                test_split=0.1, random_state=42, epsilon=1.0, init_weights=None, 
                criterion=nn.MSELoss(), reg_coeff_diag=0.1, reg_coeff_off=1.0, device=DEVICE, use_normalization=False):
    '''
        Train a model on the given data and hyperparameters.
        
        Args:
            model: The model to train (should be MultiBranchArchitecture)
            imgs: List of lists of image tensors or numpy array of shape (N, num_branches, 1, H, W)
            context: Numpy array of shape (N, context_vector_size) - context vectors
            outputs: Numpy array of shape (N, output_size) - target outputs
            num_branches: Number of branches (required if imgs is a list of lists)
            ... (other parameters remain the same)
    '''
    # move model to GPU
    model = model.to(device)

    train_loader, val_loader, test_loader, dict_norms = divide_dataset(imgs, context, outputs, batch_size, val_split, test_split, random_state, device, use_normalization)

    # Define loss function, optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # test also AdamW
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)    

    # Add early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10

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
    extended_history = history.copy()
    if init_weights:
        history_path = os.path.join(os.path.dirname(init_weights), 'results.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                prev_history = json.load(f)['history']
            for key in history:
                extended_history[key] = prev_history.get(key, [])

            print(f"Last epoch: Tr. Loss: {extended_history['train_losses'][-1]:.5f}, Val. Loss: {extended_history['val_losses'][-1]:.5f}\n", 
                f"{'':<11}Tr. MSE: {extended_history['train_mse'][-1]:.3f}, Val. MSE: {extended_history['val_mse'][-1]:.3f}")
              
    for epoch in range(epochs): 
        model.train()
        
        train_loss = 0.0
        total_train_samples = 0  # Track total number of samples for proper averaging
        all_train_outputs = []
        all_train_targets = []

        for imgs_batch, context_batch, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            # MultiBranchArchitecture expects (imgs, context_vector) where context_vector shape is (batch_size, num_branches, context_vector_size)
            # If context_batch is (batch_size, context_vector_size), we need to expand it
            if context_batch.dim() == 2:
                # Expand to (batch_size, num_branches, context_vector_size) - same context for all branches
                num_branches = imgs_batch.shape[1]
                context_batch = context_batch.unsqueeze(1).expand(-1, num_branches, -1)
            outputs = model(imgs_batch, context_batch)
            loss = calculate_loss(criterion, outputs, targets, reg_coeff_diag, reg_coeff_off)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            # Accumulate loss weighted by batch size for proper averaging over all samples
            batch_size = outputs.shape[0]
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            # Convert to numpy and ensure correct shape
            predicted_values = outputs.detach().cpu().numpy()
            true_values = targets.detach().cpu().numpy()
            
            # Ensure both arrays have the same shape
            if len(predicted_values.shape) == 1:
                predicted_values = predicted_values.reshape(1, -1)
            if len(true_values.shape) == 1:
                true_values = true_values.reshape(1, -1)

            all_train_outputs.append(predicted_values)
            all_train_targets.append(true_values)

        # Compute average loss over all samples (not batches) for consistency
        avg_train_loss = train_loss / total_train_samples if total_train_samples > 0 else 0
        
        # Concatenate all batches
        all_train_outputs = np.concatenate(all_train_outputs, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)
    
        global_train_acc, vec_local_train_acc = calculate_local_global_accuracy(all_train_targets, all_train_outputs, epsilon)
        local_train_acc = np.round(np.mean(vec_local_train_acc), 2)

        train_mse = mean_squared_error(all_train_targets, all_train_outputs)
        
        # Compute training loss on all samples for verification (should match avg_train_loss)
        # Note: This computation is done in train mode, so it should match avg_train_loss
        all_train_outputs_tensor = torch.tensor(all_train_outputs, dtype=torch.float32).to(device)
        all_train_targets_tensor = torch.tensor(all_train_targets, dtype=torch.float32).to(device)
        train_loss_all_samples = criterion(all_train_outputs_tensor, all_train_targets_tensor).item()

        history['train_losses'].append(avg_train_loss)
        history['train_accuracies'].append([global_train_acc, local_train_acc])
        history['vec_local_train_accuracy'].append(vec_local_train_acc)
        history['train_mse'].append(train_mse)

        # Validation step:
        val_loss, global_val_acc, local_val_acc, val_mse, _, vec_local_val_acc = evaluate_model(model, val_loader, criterion, epsilon, reg_coeff_diag, reg_coeff_off)
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append([global_val_acc, local_val_acc])
        history['vec_local_val_accuracy'].append(vec_local_val_acc)
        history['val_mse'].append(val_mse)
        
        # Diagnostic: Compute training loss in eval mode to quantify dropout/BN effect
        model.eval()
        with torch.no_grad():
            train_loss_eval_mode = 0.0
            train_samples_eval = 0
            for imgs_batch, context_batch, targets in train_loader:
                if context_batch.dim() == 2:
                    num_branches = imgs_batch.shape[1]
                    context_batch = context_batch.unsqueeze(1).expand(-1, num_branches, -1)
                outputs = model(imgs_batch, context_batch)
                loss = criterion(outputs, targets)
                batch_size = outputs.shape[0]
                train_loss_eval_mode += loss.item() * batch_size
                train_samples_eval += batch_size
            
            train_loss_eval_mode = train_loss_eval_mode / train_samples_eval if train_samples_eval > 0 else 0
        model.train()  # Set back to train mode

        # Update learning rate scheduler
        lr_scheduler.step(val_loss) 
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}: Tr. Loss: {avg_train_loss:.5f}, Val. Loss: {val_loss:.5f}, LR: {current_lr:.6f}")
        print(f"Reg. Coeff. Diag: {reg_coeff_diag}, Reg. Coeff. Off: {reg_coeff_off}")
        print(f"{'':<11} Tr. Acc.: {global_train_acc}% ({local_train_acc}%), "
              f"Val. Acc.: {global_val_acc}% ({local_val_acc}%)")
        print(f"{'':<11} Vec. Tr. Local Acc.: {vec_local_train_acc}%")
        print(f"{'':<11} Vec. Val. Local Acc.: {vec_local_val_acc}%")
        print(f"{'':<11} Tr. MSE: {train_mse:.5f}, Val. MSE: {val_mse:.5f}")
        # Diagnostic information
        print(f"{'':<11} [Diagnostics] Tr. Loss (all-samples): {train_loss_all_samples:.5f}, "
              f"Diff from batch-avg: {abs(avg_train_loss - train_loss_all_samples):.6f}")
        print(f"{'':<11} [Diagnostics] Tr. Loss (eval mode): {train_loss_eval_mode:.5f}, "
              f"Gap due to dropout/BN: {avg_train_loss - train_loss_eval_mode:.5f}")
        print(f"{'':<11} [Diagnostics] Val Loss vs Val MSE diff: {abs(val_loss - val_mse):.6f}")

    return model, history, test_loader, dict_norms

def evaluate_model(model, dataloader, criterion=nn.MSELoss(), epsilon=1.0, reg_coeff_diag=0.1, reg_coeff_off=1.0):
    model.eval()

    total_loss = 0.0
    total_samples = 0  # Track total number of samples for proper averaging
    output_dim = None
    all_outputs = []
    all_targets = []

    predictions = []
    with torch.no_grad():
        for imgs_batch, context_batch, targets in dataloader:
            # MultiBranchArchitecture expects (imgs, context_vector) where context_vector shape is (batch_size, num_branches, context_vector_size)
            # If context_batch is (batch_size, context_vector_size), we need to expand it
            if context_batch.dim() == 2:
                # Expand to (batch_size, num_branches, context_vector_size) - same context for all branches
                num_branches = imgs_batch.shape[1]
                context_batch = context_batch.unsqueeze(1).expand(-1, num_branches, -1)
            outputs = model(imgs_batch, context_batch)
            loss = calculate_loss(criterion, outputs, targets, reg_coeff_diag, reg_coeff_off)
            
            # Accumulate loss weighted by batch size for proper averaging over all samples
            batch_size = outputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            output_dim = outputs.shape[1]
            predicted_values = outputs.cpu().numpy()
            true_values = targets.cpu().numpy()

            predictions.append([imgs_batch.cpu().numpy(), true_values, predicted_values])

            all_outputs.extend(predicted_values)
            all_targets.extend(true_values)

    # Compute average loss over all samples (not batches) for consistency
    avg_loss = total_loss / total_samples if total_samples > 0 else 0

    all_targets = np.array(all_targets).reshape(-1, output_dim)
    all_outputs = np.array(all_outputs).reshape(-1, output_dim)
    
    global_acc, vec_local_acc = calculate_local_global_accuracy(all_targets, all_outputs, epsilon)
    local_acc = np.round(np.mean(vec_local_acc), 2)
        
    mse = mean_squared_error(all_targets, all_outputs)

    return avg_loss, global_acc, local_acc, mse, predictions, vec_local_acc

def collect_performance_metrics(model, test_loader, device=DEVICE):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for imgs_batch, context_batch, targets in tqdm(test_loader, desc="Evaluating"):
            imgs_batch = imgs_batch.to(device)
            context_batch = context_batch.to(device)
            targets = targets.to(device)
            
            # Expand context if needed (same as in training loop)
            if context_batch.dim() == 2:
                # Expand to (batch_size, num_branches, context_vector_size) - same context for all branches
                num_branches = imgs_batch.shape[1]
                context_batch = context_batch.unsqueeze(1).expand(-1, num_branches, -1)
            
            outputs = model(imgs_batch, context_batch)
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
    local_accuracy = np.round(np.mean((abs_diff < epsilon).astype(float), axis=0)*100, 2)

    # Global accuracy: proportion of samples where all dimensions are within epsilon
    global_accuracy =  np.round(np.mean((np.max(abs_diff, axis=1) <= epsilon).astype(float))*100, 2)

    return  global_accuracy, local_accuracy 

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
        mode = 'MW-' + str(result['param_names'][-2]) + '-' + str(result['param_names'][-1])
        base_model = result['config']['params'].get('base_model', 'N/A')
        init_weights = True if result['train_params']['init_weights'] is not None else False  
        epsilon = result['train_params']['epsilon']

        train_params = result['train_params']
        results_data.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            'base_model': base_model,
            'mode': mode,
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
            'reg_coeff_diag': train_params.get('reg_coeff_diag', '0.1'),
            'reg_coeff_off': train_params.get('reg_coeff_off', '1.0'),
            'criterion': train_params.get('criterion', nn.MSELoss()),
            'epsilon': epsilon,
            'test_accuracy_global': result['global_test_accuracy'],
            'test_accuracy_local':result['local_test_accuracy'],
            'MSE': result['metrics']['MSE'],
            'MAE': result['metrics']['MAE'],
            'R2': result['metrics']['R2'],
            # 'MAPE': result['metrics']['MAPE'],
            # 'CCC': result['metrics']['CCC'] # Concordance Correlation Coefficient

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
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        
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