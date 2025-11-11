import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import h5py
import glob
import json
from pathlib import Path
from typing import Union

def exp1_plot_mse_mae_r2(csv_path:str, system_name:str, output_path:str='Results/Figs', cmap:str='viridis', r2_amplitude:float=1e5, r2_scale:int=500, 
                         min_r2_color:str='#ffeda0', max_r2_color:str='#f03b20'):
    """
    Create a scatter plot of MSE vs MAE with point sizes proportional to R² values.
    
    Args:
        csv_path (str): Path to the CSV file containing model results
        system_name (str): Name of the system to filter models (e.g., '-2-1' for exact match of systems ending with -2-1)
        output_path (str, optional): Path to save the plot. If None, plot will be displayed
        cmap (str): Name of the colormap to use ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'dark_blue')
                   or 'custom' to use min_r2_color and max_r2_color
        r2_scale (float): Scaling factor for R² values to determine point sizes
        min_r2_color (str): Color for lowest R² value (hex code) when using custom colormap
        max_r2_color (str): Color for highest R² value (hex code) when using custom colormap
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for exact system name match (systems that end with the specified system_name)
    df = df[df['model_name'].str.endswith(system_name)]

    # Split data into two groups based on mode
    df_global = df[df['mode'].str.contains('GCM')]
    df_local = df[df['mode'].str.contains('LCM')]

    # Calculate averages for each group
    avg_global = {
        'MSE': df_global['MSE_mean'].mean(),
        'MAE': df_global['MAE_mean'].mean(),
        'R2': df_global['R2_mean'].mean()
    }
    
    avg_local = {
        'MSE': df_local['MSE_mean'].mean(),
        'MAE': df_local['MAE_mean'].mean(),
        'R2': df_local['R2_mean'].mean()
    }
    
    # Set the style
    # plt.style.use('ggplot')
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create custom colormap if requested
    if cmap == 'custom':
        colors = [min_r2_color, max_r2_color]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    elif cmap == 'dark_blue':
        # Use seaborn's dark_palette with #69d color
        cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    
    # Get global min and max R² values for consistent color scaling
    r2_min = min(df['R2_mean'].min(), avg_global['R2'], avg_local['R2'])
    r2_max = max(df['R2_mean'].max(), avg_global['R2'], avg_local['R2'])
    
    # Create scatter plots for each group
    scatter_global = ax.scatter(
        df_global['MSE_mean'],
        df_global['MAE_mean'],
        # s = r2_amplitude *np.exp(-df_global['R2'] * r2_amplitude),
        s = r2_scale,
        alpha=0.8,
        c=df_global['R2_mean'],
        cmap=cmap,
        marker='*',  # Star marker
        label='Global Capacitance Matrix',
        vmin=r2_min,
        vmax=r2_max
    )

    scatter_local = ax.scatter(
        df_local['MSE_mean'],
        df_local['MAE_mean'],
        # s= r2_amplitude * np.exp(-df_local['R2'] * r2_scale),
        s = r2_scale,
        alpha=0.8,
        c=df_local['R2_mean'],
        cmap=cmap,
        marker='p',  # Pentagon marker
        label='Local Capacitance Matrix',
        vmin=r2_min,
        vmax=r2_max
    )

    # Add average points
    ax.scatter(
        avg_global['MSE'],
        avg_global['MAE'],
        # s= r2_amplitude * np.exp(-avg_global['R2'] * r2_scale),
        s = r2_scale,
        alpha=0.8,
        c=avg_global['R2'],
        cmap=cmap,
        marker='*',
        edgecolor='red',
        linewidth=2,
        label='Average GCM',
        vmin=r2_min,
        vmax=r2_max
    )

    ax.scatter(
        avg_local['MSE'],
        avg_local['MAE'],
        # s= r2_amplitude * np.exp(-avg_local['R2'] * r2_scale),
        s = r2_scale,
        alpha=0.8,
        c=avg_local['R2'],
        cmap=cmap,
        marker='p',
        edgecolor='red',
        linewidth=2,
        label='Average LCM',
        vmin=r2_min,
        vmax=r2_max
    )
    
    # Add labels and title
    N,S = int(system_name.split('-')[1]), int(system_name.split('-')[2])
    K = N+S
    config_name = f'({N},{S})' + f'-{system_name.split("-")[-1]}' if len(system_name.split("-")) > 3 else f'({N},{S})'

    ax.set_xlabel('Mean Squared Error (MSE)', fontsize=20)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=20)
    ax.set_title(f'Model Performance: MSE vs MAE\nColor indicates R² value\nDevice: {config_name}', fontsize=20)
    
    # Set tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter_global)
    cbar.set_label('R² Value', fontsize=20)
    cbar.ax.tick_params(labelsize=16)  # Set colorbar tick label size
    
    # Add model names as annotations
    for i, row in df.iterrows():
        ax.annotate(
            row['model_name'].split('-')[0],
            (row['MSE_mean'], row['MAE_mean']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12
        )
    
    # Add legend
    ax.legend(title='Capaciatance matrix type',
              loc='lower right', 
              markerscale=0.5,
              fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        output_path = output_path + f'/{config_name}_mse_mae_r2.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 

def prepare_csv_stats(csv_path: str, 
                     columns_to_keep: list = None,
                     columns_to_avg: list = None,
                     group_by: list = ['base_model', 'mode'],
                     seeds: list[int] = None,
                     output_path: str = None) -> pd.DataFrame:
    """
    Process CSV data to calculate statistics (mean and std) across different seeds for specified columns.
    
    Args:
        csv_path (str): Path to the input CSV file (must end with .csv)
        columns_to_keep (list): List of column names to keep in the output. If None, keeps all columns
        columns_to_avg (list): List of column names to calculate mean and std for. If None, uses numeric columns
        group_by (list): Column names to group by (default: ['base_model', 'mode'])
        output_path (str): Path to save the processed CSV (must end with .csv).
        seeds (list): List of seed values to process. If None, uses all seeds.
    Returns:
        pd.DataFrame: Processed dataframe with original columns and calculated statistics
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Select columns to keep
    if columns_to_keep is None:
        columns_to_keep = df.columns.tolist()
    
    # Determine columns to average if not specified
    if columns_to_avg is None:
        raise ValueError("You must provide a list of columns to average")
    
    # Calculate statistics for each column across all seeds
    stats = {}
    for col in columns_to_avg:
        if col in df.columns:
            # First group by the specified columns and seed
            grouped = df.groupby(group_by + ['seed'])[col].mean()
            # Then calculate mean and std across seeds for each group
            stats[f'{col}_mean'] = grouped.groupby(level=group_by).mean()
            stats[f'{col}_std'] = grouped.groupby(level=group_by).std()
    
    # Create statistics dataframe
    stats_df = pd.DataFrame(stats).reset_index()
    
    # Filter by seeds if specified (after calculating statistics)
    if seeds is not None:
        df = df[df['seed'].isin(seeds)]
    
    # Get unique combinations of group_by columns from filtered data
    filtered_groups = df[group_by].drop_duplicates()
    
    # Merge statistics with filtered groups
    stats_df = pd.merge(
        filtered_groups,
        stats_df,
        on=group_by,
        how='left'
    )
    
    # Update mode column values
    stats_df.loc[stats_df['mode'].str.contains('False'), 'mode'] = 'LCM'
    stats_df.loc[stats_df['mode'].str.contains('True'), 'mode'] = 'GCM'
    
    # Save if output path is provided
    if output_path:
        stats_df.to_csv(output_path, index=False)
    
    return stats_df

def plot_average_loss_curves(main_folder_paths: Union[str, list],
                           systems: list = None,
                           seeds: list = None,
                           metrics: list = ['train_losses', 'val_losses'],
                           output_path: str = None,
                           figsize: tuple = (10, 6),
                           title: str = 'Average Training and Validation Loss Curves',
                           ylabel: str = 'Loss',
                           xlabel: str = 'Epochs',
                           grid: bool = True,
                           legend: bool = True,
                           save_format: str = 'png',
                           dpi: int = 300) -> None:
    """
    Plot average and std of loss curves for specified systems and seeds using data from model_results.csv.
    Matches run folders by comparing random_state from results.json with seed from model_results.csv,
    and ensures the mode (GCM/LCM) matches between CSV and results.json.
    """
    # Convert single path to list for uniform handling
    if isinstance(main_folder_paths, str):
        main_folder_paths = [main_folder_paths]
    
    # Define color schemes for GCM and LCM
    gcm_colors = ['#2281c4', '#1f77b4']  # Blue shades
    lcm_colors = ['#971c1c', '#e36869']  # Red shades
    
    # Define markers for different architectures
    markers = ['o', 's', '^', 'D', 'v']
    
    plt.figure(figsize=figsize)
    
    for folder_path in main_folder_paths:
        # Read the model results CSV
        csv_path = os.path.join(folder_path, 'model_results.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"model_results.csv not found in {folder_path}")
        df = pd.read_csv(csv_path)
        
        # Filter by systems if specified
        if systems is not None:
            df = df[df['model_name'].isin(systems)]
        
        # Filter by seeds if specified
        if seeds is not None:
            df = df[df['seed'].isin(seeds)]
        else:
            raise ValueError("You must provide a list of seeds")
            
        # Get unique architectures and create marker mapping
        architectures = sorted(df['base_model'].unique())
        arch_to_marker = {arch: markers[i % len(markers)] for i, arch in enumerate(architectures)}
        
        # Process each unique model_name and mode combination
        for (model_name, mode), group_df in df.groupby(['model_name', 'mode']):
            # Get base model name and create model folder path
            base_model = group_df['base_model'].iloc[0]
            model_folder = os.path.join(folder_path, base_model)
            
            if not os.path.exists(model_folder):
                print(f"Model folder {model_folder} not found for {model_name}")
                continue
            
            # Get color based on mode
            color_scheme = gcm_colors if mode == 'GCM' else lcm_colors
            
            # Process each metric
            for metric_idx, metric in enumerate(metrics):
                all_curves = []
                max_epochs = 0
                
                # Split system name into architecture and device parts
                arch, device = model_name.split('-', 1)
                resnet_num = arch.replace('Rn', '')
                model_folder_name = f'resnet{resnet_num}'
                model_folder = os.path.join(folder_path, model_folder_name)
                
                # Find all run folders for this system - use exact system name match
                run_folders = []
                for f in os.listdir(model_folder):
                    if os.path.isdir(os.path.join(model_folder, f)):
                        # Extract the system name from the folder name (everything before the timestamp)
                        folder_system = f.split('_')[0]
                        if folder_system == model_name:  # Exact match
                            run_folders.append(f)
                
                
                # Process each run (seed) for this model
                for _, row in group_df.iterrows():
                    seed = row['seed']
                    # Find the folder with matching random_state and mode
                    matching_folder = None
                    for run_folder in run_folders:
                        json_path = os.path.join(model_folder, run_folder, 'results.json')
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r') as f:
                                    results = json.load(f)
                                    # Check both random_state and mode
                                    if (results['train_params']['random_state'] == seed and
                                        ((mode == 'GCM' and results['param_names'][-1] == 'True') or
                                         (mode == 'LCM' and results['param_names'][-1] == 'False'))):
                                        matching_folder = run_folder
                                        break
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"Error reading {json_path}: {str(e)}")
                                continue
                    
                    if matching_folder is None:
                        print(f"No folder found for {model_name} ({mode}) with seed {seed}, path: {os.path.join(model_folder, run_folder)}")
                        continue
                    
                    # Load and process the curve data
                    h5_path = os.path.join(model_folder, matching_folder, 'results_and_history.h5')
                    if os.path.exists(h5_path):
                        try:
                            with h5py.File(h5_path, 'r') as h5_file:
                                if 'history' not in h5_file:
                                    print(f"No history found in {h5_path}")
                                    continue
                                    
                                if metric in h5_file['history']:
                                    curve = h5_file['history'][metric][:]
                                    all_curves.append(curve)
                                    max_epochs = max(max_epochs, len(curve))
                                else:
                                    print(f"Metric {metric} not found in {matching_folder}")
                        except Exception as e:
                            print(f"Error reading {h5_path}: {str(e)}")
                            continue
                    else:
                        print(f"H5 file not found: {h5_path}")
                
                if not all_curves:
                    print(f"No {metric} data found for {model_name} ({mode})")
                    continue
                
                # Calculate statistics
                curves_array = np.array(all_curves)
                mean_curve = np.nanmean(curves_array, axis=0)
                std_curve = np.nanstd(curves_array, axis=0)
                epochs = np.arange(1, max_epochs + 1)
                
                # Get color and marker
                color = color_scheme[metric_idx % len(color_scheme)]
                marker = arch_to_marker[base_model] if len(architectures) > 1 else None
                markersize = 4 if marker else None
                
                # Create label
                label = f"{model_name} ({mode}) - {'Training' if metric == 'train_losses' else 'Validation'}"
                
                # Plot mean curve
                linestyle = '--' if metric == 'val_losses' else '-'
                plt.plot(epochs, mean_curve, label=label, linewidth=2, color=color,
                        linestyle=linestyle, marker=marker, markersize=markersize, markevery=5)
                plt.yscale('log')
                
                # Plot std region
                plt.fill_between(epochs,
                               mean_curve - std_curve,
                               mean_curve + std_curve,
                               alpha=0.3,
                               color=color,
                               edgecolor=color,
                               linewidth=0.5)
    
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(np.arange(0, max_epochs+1, 2 if max_epochs%2 == 0 else 3), fontsize=14)
    plt.yticks(fontsize=14)
    if grid:
        plt.grid(True, linestyle='-', alpha=0.7)
    if legend:
        plt.legend(title='System/Mode/Metric', loc='upper right', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format=save_format, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def mse_per_element(y,y_true,element_index):
    return np.sum([(a[element_index]-b[element_index])**2 for a,b in zip(y_true,y)])/len(y_true)

def load_preds_to_nested_dict(main_folder_paths: Union[str, list],
                              systems: list = None,
                              seeds: list = None):
    """
    Load predictions from all systems and seeds into a nested dictionary.
    """
    # Convert single path to list for uniform handling
    if isinstance(main_folder_paths, str):
        main_folder_paths = [main_folder_paths]
    
    if systems is None:
        raise ValueError("You must provide a list of system names (e.g., ['Rn14-2-0', 'Rn10-2-1'])")

    # Get unique devices and architectures
    devices = sorted(list(set([system.split('-', 1)[1] for system in systems])))
    architectures = sorted(list(set([system.split('-')[0] for system in systems])))
    
    # Initialize empty nested dictionary structure
    data_dict = {
        'GCM': {device: {seed: {arch: {'pred': None, 'true': None} 
                               for arch in architectures} 
                        for seed in seeds} 
                for device in devices},
        'LCM': {device: {seed: {arch: {'pred': None, 'true': None} 
                               for arch in architectures} 
                        for seed in seeds} 
                for device in devices}
    }
    
    # Process each folder
    for main_folder_path in main_folder_paths:
        base_path = Path(main_folder_path)
        
        # Process each system
        for system in systems:
            arch, device = system.split('-', 1)
            resnet_num = arch.replace('Rn', '')
            model_folder_name = f'resnet{resnet_num}'
            model_folder = base_path / model_folder_name
            
            if not model_folder.is_dir():
                print(f"Model folder {model_folder_name} not found for system {system}")
                continue
            
            # Find all run folders for this system
            run_folders = [f for f in model_folder.iterdir() if f.is_dir() and f.name.startswith(system)]
            
            for run_folder in run_folders:
                # Check results.json to determine model type and seed
                try:
                    with open(run_folder / 'results.json', 'r') as json_file:
                        results = json.load(json_file)
                        current_seed = results['train_params']['random_state']
                        
                        # Skip if seed not in the requested seeds
                        if seeds is not None and current_seed not in seeds:
                            continue
                        
                        # Determine if GCM or LCM based on last parameter name
                        is_gcm = results['param_names'][-1] == 'True'
                        model_type = 'GCM' if is_gcm else 'LCM'
                        
                        # Load predictions
                        pred_path = run_folder / 'predictions.h5'
                        if not pred_path.exists():
                            print(f"Predictions file not found for {system} in {run_folder}")
                            continue
                        
                        with h5py.File(pred_path, 'r') as h5_file:
                            pred = h5_file['outputs'][:]
                            true = h5_file['targets'][:]
                            
                            # Update the dictionary
                            data_dict[model_type][device][current_seed][arch]['pred'] = pred
                            data_dict[model_type][device][current_seed][arch]['true'] = true
                            
                except Exception as e:
                    print(f"Error processing {run_folder}: {str(e)}")
                    continue
    return data_dict

def plot_mse_chosen_element(element_index: int = 0, 
                            interaction_type: str = 'dd',
                            main_folder_paths: Union[str, list] = None,
                            systems: list = None,
                            seeds: list = None,
                            output_path: str = None,
                            figsize: tuple = (10, 6)):
    """
    Plot the MSE of the chosen element of the predicted vector for all systems, with averages and standard deviations
    across seeds as a scatter plot against device sizes.

    Args:
        element_index (int): Index of the element to plot
        interaction_type (str): Type of interaction of capacitance matrix: dot-dot or dot-gate
        main_folder_paths (Union[str, list]): Path or list of paths to the results folders 
        systems (list): List of system names (e.g., ['Rn14-2-0', 'Rn10-2-1'])
        seeds (list): List of seeds to include (e.g., [42, 5, 8])
        output_path (str): Path to save the plot. If None, plot will be displayed
        figsize (tuple): Figure size (width, height)
    """
    if interaction_type != 'dd' and interaction_type != 'dg':
        raise ValueError(f"Invalid interaction type: {interaction_type}. Must be 'dd' for dot-dot or 'dg' for dot-gate.")
    # if element_index >= 2 or interaction_type != 'dd':
        # raise ValueError(f"This version of the code only supports element_index = 0 or 1 for dot-dot interaction. You provided: {element_index} and {interaction_type}")
    # Load the data
    mse_data = load_preds_to_nested_dict(main_folder_paths, systems, seeds)
    devices = list(mse_data['GCM'].keys()) 
    seeds = list(mse_data['GCM'][devices[0]].keys())
    architectures = list(mse_data['GCM'][devices[0]][seeds[0]].keys())

    # Calculate MSE for each combination
    for model_type in ['GCM', 'LCM']:
        for device in devices:
            for seed in seeds:
                for arch in architectures:
                    try:
                        mse_data[model_type][device][seed][arch]['mse'] = mse_per_element(
                            mse_data[model_type][device][seed][arch]['pred'],
                            mse_data[model_type][device][seed][arch]['true'],
                            element_index
                        )
                        # Remove the raw data to save memory
                        mse_data[model_type][device][seed][arch].pop('pred')
                        mse_data[model_type][device][seed][arch].pop('true')
                    except TypeError:
                        # mse_data[model_type][device][seed][arch]['mse'] = np.nan
                        print(f"The entry of {model_type}/{device}/{seed}/{arch} is np.nan. Skipping...")
                        continue

    # Create figure
    plt.figure(figsize=figsize)
    
    # Colors for GCM and LCM
    colors = {'GCM': '#2281c4', 'LCM': '#971c1c'}
    markers = {'GCM': 'o', 'LCM': 's'}  # Different markers for GCM and LCM
    
    # Calculate statistics across both seeds and architectures for each device and model type
    device_stats = {
        'GCM': {device: {'mean': None, 'std': None} for device in devices},
        'LCM': {device: {'mean': None, 'std': None} for device in devices}
    }
    
    # Calculate statistics
    for model_type in ['GCM', 'LCM']:
        for device in devices:
            # Collect all MSE values for this device and model type across all seeds and architectures
            all_mse_values = []
            for seed in seeds:
                for arch in architectures:
                    try:
                        mse = mse_data[model_type][device][seed][arch]['mse']
                        if mse is not None:
                            all_mse_values.append(float(mse))
                    except (KeyError, TypeError, ValueError):
                        continue
            
            if all_mse_values:  # Only calculate stats if we have data
                device_stats[model_type][device]['mean'] = np.mean(all_mse_values)
                device_stats[model_type][device]['std'] = np.std(all_mse_values)

    # Create a mapping of device names to x-axis positions
    device_to_x = {device: i for i, device in enumerate(devices)}
    x_positions = [device_to_x[device] for device in devices]
    
    # Plot statistics for each model type
    for model_type in ['GCM', 'LCM']:
        # Collect means and stds for this model type
        means = []
        stds = []
        x_vals = []
        for device in devices:
            try:
                mean_val = device_stats[model_type][device]['mean']
                std_val = device_stats[model_type][device]['std']
                
                if mean_val is not None and std_val is not None and not np.isnan(mean_val) and not np.isnan(std_val):
                    means.append(float(mean_val))
                    stds.append(float(std_val))
                    x_vals.append(device_to_x[device])
            except (KeyError, TypeError, ValueError) as e:
                print(f"Skipping {model_type}/{device}: {str(e)}")
                continue
        
        if means:  # Only plot if we have data
            # Convert lists to numpy arrays
            means = np.array(means)
            stds = np.array(stds)
            x_vals = np.array(x_vals)
            
            plt.errorbar(x_vals, means, yerr=stds,
                       fmt=markers[model_type], color=colors[model_type],
                       label=f'{model_type}',
                       capsize=5, capthick=1, elinewidth=1,
                       markersize=8, alpha=0.7)

    # plt.xlabel('Device Configuration', fontsize=12)
    # plt.ylabel('MSE (averaged across runs and architectures)', fontsize=18)
    plt.ylabel('MSE', fontsize=18)

    # plt.yscale('log')
    if element_index == 0 and interaction_type == 'dd':
        plt.title(f'MSE Statistics for Dot-Dot Interaction\nElement: 1st dot capacitance', fontsize=20)
    elif element_index == 1 and interaction_type == 'dd':
        plt.title(f'MSE Statistics for Dot-Dot Interaction\nElement: capacitance 1st dot - 2nd dot', fontsize=20)
    elif element_index == 3 and interaction_type == 'dd':
            plt.title(f'MSE Statistics for Dot-Dot Interaction\nElement: 2nd dot capacitance', fontsize=20)
    elif element_index == 5 and interaction_type == 'dd':
        plt.title(f'MSE Statistics for Dot-Dot Interaction\nElement: sensor dot capacitance', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=16, markerscale=1.5)
    
    # Set x-axis ticks to show device names
    plt.xticks(x_positions, devices, rotation=45, ha='right', fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        df = pd.DataFrame(device_stats)
        df.to_csv(output_path.replace('.png', '.csv'), index=False)
    else:
        plt.show()

def plot_residuals_chosen_element(element_index: int = 0, 
                                  interaction_type: str = 'dd',
                                  main_folder_paths: Union[str, list] = None,
                                  systems: list = None,
                                  seeds: list = None,
                                  output_path: str = None,
                                  figsize: tuple = (12, 6)):    
    """
    Plot scatter plots of true vs predicted values for both GCM and LCM models, with R-squared metrics.
    Creates two subplots side by side, one for each model type.

    Args:
        element_index (int): Index of the element to plot
        interaction_type (str): Type of interaction of capacitance matrix: dot-dot or dot-gate
        main_folder_paths (Union[str, list]): Path or list of paths to the results folders 
        systems (list): List of system names (e.g., ['Rn14-2-0', 'Rn10-2-1'])
        seeds (list): List of seeds to include (e.g., [42, 5, 8])
        output_path (str): Path to save the plot. If None, plot will be displayed
        figsize (tuple): Figure size (width, height)
    """
    if interaction_type != 'dd' and interaction_type != 'dg':
        raise ValueError(f"Invalid interaction type: {interaction_type}. Must be 'dd' for dot-dot or 'dg' for dot-gate.")
    # if element_index >= 2 or interaction_type != 'dd':
        # raise ValueError(f"This version of the code only supports element_index = 0 or 1 for dot-dot interaction. You provided: {element_index} and {interaction_type}")
    
    # Load the data
    mse_data = load_preds_to_nested_dict(main_folder_paths, systems, seeds)
    devices = list(mse_data['GCM'].keys()) 
    seeds = list(mse_data['GCM'][devices[0]].keys())
    architectures = list(mse_data['GCM'][devices[0]][seeds[0]].keys())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors for GCM and LCM
    colors = {'GCM': '#2281c4', 'LCM': '#971c1c'}
    
    # Process each model type
    for model_type, ax in zip(['GCM', 'LCM'], [ax1, ax2]):
        # Dictionary to store R-squared values for each system
        system_r2 = {device: [] for device in devices}
        all_true = []
        all_pred = []
        
        # Calculate R-squared for each system, architecture, and seed combination
        for device in devices:
            for seed in seeds:
                for arch in architectures:
                    try:
                        if 'pred' in mse_data[model_type][device][seed][arch] and 'true' in mse_data[model_type][device][seed][arch]:
                            pred = mse_data[model_type][device][seed][arch]['pred']
                            true = mse_data[model_type][device][seed][arch]['true']
                            
                            # Extract the chosen element
                            pred_element = pred[:, element_index]
                            true_element = true[:, element_index]
                            
                            # Calculate R-squared for this specific combination
                            ss_res = np.sum((true_element - pred_element) ** 2)
                            
                            # Check if all true values are zero
                            if np.all(true_element == 0):
                                # For zero true values, R-squared is 1 if predictions are also zero
                                # and 0 if predictions are non-zero
                                r2 = 1.0 if np.all(pred_element == 0) else 0.0
                            else:
                                try:
                                    ss_tot = np.sum((true_element - np.mean(true_element)) ** 2)
                                    r2 = 1 - (ss_res / ss_tot)
                                except ZeroDivisionError:
                                    # This should not happen now, but keeping as safety
                                    epsilon = 1e-8
                                    ss_tot = np.sum((true_element - np.mean(true_element)) ** 2) + epsilon
                                    r2 = 1 - (ss_res / ss_tot)
                            
                            system_r2[device].append(r2)
                            
                            # Also collect all values for the scatter plot
                            all_true.extend(true_element)
                            all_pred.extend(pred_element)
                    except (KeyError, TypeError, ValueError) as e:
                        continue
        
        # Calculate average R-squared for each system
        avg_r2 = {device: np.mean(r2s) if r2s else np.nan for device, r2s in system_r2.items()}
        # Calculate overall R-squared across all data
        overall_r2 = np.mean([r2 for r2s in system_r2.values() for r2 in r2s if not np.isnan(r2)])
        
        # Convert to numpy arrays for plotting
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        # Create scatter plot
        ax.scatter(all_true, all_pred, alpha=0.5, color=colors[model_type], s=10)
        
        # Add perfect prediction line
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect prediction')
        
        # Add R-squared text with both overall and per-system values
        r2_text = f'Overall R² = {overall_r2:.4f}\n'
        r2_text += 'Per-system R²:\n'
        for device, r2 in avg_r2.items():
            i = 1
            if not np.isnan(r2):
                if i<len(devices)-1:
                    r2_text += f'{device}: {r2:.4f}\n'
                else:
                    r2_text += f'{device}: {r2:.4f}'
                i += 1

        ax.text(0.05, 1.25, r2_text,
                transform=ax.transAxes, 
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and title
        if element_index == 0:
            element_name = "1st dot capacitance"
        else:
            element_name = "capacitance 1st dot - 2nd dot"
            
        ax.set_xlabel('True Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{model_type}\nElement: {element_name}', fontsize=14)
        
        # Make plot square
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='lower right')

    plt.suptitle(f'True vs Predicted Values for Dot-Dot Interaction\n', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save R-squared values to CSV
        r2_data = {
            'Model': [],
            'System': [],
            'R2': []
        }
        for model_type in ['GCM', 'LCM']:
            for device, r2s in system_r2.items():
                for r2 in r2s:
                    if not np.isnan(r2):
                        r2_data['Model'].append(model_type)
                        r2_data['System'].append(device)
                        r2_data['R2'].append(r2)
        
        df = pd.DataFrame(r2_data)
        df.to_csv(output_path.replace('.png', '_r2.csv'), index=False)
    else:
        plt.show()
    

def mapping_cm_to_vector(i: int, j: int, K: int):
    vector_index = 0
    w = 0
    while w<i:
        vector_index += K - w  
        w += 1
    vector_index += j-i
    
    return vector_index

def plot_capacitance_matrix_elements(main_folder_paths: Union[str, list],
                                   systems: list = None,
                                   seeds: list = None,
                                   output_path: str = None,
                                   figsize: tuple = (15, 15),
                                   title: str = 'Dot-Dot Capacitance Matrix Elements',
                                   grid: bool = True,
                                   save_format: str = 'png',
                                   dpi: int = 300) -> None:
    """
    Create two separate grids of subplots showing predicted vs true values for each element of the dot-dot capacitance matrix,
    one for GCM and one for LCM. Each subplot's position corresponds to the position in the upper triangular part of the C_DD matrix.
    The plots are arranged to reflect the matrix structure, with diagonal elements (self-capacitances) on the diagonal.
    R² values are shown in the lower triangular part of the matrix.
    """
    # Convert single path to list for uniform handling
    if isinstance(main_folder_paths, str):
        main_folder_paths = [main_folder_paths]
    
    if systems is None:
        raise ValueError("You must provide a list of system names")
        
    # Extract system size from first system name (e.g., '2-1' from 'Rn14-2-1')
    system_size = '-'.join(systems[0].split('-')[1:3])
    system_size_for_plot = '(' + systems[0].split('-')[1] + ',' + systems[0].split('-')[2] + ')'
    
    # Filter systems by size
    filtered_systems = [s for s in systems if system_size in s]
    if not filtered_systems:
        raise ValueError(f"No systems found with size {system_size}")
    
    # Load predictions data
    data_dict = load_preds_to_nested_dict(main_folder_paths, filtered_systems, seeds)
    
    # Get dimensions from first available data
    first_device = list(data_dict['GCM'].keys())[0]
    first_seed = list(data_dict['GCM'][first_device].keys())[0]
    first_arch = list(data_dict['GCM'][first_device][first_seed].keys())[0]
    
    # Get K (matrix size) from the first prediction
    pred_shape = data_dict['GCM'][first_device][first_seed][first_arch]['pred'].shape[1]
    # Calculate K from the number of elements in upper triangular matrix
    # K*(K+1)/2 = pred_shape -> K^2 + K - 2*pred_shape = 0
    K = int(system_size.split('-')[0]) + int(system_size.split('-')[1])
    print(f'K: {K} for systems: {systems}')
    
    # Colors for GCM and LCM
    colors = {'GCM': '#2281c4', 'LCM': '#971c1c'}
    
    # Create separate figures for GCM and LCM
    for model_type in ['GCM', 'LCM']:
        fig, axes = plt.subplots(K, K, figsize=figsize)
        fig.suptitle(f'{title} - {model_type}\nSystem: {system_size_for_plot}\n\n', fontsize=30, y=0.95)
        
        # First pass: Calculate all R² values
        r2_values = {}
        for i in range(K):
            for j in range(i+1):  # Only upper triangular part including diagonal
                all_true = []
                all_pred = []
                
                for device in data_dict[model_type].keys():
                    for seed in data_dict[model_type][device].keys():
                        for arch in data_dict[model_type][device][seed].keys():
                            try:
                                if 'pred' in data_dict[model_type][device][seed][arch] and 'true' in data_dict[model_type][device][seed][arch]:
                                    pred = data_dict[model_type][device][seed][arch]['pred']
                                    true = data_dict[model_type][device][seed][arch]['true']
                                    
                                    flat_idx = mapping_cm_to_vector(j, i, K)
                                    if flat_idx < pred.shape[1]:
                                        pred_element = pred[:, flat_idx]
                                        true_element = true[:, flat_idx]
                                        
                                        all_true.extend(true_element)
                                        all_pred.extend(pred_element)
                            except (KeyError, TypeError, ValueError) as e:
                                continue
                
                if all_true and all_pred:
                    all_true = np.array(all_true)
                    all_pred = np.array(all_pred)
                    
                    ss_res = np.sum((all_true - all_pred) ** 2)
                    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    r2_values[(j,i)] = r2
        
        # Second pass: Create the plot
        for i in range(K):
            for j in range(K):
                ax = axes[j, i]  # Note: j,i to match matrix visualization (i is column, j is row)
                
                # Add overlay box for bottom-left subplot
                # Calculate position for the overlay box (bottom-left subplot)
                box_width = 0.15  # Width of the box relative to figure
                box_height = 0.1  # Height of the box relative to figure
                box_left = 0.05   # Left position of the box
                box_bottom = 0.05 # Bottom position of the box
                
                # Create overlay axes
                overlay_ax = fig.add_axes([box_left, box_bottom, box_width, box_height])
                overlay_ax.set_xticks([])
                overlay_ax.set_yticks([])
                overlay_ax.set_frame_on(False)  # Hide the frame of the overlay axes
                overlay_ax.set_facecolor('none')  # Make the background transparent
                
                # Create text for R² values
                r2_text = "R² Values:\n\n"
                for (j_val, i_val), r2 in sorted(r2_values.items()):
                    k = 1
                    if k<len(r2_values)-2:
                        r2_text += f"({j_val+1},{i_val+1}): {r2:.3f}\n"
                    else:
                        r2_text += f"({j_val+1},{i_val+1}): {r2:.3f}"
                    k += 1
                
                # Add text 'HERE' to the overlay box
                overlay_ax.text(0.5, 1.8, r2_text,
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=overlay_ax.transAxes,
                             fontsize=27,
                             bbox=dict(facecolor='white',
                                     edgecolor='black',
                                     alpha=0.9,
                                     boxstyle='round,pad=0.25'))
                
                # Handle lower triangular part
                if j > i:
                    ax.axis('off')
                    continue
                
                # Handle upper triangular part (including diagonal)
                if j <= i:
                    all_true = []
                    all_pred = []
                    
                    for device in data_dict[model_type].keys():
                        for seed in data_dict[model_type][device].keys():
                            for arch in data_dict[model_type][device][seed].keys():
                                try:
                                    if 'pred' in data_dict[model_type][device][seed][arch] and 'true' in data_dict[model_type][device][seed][arch]:
                                        pred = data_dict[model_type][device][seed][arch]['pred']
                                        true = data_dict[model_type][device][seed][arch]['true']
                                        
                                        flat_idx = mapping_cm_to_vector(j, i, K)
                                        if flat_idx < pred.shape[1]:
                                            pred_element = pred[:, flat_idx]
                                            true_element = true[:, flat_idx]
                                            
                                            all_true.extend(true_element)
                                            all_pred.extend(pred_element)
                                except (KeyError, TypeError, ValueError) as e:
                                    continue
                    
                    if all_true and all_pred:
                        all_true = np.array(all_true)
                        all_pred = np.array(all_pred)
                        
                        # Create scatter plot
                        ax.scatter(all_true, all_pred, 
                                 alpha=0.5, color=colors[model_type], 
                                 s=10)
                        
                        # Add perfect prediction line
                        min_val = min(all_true.min(), all_pred.min())
                        max_val = max(all_true.max(), all_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 
                               'k--', alpha=0.5)
                
                # Set labels and title
                # ax.set_title(f'({j},{i})->{mapping_cm_to_vector(j,i,K)}', fontsize=14)
                ax.set_title(f'({j+1},{i+1})', fontsize=18)

                ax.set_xlabel('True Value', fontsize=16)
                ax.set_ylabel('Predicted Value', fontsize=16)
                
                # Make plot square
                ax.set_aspect('equal', adjustable='box')
                
                # Add grid
                if grid:
                    ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_path:
            # Modify output path to include model type
            model_output_path = output_path.replace(f'.{save_format}', f'_{model_type}.{save_format}')
            plt.savefig(model_output_path, format=save_format, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Save R-squared values to CSV
            r2_data = {
                'i': [],
                'j': [],
                'R2': []
            }
            
            for (j,i), r2 in r2_values.items():
                r2_data['i'].append(i)
                r2_data['j'].append(j)
                r2_data['R2'].append(r2)
            
            df = pd.DataFrame(r2_data)
            df.to_csv(model_output_path.replace(f'.{save_format}', '_r2.csv'), index=False)
        else:
            plt.show()


