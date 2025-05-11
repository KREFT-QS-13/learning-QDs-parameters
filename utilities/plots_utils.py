import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os

def exp1_plot_mse_mae_r2(csv_path:str, system_name:str, output_path:str='Results/Figs', cmap:str='viridis', r2_amplitude:float=1e5, r2_scale:int=0.05, 
                         min_r2_color:str='#ffeda0', max_r2_color:str='#f03b20'):
    """
    Create a scatter plot of MSE vs MAE with point sizes proportional to R² values.
    
    Args:
        csv_path (str): Path to the CSV file containing model results
        system_name (str): Name of the system to filter models
        output_path (str, optional): Path to save the plot. If None, plot will be displayed
        cmap (str): Name of the colormap to use ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'dark_blue')
                   or 'custom' to use min_r2_color and max_r2_color
        r2_scale (float): Scaling factor for R² values to determine point sizes
        min_r2_color (str): Color for lowest R² value (hex code) when using custom colormap
        max_r2_color (str): Color for highest R² value (hex code) when using custom colormap
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    df = df[df['model_name'].str.contains(system_name)]

    # Split data into two groups based on mode
    df_global = df[df['mode'].str.contains('True')]
    df_local = df[df['mode'].str.contains('False')]

    # Calculate averages for each group
    avg_global = {
        'MSE': df_global['MSE'].mean(),
        'MAE': df_global['MAE'].mean(),
        'R2': df_global['R2'].mean()
    }
    
    avg_local = {
        'MSE': df_local['MSE'].mean(),
        'MAE': df_local['MAE'].mean(),
        'R2': df_local['R2'].mean()
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
    r2_min = min(df['R2'].min(), avg_global['R2'], avg_local['R2'])
    r2_max = max(df['R2'].max(), avg_global['R2'], avg_local['R2'])
    
    # Create scatter plots for each group
    scatter_global = ax.scatter(
        df_global['MSE'],
        df_global['MAE'],
        # s = r2_amplitude *np.exp(-df_global['R2'] * r2_amplitude),
        s = 500,
        alpha=0.8,
        c=df_global['R2'],
        cmap=cmap,
        marker='*',  # Star marker
        label='Global Capacitance Matrix',
        vmin=r2_min,
        vmax=r2_max
    )

    scatter_local = ax.scatter(
        df_local['MSE'],
        df_local['MAE'],
        # s= r2_amplitude * np.exp(-df_local['R2'] * r2_scale),
        s = 500,
        alpha=0.8,
        c=df_local['R2'],
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
        s = 500,
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
        s = 500,
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
    config_name = f'({K},{N},{S})'

    ax.set_xlabel('Mean Squared Error (MSE)', fontsize=20)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=20)
    ax.set_title(f'Model Performance: MSE vs MAE\nPoint size and color indicate R² value\nDevice: {config_name}', fontsize=20)
    
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
            (row['MSE'], row['MAE']),
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
                     group_by: str = 'base_model',
                     seed_column: str = 'seed',
                     seeds: list = None,
                     output_path: str = None) -> pd.DataFrame:
    """
    Process CSV data to calculate statistics (mean and std) across different seeds for specified columns.
    
    Args:
        csv_path (str): Path to the input CSV file (must end with .csv)
        columns_to_keep (list): List of column names to keep in the output. If None, keeps all columns
        columns_to_avg (list): List of column names to calculate mean and std for. If None, uses numeric columns
        group_by (str): Column name to group by (default: 'base_model')
        seed_column (str): Column name containing seed values (default: 'seed')
        output_path (str): Path to save the processed CSV (must end with .csv).
        seeds (list): List of seed values to process. If None, uses all seeds.
    Returns:
        pd.DataFrame: Processed dataframe with statistics
        
    Raises:
        ValueError: If input or output paths are not valid CSV files
        PermissionError: If there are permission issues accessing the files
        FileNotFoundError: If the input file doesn't exist
    """
    # Validate input path
    if not csv_path.endswith('.csv'):
        raise ValueError(f"Input path must be a CSV file, got: {csv_path}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    # Validate output path if provided
    if output_path is not None:
        if not output_path.endswith('.csv'):
            raise ValueError(f"Output path must be a CSV file, got: {output_path}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except PermissionError:
        raise PermissionError(f"Permission denied: Cannot read file {csv_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")
    
    # If columns_to_keep is None, keep all columns
    if columns_to_keep is None:
        columns_to_keep = df.columns.tolist()
    
    # If columns_to_avg is None, use all numeric columns except the group_by and seed columns
    if seeds is not None:
        df = df[df[seed_column].isin(seeds)]

    if columns_to_avg is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_avg = [col for col in numeric_cols if col not in [group_by, seed_column]]
    
    # Create a copy of the dataframe with only the columns we want to keep
    stats_df = df[columns_to_keep].copy()
    
    # Group by the specified column and calculate statistics
    stats_dict = {}
    for col in columns_to_avg:
        if col in df.columns:
            # Calculate mean and std for each group
            mean_col = df.groupby(group_by)[col].mean()
            std_col = df.groupby(group_by)[col].std()
            
            # Add to stats dictionary
            stats_dict[f'{col}_mean'] = mean_col
            stats_dict[f'{col}_std'] = std_col
    
    # Create a new dataframe with the statistics
    stats_temp = pd.DataFrame(stats_dict)
    stats_temp.reset_index(inplace=True)
    
    # Merge the statistics with the original dataframe
    # First, get unique rows for the group_by column to avoid duplicates
    unique_groups = stats_df.drop_duplicates(subset=[group_by])
    # Merge the statistics with the unique groups
    stats_df = pd.merge(unique_groups, stats_temp, on=group_by, how='left')
    
    # Save the processed data
    if output_path is None:
        output_path = csv_path
    
    if output_path:
        try:
            # Check if the output path already exists
            stats_df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")
        except PermissionError:
            raise PermissionError(f"Permission denied: Cannot write to file {output_path}")
        except Exception as e:
            raise Exception(f"Error saving CSV file: {str(e)}")
    
    return stats_df
