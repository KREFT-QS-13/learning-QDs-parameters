import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import src.utilities.utils as u

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

def plot_prediction_vs_true_matrix_elements(targets, outputs, save_dir, 
                                             figsize_per_subplot=(6, 6), 
                                             alpha=0.6, s=20, 
                                             plotname:str='prediction_vs_true_matrix_elements',
                                             add_diagonal=True,
                                             show_metrics=True):
    """
    Create element-wise scatter plots of predicted vs true values for each matrix element.
    Assumes the output represents a square matrix (e.g., capacitance matrix).
    For each matrix element (i,j), plots all predictions vs true values across all samples.
    
    Args:
        targets (np.array): True values, shape (N, output_dim) where output_dim = num_dots^2
        outputs (np.array): Predicted values, shape (N, output_dim)
        save_dir (str): Directory to save the plot (None to display)
        figsize_per_subplot (tuple): Figure size per subplot (width, height)
        alpha (float): Transparency of scatter points (0-1)
        s (float): Size of scatter points
        plotname (str): Name for the saved plot file
        add_diagonal (bool): Whether to add diagonal reference line (y=x)
        show_metrics (bool): Whether to show metrics in title (Pearson, R², MSE, MAE)
    
    Returns:
        None
    """
    # Ensure inputs are numpy arrays
    targets = np.array(targets).squeeze()
    outputs = np.array(outputs).squeeze()
    
    # Validate shapes
    if targets.shape != outputs.shape:
        raise ValueError(f"Shapes must match: targets {targets.shape} vs outputs {outputs.shape}")
    
    if len(targets.shape) != 2:
        raise ValueError(f"Expected 2D arrays (N, output_dim), got shape {targets.shape}")
    
    num_samples, output_dim = targets.shape
    
    # Calculate grid dimensions, always square matrix
    num_dots = int(np.sqrt(output_dim))
    if num_dots * num_dots != output_dim:
        raise ValueError(f"Output dimension {output_dim} is not a perfect square. "
                        f"Expected output_dim = num_dots^2 for a square matrix.")
    
    # Reshape to matrix form: (num_samples, num_dots, num_dots)
    targets_matrix = targets.reshape(num_samples, num_dots, num_dots)
    outputs_matrix = outputs.reshape(num_samples, num_dots, num_dots)
    
    # Create figure with appropriate size
    fig_width = figsize_per_subplot[0] * num_dots
    fig_height = figsize_per_subplot[1] * num_dots
    fig, axes = plt.subplots(num_dots, num_dots, figsize=(fig_width, fig_height))
    
    # Handle case where num_dots == 1
    if num_dots == 1:
        axes = np.array([[axes]])
    
    # Define better colors
    scatter_color = '#2E86AB'  # Nice blue color
    diagonal_color = '#D62828'  # Red for diagonal line
    edge_color = '#1B263B'  # Dark blue/black for edges
    
    # Plot each matrix element
    for i in range(num_dots):
        for j in range(num_dots):
            ax = axes[i, j]
            
            # Extract all predictions and true values for this matrix element
            true_vals = targets_matrix[:, i, j]
            pred_vals = outputs_matrix[:, i, j]
            
            # Create scatter plot
            ax.scatter(true_vals, pred_vals, alpha=alpha, s=s, 
                      c=scatter_color, edgecolors=edge_color, 
                      linewidth=0.5, zorder=2)
            
            # Calculate metrics
            pearson_corr, pearson_p = pearsonr(true_vals, pred_vals)
            
            if show_metrics:
                r2 = r2_score(true_vals, pred_vals)
                mse = mean_squared_error(true_vals, pred_vals)
                mae = mean_absolute_error(true_vals, pred_vals)
                
                # Create title with metrics
                title = f'Element ({i},{j})\n'
                title += f'Pearson: {pearson_corr:.3f} | R²: {r2:.3f}\n'
                title += f'MSE: {mse:.4f} | MAE: {mae:.4f}'
            else:
                title = f'Element ({i},{j})\nPearson: {pearson_corr:.3f}'
            
            # Increase font sizes for better readability
            ax.set_title(title, fontsize=12, fontweight='bold', pad=2)
            ax.set_xlabel('True Value', fontsize=11, fontweight='medium')
            ax.set_ylabel('Predicted Value', fontsize=11, fontweight='medium')
            
            # Increase tick label font size
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            # Add diagonal reference line (y = x) with better visibility
            if add_diagonal:
                min_val = min(true_vals.min(), pred_vals.min())
                max_val = max(true_vals.max(), pred_vals.max())
                ax.plot([min_val, max_val], [min_val, max_val], 
                       color=diagonal_color, linestyle='--', linewidth=2.5, 
                       alpha=0.85, label='Perfect prediction', zorder=1)
                ax.legend(fontsize=9, loc='upper left', framealpha=0.9, 
                         edgecolor='gray', fancybox=True)
            
            # Add grid with better visibility
            ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, 
                   color='gray', zorder=0)
            
            # Set equal aspect ratio for better visualization
            ax.set_aspect('equal', adjustable='box')
            
            # Add subtle background color for better contrast
            ax.set_facecolor('#FAFAFA')
    
    # Add overall title with larger font
    fig.suptitle(f'Predicted vs True Values for Each Matrix Element \n MSE={mean_squared_error(targets, outputs):.4f}, MAE={mean_absolute_error(targets, outputs):.4f}, R2={r2_score(targets, outputs):.4f}', 
                 fontsize=18, fontweight='bold', y=0.945)
    
    # Adjust spacing for better readability
    plt.tight_layout(rect=[0, 0, 1, 0.98], pad=2.0)
    
    # Save or display
    if save_dir is not None:
        u.ensure_dir_exists(save_dir)
        output_path = save_dir + f'/{plotname}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Matrix element-wise prediction plot saved to {output_path}")
    else:
        plt.show()


 