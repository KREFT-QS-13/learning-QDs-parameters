import numpy as np
import random
from scipy.special import comb
import matplotlib as mpl
mpl.use('Agg')  # Use the 'Agg' backend which is thread-safe
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
import torch

import os, time
import sys
import ast, argparse
import shutil
import re
import h5py
import json

sys.path.append('./qdarts')
from qdarts.experiment import Experiment
from qdarts.plotting import plot_polytopes

import traceback
import utilities.config as c

DPI = c.DPI
RESOLUTION = c.RESOLUTION

def transform_to_cartesian(r:float, theta:float) -> tuple[float, float]:
    """
        Transform polar coordinates to Cartesian coordinates.
    """
    return r*np.cos(theta), r*np.sin(theta)

def dist_between_points(i:tuple[int, int], j:tuple[int, int]) -> float:
    """
        Get the distance between two points.
    """
    return round(c.d_DD*np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2), 4)

def get_dots_indices(device:np.ndarray) -> list[tuple[int, int]]:
    """
        Get the indices of the dots in the device.
    """
    device = np.asarray(device)

    if device.ndim != 2:
        raise ValueError(f"Device array must be 2D, got shape {device.shape}, type {type(device)},\n{device}")
    
    indices = np.where(device == 1)
    return list(zip(indices[0], indices[1]))

def get_dots_coordinates(device:np.ndarray) -> list[tuple[int, int]]:
    """
        Get the coordinates of the dots in the device.
    """
    device = np.asarray(device)
    
    if device.ndim != 2:
        raise ValueError(f"Device array must be 2D, got shape {device.shape}")
    
    indices = np.where(device == 1)    
    return list(zip(indices[1], -indices[0]))  # Note: x = col, y = -row

def set_dots_number_based_on_device(device:np.ndarray, S:int) -> int:
    """
        Set the number of dots based on the device size and the number of sensors S.
    """
    c.set_global_K(len(get_dots_indices(device)) + S)

def get_centroid_of_device(device:np.ndarray) -> tuple[float, float]:
    """
        Get the centroid of the device as a point in the Cartesian coordinate system.

    """ 
    x_c = np.mean(get_dots_indices(device), axis=0)[1]
    y_c = -np.mean(get_dots_indices(device), axis=0)[0]
    return (x_c, y_c), np.sqrt(x_c**2 + y_c**2)

def check_sensor_in_correct_region(r0:float, theta0:float, device:np.ndarray, r_min:float=c.r_min, r_max:float=c.r_max) -> bool:
    centroid, _ = get_centroid_of_device(device)
    x_c, y_c = centroid
    s_x, s_y = transform_to_cartesian(r0, theta0)

    return r_min < np.sqrt((s_x-x_c)**2 + (s_y-y_c)**2) <= r_max

def draw_random(S:int, device:np.ndarray, const_sensor_r:bool) -> list[float]:
    """
        Draw random r0 for S sensors.
    """
    nx, ny = device.shape
    r_min = c.r_min  # or 0.5*np.sqrt(nx**2 + ny**2)*c.d_DD * 1.5
    r_max = c.r_max

    list_r0 = []
    list_theta0 = []
    while len(list_r0) < S:
        r0 = np.random.uniform(0, r_max)
        theta0 = np.random.uniform(0, 2*np.pi) 
        if check_sensor_in_correct_region(r0, theta0, device, r_min=r_min):
            list_r0.append(r0)
            list_theta0.append(theta0)

    if const_sensor_r:
       list_r0 = [list_r0[0]]*S

    return list_r0, list_theta0

def set_sensors_positions(S:int, device:np.ndarray, const_sensor_r=False, list_r0:list[float]=None, list_theta0:list[float]=None) -> tuple[float, float]:
    """
        Return the positions of the sensors in polar coordinates.
    """
    if list_r0 is None and list_theta0 is None:
        list_r0, list_theta0 = draw_random(S, device, const_sensor_r)
    elif list_r0 is None and list_theta0 is not None:
        list_r0, _ = draw_random(S, device, const_sensor_r)
    elif list_r0 is not None and list_theta0 is None:
        _, list_theta0 = draw_random(S, device, const_sensor_r)

    return [(r0,t0)  for r0, t0 in zip(list_r0, list_theta0)]

def get_dist_dot_sensor(r0:float, theta0:float, nx:int, ny:int) -> tuple[float, float]:
    """
        Return the distance between a dot and a sensor
    """
    theta_0i = np.arctan(ny/nx) if nx != 0 else np.pi/2
    r_0i = c.d_DD*np.sqrt(nx**2 + ny**2)
    theta_si = theta_0i + theta0
    
    return np.sqrt( r0**2 + r_0i**2 - 2*r0*r_0i*np.cos(theta_si))

def get_device_distance_matrix(device:np.ndarray, sensors:list[tuple[float, float]], config_tuple:tuple[int, int, int]) -> np.ndarray:
    """
        Get the distance matrix for the device.
    """
    K, N, S = config_tuple
    dist_matrix = np.zeros((K,K))
    sensor_corr = [transform_to_cartesian(r,t) for r,t in sensors]
    sensor_corr = [(x/c.d_DD, y/c.d_DD)  for x,y in sensor_corr]
    
    system_corr = get_dots_coordinates(device) + sensor_corr

    for i in range(len(system_corr)):
        for j in range(i, len(system_corr)):
            dist_matrix[i,j] = dist_between_points(system_corr[i], system_corr[j])
            dist_matrix[j,i] = dist_matrix[i,j]
    
    return dist_matrix

def exp_decay_model(dist_matrix:np.ndarray, config_tuple:tuple[int, int, int], mean:float=1.0, std:float=0.15, sensor_self_capacitance_coeff:float=5.0) -> np.ndarray:
    """
        Exponential decay model for the capacitance matrix.
    """
    K, N, S = config_tuple
    
    decay = lambda x,p: p**x
    
    # mag_conts = np.random.choice(c.mag_list, size=K)
    mag_const = np.random.choice(c.mag_list)
    mag_consts = np.abs(np.random.normal(loc=mag_const, scale=1, size=K))
    
    C_dd_prime, C_dg = np.identity(K), np.identity(K)
    C_dd_prime[np.eye(C_dd_prime.shape[0], dtype=bool)] = [round(np.random.normal(c*mean, c*std), 4) for c in mag_consts]
    C_dg[np.eye(C_dg.shape[0], dtype=bool)] = [round(np.random.normal(c*mean, c*std), 4) for c in mag_consts]

    for i in range(K):
        for j in range(i+1, K):
            C_dd_prime[i,j] = C_dd_prime[j,i] = round(np.sqrt(C_dd_prime[i,i]*C_dd_prime[j,j]) * decay(dist_matrix[i,j]/c.d_DD, c.p_dd), 4)
            if i < N and j < N:
                C_dg[i,j] = C_dg[j,i] = round(np.sqrt(C_dd_prime[i,i]*C_dd_prime[j,j])*decay(dist_matrix[i,j]/c.d_DG, c.p_dg), 4)
            else:
                C_dg[i,j] = C_dg[j,i] = 0 # No cross-talk between the sensor dot and the target dot.

    mask = np.eye(C_dd_prime.shape[0], dtype=bool)
    mask[:-S, :-S]  = False
    C_dd_prime[mask] *= sensor_self_capacitance_coeff

    return C_dd_prime, C_dg


def generate_capacitance_matrices(config_tuple:tuple[int, int, int]=None, device:np.ndarray=None, const_sensor_r:bool=False,
                                  sensors_radius:list[float]=None, sensors_angle:list[float]=None) -> tuple[np.ndarray, np.ndarray]:
    """
        Generate random capacitance matrices for a given number of dots K from a normal distribution.
        
        The diagonal elements of C_DD are drawn from a normal distribution 
        with a mean of 2*mean and a standard deviation of 10% of the mean.

        The off-diagonal elements of C_DD and C_DG are drawn from a normal distribution 
        with a mean and standard deviation of 10% of mean.
    """
    K, N, S = config_tuple
    mean = 1.0 #aF
    std = 0.1
    C_DG = np.random.normal(mean, std, (K,K))

    if S == 0:
        for i in range(K):
            diag_const = np.random.choice(c.mag_list)

            C_DG[i,i] = np.random.normal(diag_const*mean, diag_const*std)
        
            C_m = np.zeros((K, K)) 
            mask = ~np.eye(C_m.shape[0], dtype=bool)
            C_m[mask] = np.random.normal(mean, std, K*(K-1))

            C_m = (C_m + C_m.T)/2

        C_DD = np.sum(C_DG, axis=1).T*np.eye(K) + C_m
        return C_DD, C_DG, None
    elif S>0 and device is not None:
        sensors = set_sensors_positions(S, device, const_sensor_r, sensors_radius, sensors_angle)
        dist_matrix = get_device_distance_matrix(device, sensors, config_tuple)

        C_DD, C_DG = exp_decay_model(dist_matrix, config_tuple, mean, std)
        
        C_DD = C_DD + np.sum(C_DG, axis=1)*np.eye(K) + (np.sum(C_DD, axis=1)-np.diag(C_DD))*np.eye(K) 

        return C_DD, C_DG, sensors
    else:
        raise ValueError("Device is not provided! For noise generation you need to provide the device!")

def generate_dummy_data(K:int) -> tuple[np.ndarray, np.ndarray]:
    """
        Generate dummy (identity matrix) capacitance matrices for a given number of dots K.
    """
    return np.identity(K), np.identity(K)

def get_cut(config_tuple):
    """Generate a 2d cut constructed from standard basis vectors."""
    K, N, S = config_tuple
        
    cut = np.zeros((2, K), dtype=int)
    indices = np.random.choice(np.arange(N), 2, replace=False)
    cut[tuple(zip(*enumerate(indices)))] = 1
    return cut[np.argmax(cut, axis=1).argsort()]

def get_all_euclidean_cuts(config_tuple):
    K, N, _ = config_tuple  
    num_of_cuts = int(comb(N,2))
    cuts = np.zeros((num_of_cuts, 2, K), dtype=int)

    k=0
    for i in range(N):
        for j in range(i+1,N):
            cuts[k][0][i] = 1
            cuts[k][1][j] = 1
            k+=1

    return cuts 

def plot_CSD(x: np.ndarray, y: np.ndarray, csd_or_sensor: np.ndarray, polytopesks: list[np.ndarray], 
             only_edges:bool=False, only_labels:bool=True, res:int=RESOLUTION, dpi:int=DPI):
    """
    Plot the charge stability diagram (CSD).
    
    Args:
        x (np.ndarray): x-axis values
        y (np.ndarray): y-axis values
        csd_or_sensor (np.ndarray): 2D or 3D array of CSD/sensor data
        polytopesks (list[np.ndarray]): List of polytopes
        only_edges (bool): Whether to plot only edges of polytopes
        only_labels (bool): Whether to plot only labels
        res (int): Resolution of the plot
        dpi (int): DPI of the plot
    
    Returns:
        tuple: (figure, axis) if 2D input
               list of (figure, axis) if 3D input
    """

    # if len(csd_or_sensor.shape) > 3:
    # Handle 3D array (multiple channels)
    figures_and_axes = []
    for cut in range(csd_or_sensor.shape[0]):
        for channel in range(csd_or_sensor.shape[-1]):
            plt.figure(figsize=(res/dpi, res/dpi), dpi=dpi)
            ax = plt.gca()
        
            ax.pcolormesh(1e3*x, 1e3*y, csd_or_sensor[cut,:,:,channel].squeeze())
                
            # print(polytopesks, len(polytopesks))
            plot_polytopes(ax, polytopesks[cut], axes_rescale=1e3, 
                            only_edges=only_edges, only_labels=only_labels)
                
            ax.set_xlim(x[0]*1e3, x[-1]*1e3)
            ax.set_ylim(y[0]*1e3, y[-1]*1e3)
            ax.axis('off')
            plt.tight_layout(pad=0)
                
            figures_and_axes.append((plt.gcf(), ax))
            
    return figures_and_axes
    # else:
    #     # Original behavior for 2D array
    #     plt.figure(figsize=(res/dpi, res/dpi), dpi=dpi)
    #     ax = plt.gca()

    #     ax.pcolormesh(1e3*x, 1e3*y, csd_or_sensor)
    #     plot_polytopes(ax, polytopesks, axes_rescale=1e3, 
    #                   only_edges=only_edges, only_labels=only_labels)

    #     ax.set_xlim(x[0]*1e3, x[-1]*1e3)
    #     ax.set_ylim(y[0]*1e3, y[-1]*1e3)
    #     ax.axis('off')
    #     plt.tight_layout(pad=0)

    #     return plt.gcf(), ax

def get_mask(device: np.ndarray, config_tuple: tuple[int, int, int]) -> np.ndarray:
    """
    Create a mask for nearest neighbors (horizontal and vertical only) in the device.
    
    Args:
        device (np.ndarray): Binary array where 1s represent dots
        config_tuple (tuple[int, int, int]): Tuple of (K, N, S) values
    
    Returns:
        np.ndarray: Boolean array of shape (K, K) where True indicates nearest neighbors
    """
    K, N, S = config_tuple
    mask = np.zeros((K, K), dtype=bool)
    
    # Get dot indices
    dot_indices = get_dots_indices(device)
    
    # Check each pair of dots
    for i, (row1, col1) in enumerate(dot_indices):
        for j, (row2, col2) in enumerate(dot_indices):
            if i != j:  # Don't compare dot with itself
                # Check if dots are adjacent horizontally or vertically
                is_neighbor = (
                    (abs(row1 - row2) == 1 and col1 == col2) or  # Vertical neighbor
                    (abs(col1 - col2) == 1 and row1 == row2)     # Horizontal neighbor
                )
                if is_neighbor:
                    mask[i, j] = True
                    mask[j, i] = True  # Make it symmetric
    
    return mask


def generate_experiment_config(C_DD:np.ndarray, C_DG:np.ndarray, config_tuple:tuple[int, int, int], device:np.ndarray):
    K, N, S = config_tuple
    tunnel_couplings = np.zeros((K,K))
    #TODO: Create a new mask that for each dot set c.tunnel_coupling_const to its nearest neighbors
    mask = get_mask(device, config_tuple)
    tunnel_couplings[mask] = c.tunnel_coupling_const 

    capacitance_config = {
        "C_DD" : C_DD,  #dot-dot capacitance matrix
        "C_Dg" : C_DG,  #dot-gate capacitance matrix
        "ks" : None,       #distortion of Coulomb peaks. NOTE: If None -> constant size of Coublomb peak 
    }

    tunneling_config = {
        "tunnel_couplings": tunnel_couplings, #tunnel coupling matrix
        "temperature": 0.1,                   #temperature in Kelvin
        "energy_range_factor": 5,  #energy scale for the Hamiltonian generation. NOTE: Smaller -> faster but less accurate computation 
    }

    sensor_config = {
        "sensor_dot_indices": np.arange(N,K).tolist(), #TODO: Fix it -np.arange(1,S+1),  #Indices of the sensor dots
        "sensor_detunings": [0.0005]*S,  #Detuning of the sensor dots , -0.02
        "noise_amplitude": {"fast_noise":c.fast_noise_amplitude, "slow_noise": c.slow_noise_amplitude}, #Noise amplitude for the sensor dots in eV
        "peak_width_multiplier": 30, #40 , 25  #Width of the sensor peaks in the units of thermal broadening m *kB*T/0.61.
    }


    return capacitance_config, tunneling_config, sensor_config

def generate_dataset(x_vol: np.ndarray, y_vol: np.ndarray, ks:int=0, device:np.ndarray=None, 
                     config_tuple:tuple[int,int,int]=None, sensors_radius:list[float]=None, 
                     sensors_angle:list[float]=None, const_sensor_r=False, cut:np.ndarray=None,
                     all_euclidean_cuts:bool=False):
    """
        Run the QDarts experiment.
    """

    print(f"Generating dataset for {config_tuple} configuration")
    if config_tuple is None:
        raise ValueError("config_tuple must be provided")
        
    K, N, S = config_tuple
    c.validate_state(K, N, S)
    
    if S > 0 and device is None:
        raise ValueError("Device must be provided when using sensors (S > 0)")
        
    try:
        C_DD, C_DG, sensors_coordinates = generate_capacitance_matrices(config_tuple, device, const_sensor_r, sensors_radius, sensors_angle)
        
        if cut is None:
            try:
                if all_euclidean_cuts:
                    cuts = get_all_euclidean_cuts(config_tuple)
                else:
                    cuts = get_cut(config_tuple)
                    cuts = cuts.reshape(1, *cuts.shape)
            except ValueError as e:
                print(f"Error generating cut(s): {e}")
                return None
            
        use_sensor_signal = S > 0
        
        all_csd_data = []
        all_polytopes = []
        all_sensor_data = []
        
        for cut_idx in range(cuts.shape[0]):
            current_cut = cuts[cut_idx] if cuts.shape[0] > 1 else cuts[0]
            target_state = np.zeros(K, dtype=int) + 1
            target_state[-S:] = 5
            target_transition = current_cut[0] - current_cut[1]
            
            try:
                if S == 0:
                    capacitance_config, _, _ = generate_experiment_config(C_DD, C_DG, config_tuple, device)
                    experiment = Experiment(capacitance_config)
                    result = experiment.generate_CSD(
                        x_voltages=x_vol,
                        y_voltages=y_vol,
                        plane_axes=current_cut,
                    )
                else:
                    experiment = Experiment(*generate_experiment_config(C_DD, C_DG, config_tuple, device))
                    result = experiment.generate_CSD(
                        x_voltages=x_vol,
                        y_voltages=y_vol,
                        target_state=target_state,
                        target_transition=target_transition,
                        plane_axes=current_cut,
                        compute_polytopes=True,
                        use_sensor_signal=use_sensor_signal,
                    )
                
                if result is None:
                    print(f"generate_CSD returned None for cut {cut_idx}")
                    continue
                    
                xks, yks, csd_dataks, polytopesks, sensor, _ = result
                
                all_csd_data.append(csd_dataks)
                all_polytopes.append(polytopesks)
                all_sensor_data.append(sensor if sensor is not None else None)
                
            except Exception as e:
                print(f"Error in experiment generation for cut {cut_idx}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not all_csd_data:  # If no successful generations
            return None
            
        # Convert lists to arrays
        all_csd_data = np.array(all_csd_data)
        all_polytopes = np.array(all_polytopes)
        all_sensor_data = np.array(all_sensor_data) if all_sensor_data[0] is not None else None
        
        if S > 0:
            return C_DD, C_DG, ks, cuts, xks, yks, all_csd_data, all_polytopes, all_sensor_data, device, sensors_coordinates
        else:
            return C_DD, C_DG, ks, cuts, xks, yks, all_csd_data, all_polytopes, None, device, sensors_coordinates
            
    except Exception as e:
        print(f"Error in capacitance matrix generation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None 

def ensure_path(path):
    """
    Ensure that the directory path exists, create it if it doesn't.
    
    Args:
        path (str): Path to ensure exists
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    return path

def count_directories_in_folder(config_tuple):
    """Count the number of batch directories in a given folder."""
    K, N, S = config_tuple
    path = c.get_path(K, N, S)
    
    # # Ensure the path exists before trying to list directories
    # ensure_path(path)
    
    # Now safely list directories
    batch_list = [x for x in os.listdir(path) if re.compile(r"batch-\d").match(x)] 
    return sum(os.path.isdir(os.path.join(path, x)) for x in batch_list)

def create_paths(config_tuple, path=None):
    """Creates paths for datapoints and images."""
    K, N, S = config_tuple
    if path is None:
        path = c.get_path(K, N, S)
        
    global PATH_IMG
    global PATH_DPS
 
    # Ensure base path exists
    ensure_path(path)
    
    batch_name = 'batch-' + str(count_directories_in_folder(config_tuple)+1)
    full_path = os.path.join(path, batch_name)
    
    # Create batch directory
    ensure_path(full_path)
    
    full_path_dps = full_path
    full_path_img = os.path.join(full_path, 'imgs')
    
    # Create images directory
    ensure_path(full_path_img)

    PATH_IMG = full_path_img
    PATH_DPS = full_path_dps

def clean_batch():
    # if not os.path.isfile(PATH_IMG):
    #     os.rmdir(PATH_IMG)
    # else:
    #     for f in os.listdir(PATH_IMG):
    #         os.remove(os.path.join(PATH_IMG, f))
        
    #     os.rmdir(PATH_IMG)

    if not os.path.isfile(PATH_DPS):
        try:
            shutil.rmtree(PATH_DPS)
        except Exception as e:
            print("Unable to clean empty batch folder!")
            print(f'{e}')

def save_img_csd(config_tuple, csd_plot, cut):
    """
    Save the CSD image as a PNG file with a unique name.
    
    Args:
        config_tuple (tuple): (K, N, S) configuration
        csd_plot: Either a single figure or list of figures
        cut (np.ndarray): Cut array used to generate the CSD
    
    Returns:
        tuple: (unique_id, list of (path, name) for saved images)
    """
    K, N, S = config_tuple
    
    # Generate base name
    base_name = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    
    # Create directory for this group
    group_dir = os.path.join(PATH_IMG, base_name)
    ensure_path(group_dir)

    saved_files = []

    # Handle multiple cuts
    for cut_idx in range(len(cut)):
        # Add cut indices to name
        indices = [np.argwhere(c == 1).squeeze().tolist() for c in cut[cut_idx]]
        cut_name = '_'+''.join(str(i) for i in indices)
        
        if S > 0 and len(csd_plot.shape) > 3:  # Multiple sensors
            for sensor_idx in range(csd_plot.shape[-1]):
                img_name = f"{base_name}{cut_name}_s{sensor_idx}.png"
                full_path_img = os.path.join(group_dir, img_name)
                
                fig = plt.figure(figsize=(c.RESOLUTION/c.DPI, c.RESOLUTION/c.DPI), dpi=c.DPI)
                plt.imshow(csd_plot[cut_idx, :, :, sensor_idx])
                plt.axis('off')
                plt.tight_layout(pad=0)
                
                plt.savefig(full_path_img, format='png', bbox_inches='tight', 
                           pad_inches=0, dpi=c.DPI)
                plt.close(fig)
                
                saved_files.append((full_path_img, img_name))
        else:  # Single sensor or no sensors
            img_name = f"{base_name}{cut_name}.png"
            full_path_img = os.path.join(group_dir, img_name)
            
            fig = plt.figure(figsize=(c.RESOLUTION/c.DPI, c.RESOLUTION/c.DPI), dpi=c.DPI)
            plt.imshow(csd_plot[cut_idx])
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            plt.savefig(full_path_img, format='png', bbox_inches='tight', 
                       pad_inches=0, dpi=c.DPI)
            plt.close(fig)
            
            saved_files.append((full_path_img, img_name))
    
    return base_name, saved_files

# TODO: Figure out how to save the data in multiple files after 500 datapoints generation
#     - thats for safety 
#     - also start thinking how you will orgenize for more datapoints with bigger K
#     - for bigger K -> 10 000 datapoints -> \biom{K}{2} * 10 0000 real datapoints ?
#     - how about the philosophy: you generate to learn (?)
def save_to_json(dictionary: dict):
    """
        Save the datapoints to a json file.
        # TODO: It only saves the last img -> fix it
    """      
    full_path_dps = os.path.join(PATH_DPS, 'datapoints.json')
    
    try:
        if os.path.exists(full_path_dps):
            with open(full_path_dps, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data.update(dictionary)

        with open(full_path_dps, 'w') as f:
            json.dump(data, f)

    except Exception as e:
        print(f"Json file no longer updated!")
        print(f"Error: {e}")


def save_to_hfd5(dictionary: dict):
    """
    Save the datapoints to an HDF5 file.
    """
    full_path_dps = os.path.join(PATH_DPS, 'datapoints.h5')

    with h5py.File(full_path_dps, 'a') as f:
        for key, value in dictionary.items():
            if key in f:
                del f[key]  # Delete existing group if it exists
            
            group = f.create_group(key)
            cuts_group = group.create_group('cuts')

            # Save metadata
            for meta_key in ['K', 'N', 'S', 'tunnel_coupling_const', 'slow_noise_amplitude', 'fast_noise_amplitude',
                            'p_dd', 'p_dg', 'd_DD', 'd_DG', 'r_min', 'r_max', 'base_name']:
                if meta_key in value:
                    group.attrs[meta_key] = value[meta_key]
            
            # Save matrices
            for matrix_key in ['C_DD', 'C_DG', 'device', 'sensors_coordinates']:
                if matrix_key in value:
                    group.create_dataset(matrix_key, data=value[matrix_key])
            
            # Save voltage ranges
            for vol_key in ['x_vol', 'y_vol']:
                if vol_key in value:
                    group.create_dataset(vol_key, data=value[vol_key])
            
            # Save cuts and related data
            for cut_idx in range(len(value['cuts'])):
                cut_group = cuts_group.create_group(f'cut_{cut_idx}')
                cut_group.create_dataset('cut', data=value['cuts'][cut_idx])
                cut_group.create_dataset('csd', data=value['csd'][cut_idx])
                cut_group.create_dataset('poly', data=value['poly'][cut_idx])
                cut_group.create_dataset('sensor_output', data=value['sensor_output'][cut_idx])
                if value.get('csd_gradient') is not None:
                    cut_group.create_dataset('csd_gradient', data=value['csd_gradient'][cut_idx])

def get_batch_folder_name(batch_num: int, config_tuple: tuple[int, int, int]):
    """
        Get the batch folder name.
    """
    K, N, S = config_tuple
    if batch_num <= count_directories_in_folder(config_tuple):
        return 'batch-' + str(batch_num)
    else:
        print(ValueError(f"Batch number is too high! Max: {count_directories_in_folder(config_tuple)}!"))
        return None

def get_path_hfd5(batch_num: int, config_tuple: tuple[int, int, int], v: bool=False):
    """
        Get the path to the hfd5 file.
        For now it is for testing and not yet finished.
    """
    K, N, S = config_tuple
    batch_name = get_batch_folder_name(batch_num, config_tuple)
    path = c.get_path(K, N, S)
    full_path_dps = os.path.join(path, batch_name, 'datapoints.h5')
          
    return full_path_dps

def check_and_correct_img_name(img_name: str):
    if not re.compile(r"^\d+\.png$").match(img_name):
        return img_name + ".png"
    else:
        return img_name

def load_csd_img(batch_num: int, csd_name: str, config_tuple: tuple[int, int, int], show: bool=False):
    """
        Load the PNG file 
    """
    K, N, S = config_tuple
    csd_name = check_and_correct_img_name(csd_name)
    path = c.get_path(K, N, S)
    path = os.path.join(path, get_batch_folder_name(batch_num, config_tuple), 'imgs', csd_name)
    
    img = Image.open(path)
    if show:
        img.show() 
    
    return img 

def reconstruct_img_from_tensor(tensor:np.ndarray):
    return Image.fromarray((tensor.transpose(1, 2, 0) * 255).astype(np.uint8))
    # return Image.fromarray((tensor.transpose(1, 2, 0)))

def reconstruct_img_with_matrices(batch_num: int, img_name: str, config_tuple: tuple[int, int, int], show: bool=False):
    """
    Reconstruct image and get associated matrices from HDF5 file.
    """
    img_name = check_and_correct_img_name(img_name)
    path = get_path_hfd5(batch_num, config_tuple)
    print(f"The file path: {path}")

    with h5py.File(path, 'r') as f:
        img = f[img_name]['csd'][:]
        C_DD = f[img_name]['C_DD'][:]
        C_DG = f[img_name]['C_DG'][:]

        img = Image.fromarray((img.transpose(1, 2, 0)))
        plt.axis('off')

        if show:
            img.show()
            print(img)
            
            print("C_DD matrix:")
            print(C_DD)
            print("\nC_DG matrix:")
            print(C_DG)
        
        return img, C_DD, C_DG
    
def save_datapoints(config_tuple, C_DD, C_DG, ks, x_vol, y_vol, cuts, poly, csd, sensor_output, csd_plot, csd_gradient, device, sensors_coordinates):
    """
    Combine all 'saving' functions to create a datapoint.
    """
    K, N, S = config_tuple
   
    # save img of CSD 
    base_name, saved_files = save_img_csd(config_tuple, csd_plot, cuts)
   
    # Create tensors for each cut
    csd_tensors = []
    gradient_tensors = []

    for (fpi, img_name) in saved_files:
        # save datapoints
        csd = Image.open(fpi)
        csd_array = np.array(csd)
        
        # Handle grayscale images (2D) vs RGB images (3D)
        if len(csd_array.shape) == 2:
            # If grayscale, add channel dimension
            csd_tensor = torch.tensor(csd_array[None, :, :])
        else:
            # If RGB/RGBA, permute dimensions to [C, H, W]
            csd_tensor = torch.tensor(csd_array).permute(2, 0, 1)
        csd_tensors.append(csd_tensor)
        
        # Handle gradient data similarly
        if len(csd_gradient.shape) == 2:
            csd_gradient_tensor = torch.tensor(csd_gradient[None, :, :])
        else:
            csd_gradient_tensor = torch.tensor(csd_gradient).permute(2, 0, 1)
        gradient_tensors.append(csd_gradient_tensor)

    ks = np.nan if ks is None else ks
    datapoints_dict = {img_name: {
        'K': K, 
        'N': N,
        'S': S,
        'tunnel_coupling_const': c.tunnel_coupling_const,
        'slow_noise_amplitude': c.slow_noise_amplitude,
        'fast_noise_amplitude': c.fast_noise_amplitude,
        'C_DD': C_DD, 
        'C_DG': C_DG, 
        'ks': ks,
        'x_vol': np.array(x_vol), 
        'y_vol': np.array(y_vol), 
        'cuts': np.array(cuts), 
        'poly': poly,
        'sensor_output': sensor_output,
        'csd': csd_tensor,
        'csd_gradient': csd_gradient_tensor,
        'device': device,
        'sensors_coordinates': sensors_coordinates,
        'p_dd': c.p_dd,
        'p_dg': c.p_dg,
        'd_DD': c.d_DD,
        'd_DG': c.d_DG,
        'r_min': c.r_min,
        'r_max': c.r_max,
        'base_name': base_name,
    }} # 23 elements
    
    save_to_hfd5(datapoints_dict)


def generate_datapoint(args):
    x_vol, y_vol, ks, device, i, N_batch, config_tuple, sensors_radius, sensors_angle, const_sensors_radius, all_euclidean_cuts, cut = args
    K, N, S = config_tuple
    print(f"Generating datapoint {i+1}/{N_batch}:")
    print(f"Configuration: K={K}, N={N}, S={S}")

    try:
        # Create unique seed
        process_id = os.getpid()
        current_time = int(time.time() * 1000)
        unique_seed = (process_id + current_time + i) % (2**32 - 1)
        
        # Set the seed for numpy and random
        np.random.seed(unique_seed)
        random.seed(unique_seed)
        
        result = generate_dataset(x_vol, y_vol, ks, device, config_tuple, sensors_radius, sensors_angle, const_sensors_radius, cut, all_euclidean_cuts)
        if result is None:
            return None
            
        C_DD, C_DG, ks, cut, x, y, csd, poly, sensor, device, sensors_coordinates = result
        
        if S == 0:
            fig, _ = plot_CSD(x, y, csd, poly, only_labels=False)
            gradient = np.gradient(csd,axis=0)+np.gradient(csd,axis=1)
            return (C_DD, C_DG, ks, cut, x_vol, y_vol, csd, poly, sensor, fig, gradient, device, sensors_coordinates)
        elif S == 1:
            fig, _ = plot_CSD(x, y, sensor, poly)
            gradient = np.gradient(sensor,axis=0)+np.gradient(sensor,axis=1)
            return (C_DD, C_DG, ks, cut, x_vol, y_vol, csd, poly, sensor, fig, gradient, device, sensors_coordinates)
        elif S > 1:
            figs = [fig for fig, _ in plot_CSD(x, y, sensor, poly)]
            gradients = [np.gradient(s,axis=0)+np.gradient(s,axis=1) for s in sensor]
            return (C_DD, C_DG, ks, cut, x_vol, y_vol, csd, poly, sensor, figs, gradients, device, sensors_coordinates)

    except Exception as e:
        print(f"Execution failed for datapoint {i+1}!")
        print(f"Error: {e}")
        # print("\nFull traceback:")
        # traceback.print_exc(file=sys.stdout)
        print(f"Traceback: {traceback.format_exc()}")
        return None

def ensure_dir_exists(path):
    """
    Check if the directory exists, and if not, create it.
    
    Args:
        path (str): The directory path to check/create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def plot_device_lattice(device: np.ndarray, sensors: list[tuple[int, int]], figsize=(4, 4), dot_size=25, show_grid=True):
    """
    Plot dots in a lattice according to the device configuration.
    
    Args:
        device (np.ndarray): Binary array where 1s represent dots
        figsize (tuple): Figure size in inches (default: (4, 4))
        dot_size (int): Size of the dots in the plot (default: 100)
        show_grid (bool): Whether to show the grid lines (default: True)
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Get dot coordinates
    dot_coords = get_dots_coordinates(device)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot dots
    x_coords = [coord[0] for coord in dot_coords]
    y_coords = [coord[1] for coord in dot_coords]
    
    # Plot dots with numbers
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.scatter(x, y, s=dot_size, c='black',  marker='o', zorder=2)
        # ax.text(x, y, str(i), color='black', ha='center', va='center', fontweight='bold')
    
    for i,s in enumerate(sensors):
        x, y = transform_to_cartesian(s[0], s[1])
        x_coords.append(x/c.d_DD)
        y_coords.append(y/c.d_DD)
        # ax.scatter(x/c.d_DD, y/c.d_DD, s=dot_size, c='red', marker='*', zorder=2)
        ax.text(x/c.d_DD, y/c.d_DD, 's'+str(i), color='red', ha='center', va='center', fontsize=12)


    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Set grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7, zorder=1)
    
    # Set limits with some padding
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Device Dot Configuration')
    
    plt.tight_layout()
    return fig, ax

def load_parameters(batch_num: int, img_name: str, config_tuple: tuple[int, int, int], 
                    param_names: list[str], print_available_params: bool=False) -> dict:
    """
    Load specific parameters from HDF5 file for a given image.
    
    Args:
        batch_num (int): Batch number
        img_name (str): Name of the image file
        config_tuple (tuple[int, int, int]): Tuple of (K, N, S)
        param_names (list[str]): List of parameter names to load
        print_available_params (bool): Whether to print available parameters (default: False)
    Returns:
        dict: Dictionary containing requested parameters
    """
    img_name = check_and_correct_img_name(img_name)
    path = get_path_hfd5(batch_num, config_tuple)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"HDF5 file not found at: {path}")
    
    result = {}
    try:
        with h5py.File(path, 'r') as f:
            if img_name not in f:
                raise KeyError(f"Image {img_name} not found in HDF5 file")
                
            group = f[img_name]
            
            # Get list of all available parameters (datasets and attributes)
            available_datasets = list(group.keys())
            available_attrs = list(group.attrs.keys())
            
            if print_available_params:
                print(f"\nAvailable parameters for {img_name}:")
                print(f"Datasets: {available_datasets}")
                print(f"Attributes: {available_attrs}\n")
            
            for param in param_names:
                # Check if parameter is an attribute
                if param in group.attrs:
                    result[param] = group.attrs[param]
                    continue
                    
                # Check if parameter is a dataset
                if param in group:
                    result[param] = group[param][:]
                    continue
                    
                # Parameter not found
                print(f"Warning: Parameter '{param}' not found for image {img_name}")
                result[param] = None
                    
    except Exception as e:
        print(f"Error loading parameters: {e}")
        print(f"\nAvailable parameters for {img_name}:")
        print(f"Datasets: {available_datasets}")
        print(f"Attributes: {available_attrs}\n")
        print(f"Traceback: {traceback.format_exc()}")
        return None
        
    return result

def parse_array(string):
    """
    Parse string representation of array to numpy array.
    Examples:
        "[[1,1], [1,1]]" -> np.array([[1,1], [1,1]])
        "[1,1,1]" -> np.array([1,1,1])
    """
    try:
        # Convert string to Python list
        array_list = ast.literal_eval(string)
        # Convert to numpy array
        return np.array(array_list)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Not a valid array: {string}") from e