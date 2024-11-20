import numpy as np
import random
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

def draw_random(S:int, device:np.ndarray) -> list[float]:
    """
        Draw random r0 for S sensors.
    """
    nx, ny = device.shape
    r_min = 0.5*np.sqrt(nx**2 + ny**2)*c.d_DD * 1.5 # or c.r_min
    r_max = c.r_max

    list_r0 = []
    list_theta0 = []
    while len(list_r0) < S:
        r0 = np.random.uniform(0, r_max)
        theta0 = np.random.uniform(0, 2*np.pi)
        if check_sensor_in_correct_region(r0, theta0, device, r_min=r_min):
            list_r0.append(r0)
            list_theta0.append(theta0)

   
    return list_r0, list_theta0

def set_sensors_positions(S:int, device:np.ndarray, list_r0:list[float]=None, list_theta0:list[float]=None) -> tuple[float, float]:
    """
        Return the positions of the sensors in polar coordinates.
    """
    if list_r0 is None and list_theta0 is None:
        list_r0, list_theta0 = draw_random(S, device)

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

def exp_decay_model(dist_matrix:np.ndarray, config_tuple:tuple[int, int, int], mean:float=1.0, std:float=0.15) -> np.ndarray:
    """
        Exponential decay model for the capacitance matrix.
    """
    K, N, S = config_tuple
    
    decay = lambda x,p: p**x
    
    # mag_conts = np.random.choice(c.mag_list, size=K)
    mag_const = np.random.choice(c.mag_list)
    mag_consts = np.random.normal(loc=mag_const, scale=1.25, size=K)
    
    C_dd_prime, C_dg = np.identity(K), np.identity(K)
    C_dd_prime[np.eye(C_dd_prime.shape[0], dtype=bool)] = [round(np.random.normal(c*mean, c*std), 4) for c in mag_consts]
    C_dg[np.eye(C_dg.shape[0], dtype=bool)] = [round(np.random.normal(c*mean, c*std), 4) for c in mag_consts]


    for i in range(K):
        for j in range(i+1, K):
            C_dd_prime[i,j] = C_dd_prime[j,i] = round(np.sqrt(C_dd_prime[i,i]*C_dd_prime[j,j]) * decay(dist_matrix[i,j]/c.d_DD, c.p_dd), 4)
            C_dg[i,j] = C_dg[j,i] = round(np.sqrt(C_dd_prime[i,i]*C_dd_prime[j,j])*decay(dist_matrix[i,j]/c.d_DG, c.p_dg), 4)


    return C_dd_prime, C_dg


def generate_capacitance_matrices(config_tuple:tuple[int, int, int]=None, device:np.ndarray=None) -> tuple[np.ndarray, np.ndarray]:
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
        sensors = set_sensors_positions(S, device)
        dist_matrix = get_device_distance_matrix(device, sensors, config_tuple)

        C_DD, C_DG = exp_decay_model(dist_matrix, config_tuple, mean, std)
        
        C_DD = C_DD + np.sum(C_DG, axis=1).T*np.eye(K) + (np.sum(C_DD, axis=1)-np.diag(C_DD))*np.eye(K) 

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
        
    cut = np.zeros((2, K))
    indices = np.random.choice(np.arange(N), 2, replace=False)
    cut[tuple(zip(*enumerate(indices)))] = 1
    return cut[np.argmax(cut, axis=1).argsort()]

def plot_CSD(x: np.ndarray, y: np.ndarray, csd_or_sensor: np.ndarray, polytopesks: list[np.ndarray], only_edges:bool=True, only_labels:bool=True, res:int=RESOLUTION, dpi:int=DPI):
    """
        Plot the charge stability diagram (CSD) (res by res, default 256 by 256).
    """
    plt.figure(figsize=(res/dpi, res/dpi), dpi=dpi)
    ax = plt.gca()

    ax.pcolormesh(1e3*x, 1e3*y, csd_or_sensor) #plot the background
    plot_polytopes(ax, polytopesks, axes_rescale=1e3, only_edges=only_edges, only_labels=only_labels) #plot the polytopes

    ax.set_xlim(x[0]*1e3, x[-1]*1e3)
    ax.set_ylim(y[0]*1e3, y[-1]*1e3)
    ax.axis('off')
    plt.tight_layout(pad=0)

    return plt.gcf(), ax

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
        "sensor_dot_indices": [-1],  #Indices of the sensor dots
        "sensor_detunings": [-0.02],  #Detuning of the sensor dots
        "noise_amplitude": {"fast_noise":c.fast_noise_amplitude, "slow_noise": c.slow_noise_amplitude}, #Noise amplitude for the sensor dots in eV
        "peak_width_multiplier": 25,  #Width of the sensor peaks in the units of thermal broadening m *kB*T/0.61.
    }


    return capacitance_config, tunneling_config, sensor_config

def generate_dataset(x_vol: np.ndarray, y_vol: np.ndarray, ks:int=0, device:np.ndarray=None, config_tuple:tuple[int,int,int]=None):
    """
        Run the QDarts experiment.
    """
    if config_tuple is None:
        raise ValueError("config_tuple must be provided")
        
    K, N, S = config_tuple
    c.validate_state(K, N, S)
    
    if S > 0 and device is None:
        raise ValueError("Device must be provided when using sensors (S > 0)")
        
    try:
        C_DD, C_DG, sensors = generate_capacitance_matrices(config_tuple, device)
        
        try:
            cut = get_cut(config_tuple)
        except ValueError as e:
            print(f"Error generating cut: {e}")
            return None
            
        use_sensor_signal = S > 0
        
        try:
            if S == 0:  # Replace NOISE check
                capacitance_config, _, _ = generate_experiment_config(C_DD, C_DG, config_tuple, device)
                experiment = Experiment(capacitance_config)

                result = experiment.generate_CSD(
                x_voltages=x_vol,  # V
                y_voltages=y_vol,  # V
                plane_axes=cut,
                compute_polytopes=True
            )
            else:
                # capacitance_config, tunneling_config, sensor_config = generate_experiment_config(C_DD, C_DG, config_tuple, device)
                experiment = Experiment(*generate_experiment_config(C_DD, C_DG, config_tuple, device))

                result = experiment.generate_CSD(
                    x_voltages=x_vol,  # V
                    y_voltages=y_vol,  # V
                    
                    #TODO: depending on the size of the system change the target state/transition accordingly
                    target_state = [1,0,5],  # target state for transition
                    target_transition = [-1,1,0], #target transition from target state, here transition to [2,3,2,3,5,5]
                    
                    # compensate_sensors=True,

                    plane_axes=cut,
                    compute_polytopes=True,
                    use_sensor_signal=use_sensor_signal,
                )
            
            if result is None:
                print("generate_CSD returned None")
                return None
                
            xks, yks, csd_dataks, polytopesks, sensor, _ = result
            
            if S > 0: 
                return C_DD, C_DG, ks, cut, xks, yks, csd_dataks, polytopesks, sensor[:,:,0], device, sensors
            else:
                return C_DD, C_DG, ks, cut, xks, yks, csd_dataks, polytopesks, None, device, sensors
            
        except Exception as e:
            print(f"Error in experiment generation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
            
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
    
    # Ensure the path exists before trying to list directories
    ensure_path(path)
    
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

def save_img_csd(config_tuple, csd_plot):
    """
    Save the CSD image as a PNG file with a 'unique' name.
    """
    K, N, S = config_tuple
    img_name = ''.join([str(random.randint(0, 9)) for _ in range(10)])+'.png'
    full_path_img = os.path.join(PATH_IMG, img_name)
    
    csd_plot.savefig(full_path_img, 
                     format='png', 
                     bbox_inches='tight', 
                     pad_inches=0, 
                     dpi=DPI)
    
    plt.close(csd_plot)
    return full_path_img, img_name

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
            
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (np.ndarray, list, torch.Tensor)):
                    if isinstance(sub_value, torch.Tensor):
                        sub_value = sub_value.numpy()
                    group.create_dataset(sub_key, data=sub_value)
                elif isinstance(sub_value, (int, float, str)):
                    group.attrs[sub_key] = sub_value
                elif sub_value is None or np.isnan(sub_value):
                    group.attrs[sub_key] = 'NaN'
                else:
                    print(f"Unsupported type for {sub_key}: {type(sub_value)}")

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
    return Image.fromarray((tensor.transpose(1, 2, 0)))

def reconstruct_img_with_matrices(batch_num: int, img_name: str, config_tuple: tuple[int, int, int], show: bool=False):
    """
    Reconstruct image and get associated matrices from HDF5 file.
    """
    img_name = check_and_correct_img_name(img_name)
    path = get_path_hfd5(batch_num, config_tuple)

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
    
def save_datapoints(config_tuple, C_DD, C_DG, ks, x_vol, y_vol, cuts, csd_plot, csd_gradient, device, sensors):
    """
    Combine all 'saving' functions to create a datapoint.
    """
    K, N, S = config_tuple
   
    # save img of CSD 
    fpi, img_name = save_img_csd(config_tuple, csd_plot)
   
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

    # Handle gradient data similarly
    if len(csd_gradient.shape) == 2:
        csd_gradient_tensor = torch.tensor(csd_gradient[None, :, :])
    else:
        csd_gradient_tensor = torch.tensor(csd_gradient).permute(2, 0, 1)

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
        'csd': csd_tensor,
        'csd_gradient': csd_gradient_tensor,
        'device': device,
        'sensors_coordinates': sensors,
        'p_dd': c.p_dd,
        'p_dg': c.p_dg,
        'd_DD': c.d_DD,
        'd_DG': c.d_DG,
        'r_min': c.r_min,
        'r_max': c.r_max,
    }} # 22 elements
    
    save_to_hfd5(datapoints_dict)


def generate_datapoint(args):
    x_vol, y_vol, ks, device, i, N_batch, config_tuple = args
    K, N, S = config_tuple
    print(f"Generating datapoint {i+1}/{N_batch}:")
    print(f"Configuration: K={K}, N={N}, S={S}")

    try:
        # Create a unique seed for this process
        process_id = os.getpid()
        current_time = int(time.time() * 1000)  # Current time in milliseconds
        unique_seed = (process_id + current_time + i) % (2**32 - 1)  # Ensure it's within numpy's seed range
        
        # Set the seed for numpy and random
        np.random.seed(unique_seed)
        random.seed(unique_seed)
        
        result = generate_dataset(x_vol, y_vol, ks, device, config_tuple)
        if result is None:
            return None
            
        C_DD, C_DG, ks, cut, x, y, csd, poly, sensor, device, sensors = result
        
        if S == 0:  # Replace NOISE check
            fig, _ = plot_CSD(x, y, csd, poly, only_labels=False)
            return (C_DD, C_DG, ks, cut, x_vol, y_vol, fig, np.gradient(csd,axis=0)+np.gradient(csd,axis=1), device, sensors)
        else:
            fig, _ = plot_CSD(x, y, sensor, poly)
            return (C_DD, C_DG, ks, cut, x_vol, y_vol, fig, np.gradient(csd,axis=0)+np.gradient(csd,axis=1), device, sensors)
            
    except Exception as e:
        print(f"Execution failed for datapoint {i+1}!")
        print(f"Error: {e}")
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

def load_parameters(batch_num: int, img_name: str, config_tuple: tuple[int, int, int], param_names: list[str], print_available_params: bool=False) -> dict:
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