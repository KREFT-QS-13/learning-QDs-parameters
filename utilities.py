import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')  # Use the 'Agg' backend which is thread-safe
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
import torch

import os
import sys
import shutil
import re
import h5py
import json

sys.path.append('./qdarts')
from experiment import Experiment
from plotting import plot_polytopes

PATH = "./datasets/"
DPI = 100
RESOLUTION = 256

def draw_digonal_elemnts(K:int, C_DD:np.ndarray, C_DG):
    pass

def generate_capacitance_matrices(K: int) -> tuple[np.ndarray, np.ndarray]:
    """
        Generate random capacitance matrices for a given number of dots K from a normal distribution.
        
        The diagonal elements of C_DD are drawn from a normal distribution 
        with a mean of 2*mean and a standard deviation of 10% of the mean.

        The off-diagonal elements of C_DD and C_DG are drawn from a normal distribution 
        with a mean and standard deviation of 10% of mean.
    """
    mean = 1.0 #aF
    std = 0.15
    C_DD, C_DG = np.random.normal(mean, std, (K,K)), np.random.normal(mean, std, (K,K))
    
    # diag_const = np.random.uniform(low=3, high=7)
    diag_const = np.random.choice([2,3,5,7,9,11,13,15,17,21])

    for i in range(K):
        C_DD[i,i] = np.random.normal(diag_const*mean, diag_const*std)
        C_DG[i,i] = np.random.normal(diag_const*mean, diag_const*std)


        # if we want to keep similiar magnitude on both dot-dot capacitance 
    # for i in range(K):
        # C_DD[i,i] = np.random.normal(diag_const*mean, std)
        # C_DG[i,i] = np.random.normal(diag_const*mean, std)
        # coin filp: between those two

    C_DD = (C_DD + C_DD.T)/2

    return C_DD, C_DG

def generate_dummy_data(K: int) -> tuple[np.ndarray, np.ndarray]:
    """
        Generate dummy (identity matrix) capacitance matrices for a given number of dots K.
    """
    return np.identity(K), np.identity(K)

def generate_cuts(K: int):
    """
        Generate all possible cuts multidimensional volatge space based on the number of dots K.
    """
    #TODO: Wirte a function that based on K dots will generate \biom{K}{2} diffrent cuts
    pass

def plot_CSD(x: np.ndarray, y: np.ndarray, csd: np.ndarray, polytopesks: list[np.ndarray], res:int=RESOLUTION, dpi:int=DPI):
    """
        Plot the charge stability diagram (CSD) (res by res, default 256 by 256).
    """
    plt.figure(figsize=(res/dpi, res/dpi), dpi=dpi)
    ax = plt.gca()

    ax.pcolormesh(1e3*x, 1e3*y, csd) #plot the background
    plot_polytopes(ax, polytopesks, axes_rescale=1e3, only_edges=True) #plot the polytopes

    ax.set_xlim(x[0]*1e3, x[-1]*1e3)
    ax.set_ylim(y[0]*1e3, y[-1]*1e3)
    ax.axis('off')
    plt.tight_layout(pad=0)

    return plt.gcf(), ax

def generate_dataset(K: int, x_vol: np.ndarray, y_vol: np.ndarray, ks: int=0):
    """
        Run the QDarts experiment for a given number of dots K and
          ranges of voltages to create needed data for CSD creation.
    """
    C_DD, C_DG = generate_capacitance_matrices(K)
    # C_DD, C_DG = generate_dummy_data(K)

    capacitance_config = {
        "C_DD" : C_DD,  #dot-dot capacitance matrix
        "C_Dg" : C_DG,  #dot-gate capacitance matrix
        "ks" : ks,       
    }

    cuts = [[1,0],[0,1]]

    xks, yks, csd_dataks, polytopesks, _, _ =  Experiment(capacitance_config).generate_CSD(
                                                x_voltages = x_vol,  #V
                                                y_voltages = y_vol,  #V
                                                plane_axes = cuts,
                                                compute_polytopes = True,
                                                use_virtual_gates = False)   
    
    return C_DD, C_DG, ks, cuts, xks, yks, csd_dataks, polytopesks

def count_directories_in_folder(K:int, path:str = PATH):
    """
        Count the number of batch directories in a given folder.
    """
    path = os.path.join(path, 'K-'+str(K))
    batch_list = [x for x in os.listdir(path) if re.compile(r"batch-\d").match(x)] 

    return sum(os.path.isdir(os.path.join(path, x)) for x in batch_list)


def create_paths(K:int, path:str=PATH):
    """
        Creates paths for datapoints and images where the data will be saved.
    """
    global PATH_IMG
    global PATH_DPS

    batch_name = 'batch-' + str(count_directories_in_folder(K)+1)
    
    full_path = os.path.join(PATH, 'K-'+str(K), batch_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    full_path_dps = full_path
    
    full_path_img = os.path.join(full_path, 'imgs')
    if not os.path.exists(full_path_img):
        os.makedirs(full_path_img)

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

def save_img_csd(K, csd_plot):
    """
        Save the CSD image as a PNG file with a 'unique' name.
    """
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
        Save the datapoints to a hfd5 file.
    """
    full_path_dps = os.path.join(PATH_DPS, 'datapoints.h5')

    with h5py.File(full_path_dps, 'a') as f: 
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Handle nested dictionaries by updating or creating groups
                if key in f:
                    raise ValueError("Key already exists!")
                else:
                    group = f.create_group(key)
                
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray) or isinstance(sub_value, list):
                        if sub_key in group:
                            # Overwrite the existing dataset
                            del group[sub_key]  # Delete the existing dataset
                        group.create_dataset(sub_key, data=sub_value)
                    elif isinstance(sub_value, int) or isinstance(sub_value, float):
                        # Overwrite or add new attribute
                        group.attrs[sub_key] = sub_value
            elif isinstance(value, np.ndarray):
                # Update or add new NumPy arrays as datasets
                if key in f:
                    # Overwrite the existing dataset
                    del f[key]  # Delete the existing dataset
                f.create_dataset(key, data=value)
            else:
                # Overwrite or add scalar values as attributes
                f.attrs[key] = value

def get_batch_folder_name(K:int, batch_num:int):
    if batch_num <= count_directories_in_folder(K):
        return 'batch-' + str(batch_num)
    else:
        print(ValueError(f"Batch number is too high! Max: {count_directories_in_folder(K)}!"))
        return None

def get_path_hfd5(K:int, batch_num:int, v:bool=False):
    """
        Load the datapoints from a hfd5 file.
        For know it is for testing and not yet finished.
    """
    batch_name = get_batch_folder_name(K, batch_num)

    full_path_dps = os.path.join(PATH, 'K-'+str(K), batch_name, 'datapoints.h5')
          
    return full_path_dps

def check_and_correct_img_name(img_name: str):
    if not re.compile(r"^\d+\.png$").match(img_name):
        return img_name + ".png"
    else:
        return img_name

def load_csd_img(K:int, batch_num:int, csd_name: str, show:bool=False):
    """
        Load the PNG file 
    """
    csd_name =  check_and_correct_img_name(csd_name)
    path = os.path.join(PATH, 'K-'+str(K), get_batch_folder_name(K, batch_num), 'imgs', csd_name)
    
    img = Image.open(path)
    if show:
        img.show() 
    
    return img 

def reconstruct_img_with_matrices(K:int, batch_num:int, img_name:str, show:bool = False):
    img_name = check_and_correct_img_name(img_name)
    path = get_path_hfd5(K, batch_num)

    with h5py.File(path, 'r') as f:
        img = f[img_name]['csd'][:]
        C_DD = f[img_name]['C_DD'][:]
        C_DG = f[img_name]['C_DG'][:]
        
        img = plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')

        return img, C_DD, C_DG
    
def save_datapoints(K, C_DD, C_DG, ks, x_vol, y_vol, cuts, csd_plot):
    """
       Combine all 'saveing' function to create a datapoint containg an PNG image, 
       a new json instantce in the 'batch' datapoints.json file, as well as a new hfd5 
       instantce in the 'batch' datapoints.h5 file.
    """
   
    # save img of CSD 
    fpi, img_name = save_img_csd(K, csd_plot)
   
    # save datapoints
    csd = Image.open(fpi)
    csd_tensor = torch.tensor(np.array(csd)).permute(2, 0, 1)
    
    ks = np.nan if ks == None else ks
    datapoints_dict = {img_name : {'K':K, 
                                   'C_DD':C_DD, 
                                   'C_DG':C_DG, 
                                   'ks':ks,
                                   'x_vol':x_vol, 
                                   'y_vol':y_vol, 
                                   'cuts':cuts, 
                                   'csd':csd_tensor}} # 8 elements
    # save_to_json(datapoints_dict)
    
    save_to_hfd5(datapoints_dict)


def generate_datapoint(args):
    K, x_vol, y_vol, ks, i, N = args
    print(f"Generating datapoint {i+1}/{N}:")
    try:
        C_DD, C_DG, ks, cuts, x, y, csd, poly = generate_dataset(K, x_vol, y_vol, ks)
        fig, _ = plot_CSD(x, y, csd, poly)
        return (C_DD, C_DG, ks, cuts, x_vol, y_vol, fig)
    except Exception as e:
        print(f"Execution failed for datapoint {i+1}!")
        print(f"Error: {e}")
        return None
