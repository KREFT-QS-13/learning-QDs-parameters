import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms
import torch

import os
import sys
import re
import h5py
import json

sys.path.append('./qdarts')
from experiment import Experiment
from plotting import plot_polytopes

PATH = "./datasets/"


mpl.rcParams["savefig.bbox"] = 'tight'
mpl.rcParams["savefig.pad_inches"] = 0
mpl.rcParams["savefig.dpi"] = 100

def generate_capacitance_matrices(K: int) -> tuple[np.ndarray, np.ndarray]:
    """
        Generate random capacitance matrices for a given number of dots K from a normal distribution.
        
        The diagonal elements of C_DD are drawn from a normal distribution 
        with a mean of 2*mean and a standard deviation of 10% of the mean.

        The off-diagonal elements of C_DD and C_DG are drawn from a normal distribution 
        with a mean and standard deviation of 10% of mean.
    """
    mean = 3 #aF
    std = 0.1*mean
    C_DD, C_DG = np.random.normal(mean, std, (K,K)), np.random.normal(2*mean, std, (K,K))
    for i in range(K):
        C_DD[i,i] = np.random.normal(2*mean, std)
    
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

def plot_CSD(x: np.ndarray, y: np.ndarray, csd: np.ndarray, polytopesks: list[np.ndarray], res:int=256, dpi:int=100):
    """
        Plot the charge stability diagram (CSD) (res by res, default 256 by 256).
    """
    fig, ax = plt.subplots(figsize=(res/dpi, res/dpi), dpi=dpi)

    ax.pcolormesh(1e3*x, 1e3*y, csd) #plot the background
    plot_polytopes(ax, polytopesks, axes_rescale=1e3, only_edges=True) #plot the polytopes

    ax.set_xlim(x[0]*1e3, x[-1]*1e3)
    ax.set_ylim(y[0]*1e3, y[-1]*1e3)
    ax.axis('off')
    plt.tight_layout(pad=0)

    return fig, ax

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

def save_img_csd(K, csd_plot):
    """
        Save the CSD image as an PNG file with the 'unique' name.
        (There might be problem with uniqness for more datapoint. However
        the batch approach would solve it)
    """
    img_name = ''.join([str(random.randint(0, 9)) for _ in range(10)])+'.png'

    full_path_img = os.path.join(PATH_IMG, img_name)
    
    csd_plot.savefig(full_path_img, format='png') 
    plt.close(csd_plot)
    
    return full_path_img, img_name

# TODO: Figure out how to save the data in multiple files after 500 datapoints generation
#     - thats for safty 
#     - also start thinking how you will orgenize for more datapoints with bigger K
#     - for bigger K -> 10 000 datapoints -> \biom{K}{2} * 10 0000 real datapoints ?
#     - how about the philosophy: you generate to learn (?)
def save_to_json(dictionary: dict):
    """
        Save the datapoints to a json file.
    """      
    full_path_dps = os.path.join(PATH_DPS, 'datapoints.json')
    
    try:
        if os.path.exists(full_path_dps):
            with open(full_path_dps, 'r') as f:
                data = json.load(f)
            
            data.update(dictionary)
        else:
            data = dictionary

        with open(full_path_dps, 'w') as f:
            json.dump(dictionary, f)
    except:
        print("Json file no longer updated!")


def save_to_hfd5(dictionary: dict):
    """
        Save the datapoints to a hfd5 file.
    """
    # TODO: Finish this function:
    #  - use img id as key as save under the inside of nested dict 
    #  - use the key of the inside dict as attributes

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

def get_path_hfd5(K:int, batch_num:int, v:bool=False):
    """
        Load the datapoints from a hfd5 file.
        For know it is for testing and not yet finished.
    """
    batch_name = 'batch-' + str(int(batch_num)) if batch_num <= count_directories_in_folder(K) else ValueError("Batch number is too high!")
    full_path_dps = os.path.join(PATH, 'K-'+str(K), batch_name, 'datapoints.h5')
          
    return full_path_dps

def load_csd_img(K:int, batch_num:int, csd_name: str, show:bool=False):
    """
        Load the PNG file 
    """
    if not re.compile(r"\d.png").match(csd_name):
        csd_name = csd_name + ".png"

    path = os.path.join(PATH, 'K-'+str(K), 'batch-'+str(batch_num), 'imgs', csd_name)
    img = Image.open(path)
    if show:
        img.show() 
    
    return img 

def save_datapoints(K, C_DD, C_DG, ks, x_vol, y_vol, cuts, csd_plot):
    """
       Combine all 'saveing' function to create a datapoint containg an PNG image, 
       a new json instantce in the 'batch' datapoints.json file, as well as a new hfd5 
       instantce in the 'batch' datapoints.h5 file.

       #TODO: should I save ks as well?
    """
   
    # save img of CSD 
    fpi, img_name = save_img_csd(K, csd_plot)
   
    # save datapoints
    csd = Image.open(fpi)
    csd_tensor = torch.tensor(np.array(csd)).permute(2, 0, 1)

    datapoints_dict = {img_name : {'K':K, 
                                   'C_DD':C_DD.tolist(), 
                                   'C_DG':C_DG.tolist(), 
                                   'ks':ks,
                                   'x_vol':x_vol, 
                                   'y_vol':y_vol, 
                                   'cuts':cuts, 
                                   'csd':csd_tensor.tolist()}}
    save_to_json(datapoints_dict)
    
    # img_name = img_name.split('.')[0]
    datapoints_dict = {img_name:{k: np.array(v) if not isinstance(v, int) else v for (k,v) in datapoints_dict[img_name].items()}}
    save_to_hfd5(datapoints_dict)


def generate_and_save_datapoints(K, x_vol, y_vol):
    x_vol_range = (x_vol[-1], len(x_vol))
    y_vol_range = (y_vol[-1], len(y_vol))

    C_DD, C_DG, ks, cuts, x, y, csd, poly = generate_dataset(K, x_vol, y_vol)
    
    fig, _ = plot_CSD(x, y, csd, poly)    
     
    save_datapoints(K, C_DD, C_DG, ks, x_vol_range, y_vol_range, cuts, fig)