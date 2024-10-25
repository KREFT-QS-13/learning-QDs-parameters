import os 
import torch

PATH_0 = "./datasets/"

DPI = 100
RESOLUTION = 96 #96 256

K = 2
def set_global_K(value):
    global K
    K = value

def get_global_K():
    return K


PATH = os.path.join(PATH_0, 'K-'+str(K), str(RESOLUTION)+'x'+str(RESOLUTION))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE = 1
def set_global_MODE(value):
    global MODE
    MODE = value

def get_global_MODE():
    return MODE

