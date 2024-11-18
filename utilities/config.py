import os 
import torch

PATH_0 = "./datasets/"
# PATH_0 = "./ALICE/"

DPI = 100
RESOLUTION = 256 #96 256

d_DD = 100 # nm
d_DG = 100 # nm
p_dd = 0.4
p_dg = 0.15

N = 2 # Number of dots (in the device)
def set_global_N(value):
    global N, K
    N = value
    K = N + S  # Update K whenever N changes

def get_global_N():
    return N

S = 0 # Number of sensors
def set_global_S(value):
    global S, K
    S = value
    K = N + S  # Update K whenever S changes

def get_global_S():
    return S

r_min = 2*d_DD
r_max = 6*d_DD


K = N+S # Total number of dots
def set_global_K(value):
    global K
    K = value

def get_global_K():
    return K

NOISE = False
def set_global_NOISE(value):
    global NOISE, S
    S = 1
    NOISE = value

if NOISE:
    PATH = os.path.join(PATH_0, 'noise', 'N-'+str(N)+'_S-'+str(S), str(RESOLUTION)+'x'+str(RESOLUTION))
else:
    PATH = os.path.join(PATH_0, 'K-'+str(K), str(RESOLUTION)+'x'+str(RESOLUTION))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE = 1
def set_global_MODE(value):
    global MODE
    MODE = value

def get_global_MODE():
    return MODE

