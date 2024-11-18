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

N = 2  # Number of dots (in the device)
S = 0  # Number of sensors
K = N + S  # Total number of dots

r_min = 2 * d_DD
r_max = 6 * d_DD

NOISE = False
MODE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_K():
    global K
    K = N + S
    
def set_global_N(value):
    global N
    N = value
    update_K()

def set_global_S(value):
    global S
    S = value
    update_K()

def set_global_K(value):
    global K
    K = value

def set_global_NOISE(value):
    global NOISE
    NOISE = value

def get_path():
    if NOISE:
        return os.path.join(PATH_0, 'noise', f'N-{N}_S-{S}', f'{RESOLUTION}x{RESOLUTION}')
    else:
        return os.path.join(PATH_0, f'K-{K}', f'{RESOLUTION}x{RESOLUTION}')

def validate_state():
    """Validate the current configuration state."""
    if K != N + S:
        raise ValueError(f"Invalid configuration: K ({K}) != N ({N}) + S ({S})")
    if N < 2:
        raise ValueError(f"Invalid configuration: N ({N}) must be at least 2")
    if NOISE and S < 1:
        raise ValueError(f"Invalid configuration: When NOISE is True, S ({S}) must be at least 1")



