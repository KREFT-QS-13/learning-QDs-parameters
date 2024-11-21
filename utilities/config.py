import os 
import torch

PATH_0 = "./datasets/"
# PATH_0 = "./ALICE/"

DPI = 100
RESOLUTION = 96 #96 256

## ------- Quantum dots system parameters -------
d_DD = 100 # nm
d_DG = 100 # nm
p_dd = 0.65
p_dg = 0.3

mag_list = [6,7,8,9,10,12,13,15,17,20]

tunnel_coupling_const = 100*1e-6    
slow_noise_amplitude = 5*1e-8 # 0.8*1e-90
fast_noise_amplitude = 5*1e-8 # 2*1e-90  

r_min = 21 * d_DD
r_max = 22   * d_DD

system_name = ''
def get_path(K, N, S):
    """Get the path based on configuration."""
    if S > 0:  # Replaces NOISE check
        if not system_name or system_name.isspace():  # Check if empty or whitespace
            return os.path.join(PATH_0, f'N-{N}_S-{S}', f'{RESOLUTION}x{RESOLUTION}')
        return os.path.join(PATH_0, f'{system_name}-N-{N}_S-{S}', f'{RESOLUTION}x{RESOLUTION}')    
    else:
        return os.path.join(PATH_0, f'K-{K}', f'{RESOLUTION}x{RESOLUTION}')

def validate_state(K, N, S):
    """Validate the configuration state."""
    if K != N + S:
        raise ValueError(f"Invalid configuration: K ({K}) != N ({N}) + S ({S})")
    if N < 2:
        raise ValueError(f"Invalid configuration: N ({N}) must be at least 2")
    if S < 0:
        raise ValueError(f"Invalid configuration: S ({S}) cannot be negative")


## ------- Machine Learning Model parameters ------- 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
