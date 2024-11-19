import os 
import torch

PATH_0 = "./datasets/"
# PATH_0 = "./ALICE/"

DPI = 100
RESOLUTION = 96 #96 256

d_DD = 100 # nm
d_DG = 100 # nm
p_dd = 0.65
p_dg = 0.25

tunnel_coupling_const = 100*1e-8
slow_noise_amplitude = 5*1e-5 # 0.8*1e-90
fast_noise_amplitude = 5*1e-5 # 2*1e-90  

r_min = 3 * d_DD
r_max = 3.5 * d_DD

system_name = ''
def get_path(K, N, S):
    """Get the path based on configuration."""
    if S > 0:  # Replaces NOISE check
        if not system_name or system_name.isspace():  # Check if empty or whitespace
            return os.path.join(PATH_0, f'K-{K}', f'{RESOLUTION}x{RESOLUTION}')
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

### Machine Learning conf   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
