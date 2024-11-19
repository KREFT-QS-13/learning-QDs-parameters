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

NOISE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_path(K, N, S):
    if NOISE:
        return os.path.join(PATH_0, 'noise', f'N-{N}_S-{S}', f'{RESOLUTION}x{RESOLUTION}')
    else:
        return os.path.join(PATH_0, f'K-{K}', f'{RESOLUTION}x{RESOLUTION}')

def validate_state(K, N, S):
    """Validate the configuration state."""
    print(f"K: {K}, N: {N}, S: {S}")
    print(f"c.NOISE: {NOISE}, p_dd: {p_dd}, p_dg: {p_dg}")
    if K != N + S:
        raise ValueError(f"Invalid configuration: K ({K}) != N ({N}) + S ({S})")
    if N < 2:
        raise ValueError(f"Invalid configuration: N ({N}) must be at least 2")
    if NOISE and S < 1:
        raise ValueError(f"Invalid configuration: When NOISE is True, S ({S}) must be at least 1")



