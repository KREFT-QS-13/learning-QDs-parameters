import os 
import torch

PATH_0 = "./datasets/"
# PATH_0 = "./ALICE/"

DPI = 100
RESOLUTION = 256 #96 256

d_DD = 100 # nm
d_DG = 100 # nm
p_dd = 0.5
p_dg = 0.2

tunnel_coupling_const = 500*1e-6
slow_noise_amplitude = 1e-12 # 0.8*1e-90
fast_noise_amplitude = 1e-12 # 2*1e-90  

r_min = 2.5 * d_DD
r_max = 5 * d_DD

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



