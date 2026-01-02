import json
import os, glob
import numpy as np
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from qdarts.experiment import Experiment
from qdarts.plotting import plot_polytopes

from PIL import Image
import torchvision.transforms as transforms
import torch

# Physical constants
e = 1.602176634e-19  # Coulomb (elementary charge)
meV = 1.602176634e-22  # Joule (milli-electron-volt = e * 1e-3)

# Energy conversion factor
E_CONV_FACTOR = (e**2) / (2.0 * meV)  # C^2 / (J/meV)

class QuantumDotModel:
    """
        Initializes the random quantum dot model.

        Parameters
        ----------
        Nd : int
            The total number of quantum dots in the system (e.g., 4 for a
            3-dot system + 1 sensor).
        Ng : int
            The total number of gates in the system. This is typically
            equal to Nd, as each dot has at least one plunger gate.
        params : dict[str, float]
            A dictionary of hyperparameters used to control the statistical
            generation of the device geometry and capacitances.
        """

    def __init__(self, Ndots: int, Ngates: int, params: dict[str, float]):
        self.Ndots = Ndots
        self.Ngates = Ngates
        self.params = params

    def _generate_dot_distances_2d_batch(
        self, 
        Nconfigurations: int = 10000,
        batch_size: int = 12000, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates 2D coordinates for Nd dots in batches, ensuring minimum separation.
        
        Parameters
        ----------
        Nconfigurations : int
            Total number of valid geometries to generate
        batch_size : int
            Number of geometries to generate per batch (should be >= Nconfigurations
            to account for some invalid geometries)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            1. dot_distances: The (Nconfigurations, Nd, Nd) Euclidean distance matrices
            2. coords: The (Nconfigurations, Nd, 2) array of (x, y) coordinates
        """
        Nd = self.Ndots
        d_mean = self.params.get("d_mean_nm", 100.0) * 1e-9
        d_std = self.params.get("d_std_nm", 20.0) * 1e-9
        d_min = self.params.get("d_min_nm", 60.0) * 1e-9
        
        # Pre-allocate output arrays
        all_coords = np.zeros((Nconfigurations, Nd, 2))
        all_distances = np.zeros((Nconfigurations, Nd, Nd))
        n_collected = 0
        
        # Keep generating batches until we have enough valid geometries
        while n_collected < Nconfigurations:
            # Generate a batch
            coords = np.zeros((batch_size, Nd, 2))
            
            # Pre-generate all random samples for this batch
            anchors = []
            distances = np.zeros((batch_size, Nd - 1))
            angles = np.zeros((batch_size, Nd - 1))
            
            for i in range(1, Nd):
                anchor_choices = np.random.randint(0, i, size=batch_size)
                anchors.append(anchor_choices)
                
                distances[:, i-1] = np.maximum(
                    d_min, 
                    np.random.normal(d_mean, d_std, size=batch_size)
                )
                angles[:, i-1] = np.random.uniform(0, 2 * np.pi, size=batch_size)
            
            # Build coordinates sequentially but vectorized across batch
            for i in range(1, Nd):
                anchor_idx = anchors[i-1]
                d = distances[:, i-1]
                theta = angles[:, i-1]
                
                anchor_coords = coords[np.arange(batch_size), anchor_idx]
                offsets = np.column_stack([
                    d * np.cos(theta),
                    d * np.sin(theta)
                ])
                coords[:, i] = anchor_coords + offsets
            
            # Vectorized distance checking
            coords_expanded_i = coords[:, :, np.newaxis, :]  # (batch_size, Nd, 1, 2)
            coords_expanded_j = coords[:, np.newaxis, :, :]  # (batch_size, 1, Nd, 2)
            
            diff = coords_expanded_i - coords_expanded_j  # (batch_size, Nd, Nd, 2)
            pairwise_distances = np.linalg.norm(diff, axis=-1)  # (batch_size, Nd, Nd)
            
            # Check validity
            identity_mask = np.eye(Nd, dtype=bool)
            valid_pairs = pairwise_distances >= d_min
            valid_pairs[:, identity_mask] = True  # Ignore diagonal
            
            geometry_valid = np.all(valid_pairs, axis=(1, 2))  # (batch_size,)
            
            # Keep only valid geometries and fill pre-allocated arrays
            valid_coords = coords[geometry_valid]
            valid_distances = pairwise_distances[geometry_valid]
            n_valid = np.sum(geometry_valid)
            
            if n_valid > 0:
                n_needed = Nconfigurations - n_collected
                n_to_take = min(n_needed, n_valid)
                
                # Fill pre-allocated arrays directly (no appending/concatenation)
                all_coords[n_collected:n_collected + n_to_take] = valid_coords[:n_to_take]
                all_distances[n_collected:n_collected + n_to_take] = valid_distances[:n_to_take]
                n_collected += n_to_take
        
        return all_distances, all_coords


    def _generate_from_physics_batch(
        self, 
        dot_distances_batch: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Vectorized version: Samples random device parameters for a batch of geometries.
        
        Parameters
        ----------
        dot_distances_batch : np.ndarray
            Shape (batch_size, Nd, Nd) - batch of distance matrices
            
        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing all capacitance matrices and derived quantities:
            - 'C_tilde_DG': (batch_size, Nd, Ng)
            - 'C_DG': (batch_size, Nd, Ng)
            - 'C_m': (batch_size, Nd, Nd)
            - 'C_DD': (batch_size, Nd, Nd)
            - 'C_DD_inv': (batch_size, Nd, Nd)
            - 'E_matrix_meV': (batch_size, Nd, Nd)
            - 'Ec_meV': (batch_size, Nd)
            - 'C_tilde_DD': (batch_size, Nd, Nd)
            - 'tc_meV': (batch_size, Nd, Nd)
            - 'alpha': (batch_size, Nd, Ng)
        """
        batch_size, Nd, _ = dot_distances_batch.shape
        Ndots = Nd - 1
        Ng = self.Ngates
        d_nn_proxy = self.params.get("d_mean_nm", 100.0) * 1e-9
        
        # Get sensor indices from params
        sensor_idx = self.params.get("sensor_idx", Nd - 1)
        sensor_gate_idx = self.params.get("sensor_gate_idx", Nd - 1)
        
        # --- 1. Model Gate Capacitance (C_tilde_DG) ---
        C_tilde_DG = np.zeros((batch_size, Nd, Ng))
        
        # Diagonal elements: Dot_i to its own Gate_i
        # For diagonal, d_nm = 0, so geometric factor = 50/sqrt(50^2 + 0^2) = 1
        diag_base_values = np.random.normal(
            self.params["C_dg_diag_mean"],
            self.params["C_dg_diag_std"],
            size=(batch_size, Nd)
        )
        diag_base_values = np.maximum(0.0, diag_base_values)
        
        # Set diagonal elements (geometric factor = 1 for d_nm = 0)
        for i in range(Nd):
            C_tilde_DG[:, i, i] = diag_base_values[:, i]
        
        # Cross-talk elements: Dot_i to Gate_j (where i != j)
        # Apply geometric model: C_dg(diag) * 50/(sqrt(50^2 + d_nm^2))
        for i in range(Nd):
            for j in range(Ng):
                if i != j:
                    # Use distance from dot i to dot j as proxy for dot-gate distance
                    dist_ij_nm = dot_distances_batch[:, i, j] * 1e9  # Convert to nanometers
                    # Geometric model: 50/(sqrt(50^2 + d_nm^2))
                    geometric_factor = (30.0**2 / (30.0**2 + dist_ij_nm**2))**1.5
                    # Scale the diagonal base value by geometric factor
                    mean = diag_base_values[:, i] * geometric_factor
                    std_dev = self.params["C_dg_diag_std"] * geometric_factor
                    C_tilde_DG[:, i, j] = np.maximum(0.0, np.random.normal(mean, std_dev))
        
        # --- 2. Enforce Sensor Constraints ---
        # Set qubit-sensor gate capacitances to zero
        for i in range(sensor_idx):  # All qubit dots (indices < sensor_idx)
            C_tilde_DG[:, i, sensor_gate_idx] = 0.0
        
        # Set sensor dot to all gates (except its own) to zero
        if "C_dg_sensor_gate_fixed" in self.params:
            C_tilde_DG[:, sensor_idx, :] = 0.0
            C_tilde_DG[:, sensor_idx, sensor_gate_idx] = self.params["C_dg_sensor_gate_fixed"]
        
        C_DG = C_tilde_DG.copy()
        
        # --- 3. Model Dot-Dot Mutual Capacitance (C_m) ---
        C_m = np.zeros((batch_size, Nd, Nd))
        # Use new parameter names: Cm_qq and Cm_sq
        C_m_qq_mean = self.params.get("Cm_qq_mean", 8e-18)
        C_m_qq_std = self.params.get("Cm_qq_std", 3e-18)
        C_m_sq_mean = self.params.get("Cm_sq_mean", 1.5e-18)
        C_m_sq_std = self.params.get("Cm_sq_std", 0.3e-18)
        
        # Scale by distance: k = C_mean * d_mean, then C = k / d
        k_m_qq = C_m_qq_mean * d_nn_proxy
        k_m_sq = C_m_sq_mean * d_nn_proxy

        # Vectorized computation for all pairs (i, j) where i < j
        for i in range(Nd):
            for j in range(i + 1, Nd):
                dist_ij = dot_distances_batch[:, i, j]  # (batch_size,) in meters
                
                if i < sensor_idx and j < sensor_idx:
                    # Quantum-quantum dot coupling
                    mean = k_m_qq / dist_ij  # (batch_size,) in Farads
                    std_dev = C_m_qq_std * (d_nn_proxy / dist_ij)  # (batch_size,) in Farads
                else:
                    # Sensor-quantum dot coupling (one of i or j is sensor)
                    mean = k_m_sq / dist_ij  # (batch_size,) in Farads
                    std_dev = C_m_sq_std * (d_nn_proxy / dist_ij)  # (batch_size,) in Farads
                
                values = np.maximum(0.0, np.random.normal(mean, std_dev))
                C_m[:, i, j] = values
                C_m[:, j, i] = values


        # --- 4. Build the Full C_DD Matrix (Maxwell Matrix) ---
        # Use new parameter names: C0_q for qubit dots, C0_s for sensor dot
        C0_q_mean = self.params.get("C0_q_mean", 8e-18)
        C0_q_std = self.params.get("C0_q_std", 2e-18)
        C0_s_mean = self.params.get("C0_s_mean", 30e-18)
        C0_s_std = self.params.get("C0_s_std", 1e-18)
        
        # Generate substrate capacitance for each dot in each geometry
        C_d0 = np.zeros((batch_size, Nd))
        # Generate for qubit dots
        for i in range(sensor_idx):
            C_d0[:, i] = np.maximum(0.0, np.random.normal(C0_q_mean, C0_q_std, size=batch_size))
        # Generate for sensor dot
        C_d0[:, sensor_idx] = np.maximum(0.0, np.random.normal(C0_s_mean, C0_s_std, size=batch_size))
        # --- 4. Build the Full C_DD Matrix (Maxwell Matrix) ---
        C_DD = np.zeros((batch_size, Nd, Nd))
        C_cap_to_all_gates = np.sum(C_DG, axis=2)  # (batch_size, Nd)

        # Diagonal: sum of all gate capacitances + sum of mutual capacitances
        for i in range(Nd):
            C_DD[:, i, i] = C_cap_to_all_gates[:, i] + np.sum(C_m[:, i, :], axis=1) + C_d0[:, i]
        
        # Off-diagonal: negative mutual capacitances
        for i in range(Nd):
            for j in range(i + 1, Nd):
                C_DD[:, i, j] = -C_m[:, i, j]
                C_DD[:, j, i] = -C_m[:, i, j]
        

        # Fix sensor dot's self-capacitance (diagonal element) if specified
        if "C_DD_sensor_diag_fixed" in self.params:
            C_DD[:, sensor_idx, sensor_idx] = self.params["C_DD_sensor_diag_fixed"]

        # --- 5. Derive C_DD_inv and E_matrix (vectorized matrix inversion) ---
        C_DD_inv = np.zeros_like(C_DD)
        E_matrix_meV = np.zeros_like(C_DD)
        Ec_meV = np.zeros((batch_size, Nd))
        
        # Invert each matrix in the batch
        for b in range(batch_size):
            C_DD_inv[b] = np.linalg.inv(C_DD[b])
            E_matrix_meV[b] = E_CONV_FACTOR * C_DD_inv[b]
            Ec_meV[b] = np.diag(E_matrix_meV[b])
        
        # --- 6. Derive Canonical C_tilde_DD ---
        C_tilde_DD = C_m.copy()
        for i in range(Nd):
            C_tilde_DD[:, i, i] = np.sum(C_DD[:, i, :], axis=1)
        
        # --- 7. Derive Tunnel Couplings (tc) with new formula ---
        # New formula: tc = tc_max * exp(-att_per_nm * (d - d_min))
        tc_meV = np.zeros((batch_size, Nd, Nd))
        tc_max = self.params.get("tc_max_meV", 1.0)
        att_per_nm = self.params.get("att_per_nm", 0.05)
        d_dot_nm = self.params.get("d_dot_nm", self.params.get("d_min_nm", 50.0)/2)

        for i in range(Ndots):
            for j in range(i + 1, Ndots):
                # Get distance between dots in nanometers
                dist_ij_nm = dot_distances_batch[:, i, j] * 1e9  # (batch_size,) in nanometers
                
                # New formula: tc = tc_max * exp(-att_per_nm * (d - d_min))
                tc_values = tc_max * np.exp(-att_per_nm * (dist_ij_nm - 2*d_dot_nm))  
                
                # Ensure non-negative and within bounds
                tc_values = np.maximum(0.0, tc_values)
                tc_values = np.minimum(tc_max, tc_values)
                
                tc_meV[:, i, j] = tc_values
                tc_meV[:, j, i] = tc_values  # Symmetric
        
        for b in range(batch_size):
            if np.any(tc_meV[b, :, :] > 0.4):
                for i in range(Ndots):
                    for j in range(i + 1, Ndots):
                        if tc_meV[b, i, j] > 0.4:
                            print(f"i={i}, j={j}, dist_ij_nm={dot_distances_batch[b, i, j] * 1e9}, tc_value={tc_meV[b, i, j]}")
        
        # --- 8. Derive Alpha (vectorized matrix multiplication) ---
        # alpha = C_DD_inv @ C_DG for each batch element
        alpha = np.zeros((batch_size, Nd, Ng))
        for b in range(batch_size):
            alpha[b] = C_DD_inv[b] @ C_DG[b]
        
        return {
            'C_tilde_DG': C_tilde_DG,
            'C_DG': C_DG,
            'C_m': C_m,
            'C_DD': C_DD,
            'C_DD_inv': C_DD_inv,
            'E_matrix_meV': E_matrix_meV,
            'Ec_meV': Ec_meV,
            'C_tilde_DD': C_tilde_DD,
            'tc_meV': tc_meV,
            'alpha': alpha,
            'dot_distances_batch': dot_distances_batch
        }

def get_virtual_gate_transitions( 
    alpha: np.ndarray,
    Nd: int,
    C_DD_inv: np.ndarray,
    base_charge_state: np.ndarray = None,
) -> np.ndarray:
    """Return gate-voltage vectors that realise 0→1 transitions per dot."""
    if base_charge_state is None:
        base_charge_state = np.zeros( Nd, dtype=int)
    else:
        base_charge_state = np.asarray(base_charge_state)
        if len(base_charge_state) != Nd:
            raise ValueError("base_charge_state must match the number of dots.")

    try:
        alpha_inv = np.linalg.inv(alpha)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Alpha matrix is singular; transitions are undefined.") from exc

    transition_voltages = np.zeros((Nd, Nd))
    for i in range(Nd):
        v_i_full = alpha_inv[:, i]
        term1_V = -e * (C_DD_inv[i, :] @ base_charge_state)
        term2_V = -(e / 2.0) * C_DD_inv[i, i]
        V_i_scalar = term1_V + term2_V
        transition_voltages[i] = v_i_full * V_i_scalar

    return transition_voltages

def get_coulomb_diamond_sizes(C_DG: np.ndarray):
    """
    Calculate coulomb diamond sizes for each gate.
    
    For each gate j, the diamond size is calculated using the lever arm
    of the primary dot (dot j) to that gate.
    """
    return e/np.diag(C_DG)

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def config_to_params(config: dict) -> dict:
    """Convert JSON config to params dictionary for QuantumDotModel."""
    params = {}
    
    # System configuration
    params['Ndots'] = config['system']['Ndots']
    params['Ngates'] = config['system']['Ngates']
    
    # Geometry parameters
    params.update(config['geometry'])
    
    # Sensor configuration
    params.update(config['sensor_config'])
    
    # Parasitic capacitance
    params.update(config['parasitic_capacitance'])
    
    # Mutual capacitance
    params.update(config['mutual_capacitance'])
    
    # Gate capacitance
    params.update(config['gate_capacitance'])
    
    # Tunnel coupling
    params.update(config['tunnel_coupling'])
    
    return params


def plane_axes_from_pair(pair: tuple, Ng: int) -> np.ndarray:
    """Create plane axes from gate pair."""
    axes = np.zeros((2, Ng), dtype=float)
    axes[0, pair[0]] = 1.0
    axes[1, pair[1]] = 1.0
    return axes


def generate_cuts(Ndots: int, Nsensors: int, sensor_gate_idx: int) -> list:
    """Generate all possible cuts (combinations of target gates)."""
    # Target gates exclude the sensor gate
    target_gate_indices = [idx for idx in range(Ndots + Nsensors) if idx != sensor_gate_idx]
    plane_axis_specs = list(combinations(target_gate_indices, 2))
    return plane_axis_specs


def generate_datapoint(
    config: dict,
    datapoint_id: int,
    output_dir: str
) -> bool:
    """
    Generate a single data point with all cuts and save to folder.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Extract system parameters
    Ndots = config['system']['Ndots']
    Nsensors = config['system']['Nsensors']
    Ngates = config['system']['Ngates']
    Nd = Ndots + Nsensors
    sensor_idx = config['sensor_config']['sensor_idx']
    sensor_gate_idx = config['sensor_config']['sensor_gate_idx']
    n_diamonds_factor = config['system']['n_diamonds_factor']
    resolution = config['system']['resolution']
    num_realizations = config['CSD_generation']['number_of_realizations']
    # Convert config to params
    params = config_to_params(config)
    
    # Don't create folder yet - only create after successful generation
    num_digits = len(str(num_realizations))
    datapoint_dir = os.path.join(output_dir, f"datapoint_{datapoint_id:0{num_digits}d}")
    
    try:
        # Initialize quantum dot model
        Qdots = QuantumDotModel(Nd, Ngates, params)
        
        # Generate geometry (single configuration)
        all_distances, all_coords = Qdots._generate_dot_distances_2d_batch(
            Nconfigurations=1,
            batch_size=100
        )
        
        # Generate capacitance data
        capacitance_data = Qdots._generate_from_physics_batch(all_distances)
        
        # Extract single configuration (remove batch dimension)
        geometry = all_coords[0]  # (Nd, 2)
        C_tilde_DD = capacitance_data['C_tilde_DD'][0]  # (Nd, Nd)
        C_DG = capacitance_data['C_DG'][0]  # (Nd, Ng)
        tc_meV = capacitance_data['tc_meV'][0]  # (Nd, Nd)
        alpha = capacitance_data['alpha'][0]  # (Nd, Ng)
        E_matrix_meV = capacitance_data['E_matrix_meV'][0]  # (Nd, Nd)
        C_DD_inv = capacitance_data['C_DD_inv'][0]  # (Nd, Nd)
        
        # Deploy experiment
        qdarts_params = config['qdarts']
        system_matrices = {
            "C_tilde_DD": C_tilde_DD,
            "C_DG": C_DG,
            "tc_meV": tc_meV
        }
        
        capacitance_config = {
            "C_DD": np.abs(C_tilde_DD) * 1e18,  # Convert to attoFarads
            "C_Dg": np.abs(C_DG) * 1e18,
            "ks": None,
        }
        
        tunneling_config = {
            "tunnel_couplings": tc_meV * 1e-3,  # Convert to eV
            "temperature": qdarts_params["temperature"],
            "energy_range_factor": qdarts_params["energy_range_factor"],
        }
        
        sensor_config = {
            "sensor_dot_indices": [sensor_idx],
            "sensor_detunings": qdarts_params["sensor_detunings"],
            "noise_amplitude": qdarts_params["noise_amplitude"],
            "peak_width_multiplier": qdarts_params["peak_width_multiplier"],
        }
        
        experiment = Experiment(capacitance_config, tunneling_config, sensor_config)
        
        # Get base state and transition vectors
        cap_sim = experiment.capacitance_sim
        base_state_hint = np.zeros(cap_sim.num_dots, dtype=int)
        base_state_hint[sensor_idx] = 5
        base_state = cap_sim.find_state_of_voltage(np.zeros(cap_sim.num_dots), state_hint=base_state_hint)
        base_state[sensor_idx] = 5
        
        transition_vectors = get_virtual_gate_transitions(
            alpha=alpha,
            Nd=Nd,
            C_DD_inv=C_DD_inv,
            base_charge_state=base_state,
        )
        
        v_offset = -np.sum(transition_vectors, axis=0)  # (Ng,)
        
        # Get coulomb diamond sizes
        coulomb_diamond_sizes = get_coulomb_diamond_sizes(C_DG)
        
        # Generate cuts
        cuts = generate_cuts(Ndots, Nsensors, sensor_gate_idx)
        Ncuts = len(cuts)
        
        # Store full plane axes arrays
        cuts_axes = []  # Store full plane axes arrays
        # Store PNG data in memory first (if saving PNGs)
        save_png_images = config.get('CSD_generation', {}).get('save_png_images', False)
        png_figures = []  # Store figure objects if saving PNGs
        
        # Process each cut
        for cut_idx, cut_pair in enumerate(cuts):
            axes = plane_axes_from_pair(cut_pair, Ngates)
            cuts_axes.append(axes)  # Store full axes array
            span_x = coulomb_diamond_sizes[cut_pair[0]]
            span_y = coulomb_diamond_sizes[cut_pair[1]]
            
            # Generate voltage ranges
            x_voltages = np.linspace(-0.4 * span_x, n_diamonds_factor * span_x, resolution)
            y_voltages = np.linspace(-0.4 * span_y, n_diamonds_factor * span_y, resolution)
            
            # Generate CSD
            xout, yout, _, polytopes, sensor_values, _ = experiment.generate_CSD(
                plane_axes=axes,
                x_voltages=x_voltages,
                y_voltages=y_voltages,
                v_offset=v_offset,
                compute_polytopes=True,
                compensate_sensors=False,
                use_virtual_gates=False,
                use_sensor_signal=True,
            )
            
            # Create PNG figure in memory (don't save yet)
            if save_png_images:
                fig, ax = plt.subplots(figsize=(1.0, 1.0))
                ax.pcolormesh(
                    1e3 * xout - 1e3 * v_offset[cut_pair[0]],
                    1e3 * yout - 1e3 * v_offset[cut_pair[1]],
                    sensor_values[:, :, 0].T,
                    cmap='viridis'
                )
                #plot_polytopes(ax, polytopes, axes_rescale=1e3)
                #ax.set_xlabel(f'Gate {cut_pair[0]} Voltage (mV)')
                #ax.set_ylabel(f'Gate {cut_pair[1]} Voltage (mV)')
                #ax.set_title(f'Cut {cut_str}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')  # Remove axes completely
                ax.set_position([0, 0, 1, 1])  # Make axes fill entire figure
                png_figures.append((fig, cut_idx))
        
        # Store voltage parameters [x0, x1, N] for each cut
        # x_voltage and y_voltage should be (batch_size, Ncuts, 3) where 3 = [x0, x1, N]
        x_voltage_params = np.zeros((Ncuts, 3))  # [x0, x1, N] for each cut
        y_voltage_params = np.zeros((Ncuts, 3))  # [y0, y1, N] for each cut
        
        for cut_idx, cut_pair in enumerate(cuts):
            span_x = coulomb_diamond_sizes[cut_pair[0]]
            span_y = coulomb_diamond_sizes[cut_pair[1]]
            x_voltage_params[cut_idx] = [-0.4 * span_x, 2.1 * span_x, resolution]
            y_voltage_params[cut_idx] = [-0.4 * span_y, 2.1 * span_y, resolution]
        
        # Convert cuts_axes to numpy array: (Ncuts, 2, Ng)
        cuts_axes_array = np.array(cuts_axes)  # (Ncuts, 2, Ng)
        
        # Prepare data for saving
        # Add batch dimension (batch_size=1) to match requested format
        data_dict = {
            'C_tilde_DD': C_tilde_DD[np.newaxis, :, :],  # (1, Nd, Nd)
            'C_DG': C_DG[np.newaxis, :, :],  # (1, Nd, Ng)
            'geometry': geometry[np.newaxis, :, :],  # (1, Nd, 2)
            'tc_meV': tc_meV[np.newaxis, :, :],  # (1, Nd, Nd)
            'v_offset': v_offset[np.newaxis, :],  # (1, Ng)
            'x_voltage': x_voltage_params[np.newaxis, :, :],  # (1, Ncuts, 3) - [x0, x1, N] for each cut
            'y_voltage': y_voltage_params[np.newaxis, :, :],  # (1, Ncuts, 3) - [y0, y1, N] for each cut
            'alpha': alpha[np.newaxis, :, :],  # (1, Nd, Ng)
            'E_c': E_matrix_meV[np.newaxis, :, :],  # (1, Nd, Nd) - using E_matrix_meV as E_c
            'cuts': cuts_axes_array[np.newaxis, :, :, :],  # (1, Ncuts, 2, Ng) - full plane axes arrays
        }
        
        # Only create folder and save files after all computation is successful
        os.makedirs(datapoint_dir, exist_ok=True)
        
        # Save PNG files if they were created
        if save_png_images and png_figures:
            for fig, cut_idx in png_figures:
                try:
                    fig.savefig(os.path.join(datapoint_dir, f"cut_{cut_idx}.png"), dpi=int(resolution), pad_inches=0, bbox_inches='tight')
                finally:
                    plt.close(fig)
        
        # Save data as .npz file (includes cuts)
        np.savez(os.path.join(datapoint_dir, "data.npz"), **data_dict)
        
        print(f"Generated datapoint {datapoint_id:05d} with {Ncuts} cuts")
        return True
        
    except Exception as e:
        print(f"Error generating datapoint {datapoint_id:05d}: {e}")
        import traceback
        traceback.print_exc()
        # Clean up folder if it was created (shouldn't happen, but just in case)
        if os.path.exists(datapoint_dir):
            try:
                import shutil
                shutil.rmtree(datapoint_dir)
            except:
                pass
        return False

def print_keys_in_datapoint(path_to_file: str) -> dict:
    """
    Print all keys in a datapoint.
    """
    keys = list(np.load(path_to_file).keys())
    return keys

def load_all_data(path: str, load_images: bool = False, folder_name: str = "datapoint", file_name: str = "data.npz"):
    """
    Load all datapoints from subfolders under the given path.
    
    This function searches for all folders matching the pattern (e.g., "datapoint_*")
    and loads the data.npz file from each folder. Returns a dictionary where each key
    corresponds to a key in the npz file, and each value is a list containing the
    values from all loaded files.
    
    Parameters
    ----------
    path : str
        Base path to search for datapoint folders (e.g., "datasets/sys_3_1__2")
    load_images : bool, optional
        If True, loads images and returns them as a torch.Tensor of shape (N, num_branches, 1, H, W)
        If False, returns images as an empty list
    folder_name : str, optional
        Prefix of folder names to search for (default: "datapoint")
    file_name : str, optional
        Name of the npz file to load from each folder (default: "data.npz")
    
    Returns
    -------
    dict
        Dictionary with the same keys as in the npz files, where each value is
        a list of arrays/values from all loaded files.
    images : torch.Tensor or list
        If load_images=True: torch.Tensor of shape (N, num_branches, 1, H, W) containing all images
        If load_images=False: empty list
    missing_folders : list
        List of folders that were found but missing the data file
    failed_folders : list
        List of tuples (folder_path, error_message) for folders that failed to load
    """
    # Find all folders matching the pattern
    datapoint_folders = []
    missing_folders = []
    failed_folders = []
    total_folders = 0
    
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item.startswith(folder_name):
            total_folders += 1
            npz_path = os.path.join(item_path, file_name)
            if os.path.exists(npz_path):
                datapoint_folders.append(npz_path)
            else:
                missing_folders.append(item_path)
    
    if not datapoint_folders:
        raise ValueError(f"No datapoint folders with {file_name} found in {path}")
    
    # Report missing folders if any
    if missing_folders:
        print(f"Warning: {len(missing_folders)} datapoint folder(s) are missing {file_name}:")
        for missing_folder in sorted(missing_folders)[:10]:  # Show first 10
            try:
                folder_contents = os.listdir(missing_folder)
                if folder_contents:
                    print(f"  - {os.path.basename(missing_folder)}: contains {folder_contents}")
                else:
                    print(f"  - {os.path.basename(missing_folder)}: empty folder")
            except Exception as e:
                print(f"  - {os.path.basename(missing_folder)}: error accessing folder ({e})")
        if len(missing_folders) > 10:
            print(f"  ... and {len(missing_folders) - 10} more")
        print(f"Total folders found: {total_folders}, folders with {file_name}: {len(datapoint_folders)}, missing: {len(missing_folders)}")
    
    # Sort folders to ensure consistent ordering
    datapoint_folders.sort()
    # Load first file to get the keys
    first_data = np.load(datapoint_folders[0])
    keys = list(first_data.keys())
    
    # Initialize dictionary with empty lists for each key
    result_dict = {key: [] for key in keys}
    images = []
    
    # Load all files and collect values with error handling
    for npz_path in datapoint_folders:
        try:
            data = np.load(npz_path)
            for key in keys:
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in {npz_path}")
                result_dict[key].append(data[key])
        except Exception as e:
            failed_folders.append((npz_path, str(e)))
            print(f"Warning: Failed to load {npz_path}: {e}")
            continue

        if load_images:
            folder_path = os.path.dirname(npz_path)  # Use os.path.dirname for cross-platform compatibility
            try:
                num_cuts = np.array(data['cuts'][0]).squeeze().shape[0]
                trio = []
                for cut_idx in range(num_cuts):
                    img_path = os.path.join(folder_path, f"cut_{cut_idx}.png")
                    img = Image.open(img_path)
                    img_tensor = img_to_transform_tensor(img)  # (1, H, W) FloatTensor in [0,1], single channel
                    trio.append(img_tensor)
                images.append(trio)
            except Exception as e:
                for key in keys:
                    result_dict[key].pop()  # Remove the last added item    
                failed_folders.append((npz_path, str(e)))
                print(f"Warning: Failed to load images from {npz_path}: {e}")
    
    # Report summary
    print(f"Loading {len(datapoint_folders)} datapoints from {path}.")
    if len(datapoint_folders) > 0:
        print(f"First file: {datapoint_folders[0]}, Last file: {datapoint_folders[-1]}")
    if failed_folders:
        print(f"Warning: {len(failed_folders)} file(s) failed to load. Successfully loaded: {len(result_dict[keys[0]])} datapoints")
    
    # Convert images from list of lists to tensor if load_images is True
    if load_images and images:
        print(f"Starting to load and preprocess images...")
        # Convert imgs: list of lists of tensors -> tensor of shape (N, num_branches, 1, H, W)
        img_tensors = []
        for datapoint_imgs in images:
            # Stack the num_branches images for this datapoint: (num_branches, 1, H, W)
            stacked = torch.stack(datapoint_imgs, dim=0)
            img_tensors.append(stacked)
        images_tensor = torch.stack(img_tensors, dim=0)  # (N, num_branches, 1, H, W)
        return result_dict, images_tensor, missing_folders, failed_folders
    
   
    # Validation before returning
    if load_images and images:
        # Validate alignment
        num_data_points = len(result_dict[keys[0]])
        num_images = len(images)
        if num_data_points != num_images:
            raise ValueError(
                f"Data/Image mismatch: {num_data_points} data points but {num_images} image sets. "
                f"This indicates some datapoints failed to load images. Please check failed_folders."
            )
    return result_dict, images, missing_folders, failed_folders

def img_to_transform_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Apply the standard preprocessing (grayscale + ToTensor) to a given image.

    Parameters
    ----------
    img :
        Input image. Can be a PIL.Image.Image or a NumPy array with shape (H, W) or (H, W, C).
    as_tensor : bool, optional
        Whether to return a torch.FloatTensor (default: True). If False, returns a numpy array.

    Returns
    -------
    torch.Tensor or np.ndarray
        If as_tensor=True: torch.FloatTensor (C, H, W), normalized to [0, 1].
        If as_tensor=False: numpy array (H, W), normalized to [0, 1].
    """
    # Ensure we have a PIL image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel
        transforms.ToTensor(),                        # -> [C, H, W] in [0, 1]
    ])

    img_tensor = img_transform(img)  # (1, H, W) FloatTensor in [0,1], single channel

    return img_tensor

def visualize_image(img: torch.Tensor) -> None:
    img_np = img.squeeze(0).cpu().numpy()   # remove channel dim

    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    plt.show()


def create_context(data: dict) -> torch.Tensor:
    """
    Create context vectors as torch tensors.
    
    Args:
        data: Dictionary containing 'v_offset', 'x_voltage', 'y_voltage', 'cuts'
    
    Returns:
        torch.Tensor of shape (N, context_vector_size) - context vectors
        Takes the first context vector from each datapoint (assuming same context for all branches)
    """    
    # shape: (num_realizations, num_cuts, 9), 9 = [v_off, x_volts, y_volts]
    context = []

    for v_off, x_volts, y_volts, cuts in zip(data['v_offset'], data['x_voltage'], data['y_voltage'], data['cuts']):
        v_off = v_off.squeeze()[:-1]
        y_volts = y_volts.squeeze()
        x_volts = x_volts.squeeze()
        cuts = cuts.squeeze()

        inner_list = [] # contain 3 lists: v_off, x_volts_cut_i, y_volts_cut_i
        for cut in range(x_volts.shape[0]):
                inner_list.append(np.concatenate([v_off, x_volts[cut], y_volts[cut], cuts[cut]], axis=None))
                
        context.append(inner_list)
    
    # Take the first context vector from each datapoint (assuming same context for all branches)
    context_list = [ctx[0] for ctx in context]  # Take first context vector per datapoint
    context_tensor = torch.FloatTensor(np.array(context_list))  # (N, context_vector_size)
    
    return context_tensor

def create_outputs(data: dict) -> torch.Tensor:
    '''
    Function to create the outputs for the model as torch tensors.
    As the multiplication of inverse of lever arm matrix and the interaction matrix
    
    Args:
        data: Dictionary containing 'alpha' and 'E_c'
    
    Returns:
        torch.Tensor of shape (N, output_size) - flattened output matrices

    '''
    alpha = data['alpha']
    e_c = data['E_c']

    outputs = [np.matmul( np.linalg.inv(alpha), e_c).squeeze()[:-1, :-1] for alpha, e_c in zip(alpha, e_c)]
    
    # Flatten each output if it's 2D and convert to torch tensor
    outputs_list = []
    for out in outputs:
        if isinstance(out, np.ndarray):
            outputs_list.append(out.flatten())
        else:
            outputs_list.append(np.array(out).flatten())
    outputs_tensor = torch.FloatTensor(np.array(outputs_list))  # (N, output_size)
    
    return outputs_tensor


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory : str
        Path to the directory to ensure exists
    """
    if directory:  # Only create if directory path is not empty
        os.makedirs(directory, exist_ok=True)

