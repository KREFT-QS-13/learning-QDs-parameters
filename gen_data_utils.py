from __future__ import annotations

"""Utility classes for stochastic quantum-dot device generation."""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import inv



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
        base_geometry: np.ndarray = None,
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
        base_geometry : np.ndarray, optional
            Base positions array of shape (Nd, 2). If provided, adds variability
            around these positions instead of generating randomly.
            
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
        
        # Variability parameters for base geometry (can be added to params if needed)
        geometry_noise_std = self.params.get("geometry_noise_std_nm", 5.0) * 1e-9  # Default 5nm std
        
        # Pre-allocate output arrays
        all_coords = np.zeros((Nconfigurations, Nd, 2))
        all_distances = np.zeros((Nconfigurations, Nd, Nd))
        n_collected = 0
        
        # Keep generating batches until we have enough valid geometries
        while n_collected < Nconfigurations:
            # Generate a batch
            coords = np.zeros((batch_size, Nd, 2))
            if base_geometry is not None:
                # Use base geometry with added variability
                base_geom_array = np.asarray(base_geometry)
                if base_geom_array.shape != (Nd, 2):
                    raise ValueError(f"base_geometry must have shape (Nd, 2) = ({Nd}, 2), got {base_geom_array.shape}")
                
                # Broadcast base_geometry to batch_size and add Gaussian noise
                # Shape: (batch_size, Nd, 2)
                coords = base_geom_array[np.newaxis, :, :] + np.random.normal(
                    0.0, 
                    geometry_noise_std, 
                    size=(batch_size, Nd, 2)
                )
            else:
                # Original random generation code
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
                    mean = C_m_sq_mean*(d_nn_proxy / dist_ij)**3  # (batch_size,) in Farads
                    std_dev = C_m_sq_std * (d_nn_proxy / dist_ij)**3 # (batch_size,) in Farads
                
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
            C_DD_inv[b] = inv(C_DD[b])
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
    base_charge_state: np.ndarray | None = None,
) -> np.ndarray:
    """Return gate-voltage vectors that realise 0→1 transitions per dot."""
    if base_charge_state is None:
        base_charge_state = np.zeros( Nd, dtype=int)
    else:
        base_charge_state = np.asarray(base_charge_state)
        if len(base_charge_state) != Nd:
            raise ValueError("base_charge_state must match the number of dots.")

    try:
        alpha_inv = inv(alpha)
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