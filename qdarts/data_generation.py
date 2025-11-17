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
        Ng = self.Ngates
        d_nn_proxy = self.params.get("d_mean_nm", 100.0) * 1e-9
        
        # Get sensor indices from params
        sensor_idx = self.params.get("sensor_idx", Nd - 1)
        sensor_gate_idx = self.params.get("sensor_gate_idx", Nd - 1)
        
        # --- 1. Model Gate Capacitance (C_tilde_DG) ---
        C_tilde_DG = np.zeros((batch_size, Nd, Ng))
        k_g = self.params["C_dg_cross_mean"] * d_nn_proxy
        
        # Diagonal elements: Dot_i to its own Gate_i
        diag_values = np.random.normal(
            self.params["C_dg_diag_mean"],
            self.params["C_dg_diag_std"],
            size=(batch_size, Nd)
        )
        diag_values = np.maximum(0.0, diag_values)
        
        # Set diagonal elements
        for i in range(Nd):
            C_tilde_DG[:, i, i] = diag_values[:, i]
        
        # Cross-talk elements: Dot_i to Gate_j (where i != j)
        # Use broadcasting: (batch_size, Nd, Ng) where we need to avoid diagonal
        # Create masks for off-diagonal elements
        i_indices, j_indices = np.meshgrid(np.arange(Nd), np.arange(Ng), indexing='ij')
        off_diag_mask = (i_indices != j_indices)
        
        # For off-diagonal: use distance from dot i to dot j (assuming gate j is near dot j)
        # Extract distances: dot_distances_batch[:, i, j] gives distance from dot i to dot j
        for i in range(Nd):
            for j in range(Ng):
                if i != j:
                    # Use distance from dot i to dot j as proxy for dot-gate distance
                    dist_ij = dot_distances_batch[:, i, j]  # (batch_size,)
                    mean = k_g / dist_ij  # (batch_size,)
                    std_dev = self.params["C_dg_cross_std"] * (d_nn_proxy / dist_ij)  # (batch_size,)
                    C_tilde_DG[:, i, j] = np.maximum(0.0, np.random.normal(mean, std_dev))
        
        # --- 2. Enforce Sensor Constraints ---
        if "C_dg_sensor_gate_fixed" in self.params:
            C_tilde_DG[:, sensor_idx, sensor_gate_idx] = self.params["C_dg_sensor_gate_fixed"]

            C_DG = C_tilde_DG.copy()
        
        # --- 3. Model Dot-Dot Mutual Capacitance (C_m) ---
        C_m = np.zeros((batch_size, Nd, Nd))
        C_m_nn_mean_proxy = self.params.get("C_m_nn_mean", self.params["C_dg_cross_mean"] * 3.0)
        C_m_nn_std_proxy = self.params.get("C_m_nn_std", self.params["C_dg_cross_std"] * 3.0)
        k_m = C_m_nn_mean_proxy * d_nn_proxy
        
        # Vectorized computation for all pairs (i, j) where i < j
        for i in range(Nd):
            for j in range(i + 1, Nd):
                dist_ij = dot_distances_batch[:, i, j]  # (batch_size,)
                mean = k_m / dist_ij  # (batch_size,)
                std_dev = C_m_nn_std_proxy * (d_nn_proxy / dist_ij)  # (batch_size,)
                values = np.maximum(0.0, np.random.normal(mean, std_dev))
                C_m[:, i, j] = values
                C_m[:, j, i] = values  # Symmetric
        
        # --- 4. Build the Full C_DD Matrix (Maxwell Matrix) ---
        C_0_mean = self.params.get("C_0_mean", 20.0e-18)  # Default: 20 aF
        C_0_std = self.params.get("C_0_std", 10.0e-18)     # Default: 10 aF

        # Generate substrate capacitance for each dot in each geometry
        C_0 = np.random.normal(C_0_mean, C_0_std, size=(batch_size, Nd))
        C_0 = np.maximum(0.0, C_0)  # Ensure non-negative

        # --- 4. Build the Full C_DD Matrix (Maxwell Matrix) ---
        C_DD = np.zeros((batch_size, Nd, Nd))
        C_cap_to_all_gates = np.sum(C_DG, axis=2)  # (batch_size, Nd)

        # Diagonal: sum of all gate capacitances + sum of mutual capacitances
        for i in range(Nd):
            C_DD[:, i, i] = C_cap_to_all_gates[:, i] + np.sum(C_m[:, i, :], axis=1) + C_0[:, i]
        
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
        
        # --- 7. Derive Tunnel Couplings (tc) with WKB exponential decay ---
        tc_meV = np.zeros((batch_size, Nd, Nd))
        tc_slope = self.params["tc_C_tilde_DD_slope"]
        tc_std_frac = self.params["tc_std"]

        # WKB exponential decay parameters
        # decay_length_nm: characteristic decay length in nanometers (typical: 10-50 nm)
        # This controls how fast tunneling decays with distance
        decay_length_nm = self.params.get("tc_decay_length_nm", 20.0)  # Default: 20 nm
        decay_length_m = decay_length_nm * 1e-9  # Convert to meters

        # Reference distance for normalization (typically the mean nearest-neighbor distance)
        d_ref = self.params.get("d_mean_nm", 100.0) * 1e-9  # meters

        for i in range(Nd):
            for j in range(i + 1, Nd):
                # Get distance between dots
                dist_ij = dot_distances_batch[:, i, j]  # (batch_size,) in meters
                
                # Base tunnel coupling based on mutual capacitance C_m
                # (using C_m instead of C_tilde_DD as requested)
                c_m_ij = C_m[:, i, j]  # (batch_size,)
                base_tc = tc_slope * c_m_ij  # (batch_size,)
                
                # Apply WKB exponential decay: exp(-(distance - d_ref) / decay_length)
                # This gives exponential penalty for displacement beyond reference distance
                # For distances < d_ref, we get enhancement (exp > 1)
                # For distances > d_ref, we get suppression (exp < 1)
                distance_penalty = np.exp(-(dist_ij - d_ref) / decay_length_m)  # (batch_size,)
                
                # Mean tunnel coupling with exponential decay
                mean_tc = base_tc * distance_penalty  # (batch_size,)
                std_tc = mean_tc * tc_std_frac  # (batch_size,)
                
                # Generate values (always non-negative)
                tc_values = np.maximum(0.0, np.random.normal(mean_tc, std_tc))
                tc_values = np.minimum(self.params["tc_max_meV"], tc_values)
                
                tc_meV[:, i, j] = tc_values
                tc_meV[:, j, i] = tc_values  # Symmetric
            
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
            'alpha': alpha
        }