from __future__ import annotations

"""Utility classes for stochastic quantum-dot device generation."""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import inv



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