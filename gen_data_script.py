#!/usr/bin/env python3
"""
Generate quantum dot data based on configuration file.

This script loads configuration from config_data_gen.json and generates
100 data points, each with unique ID and folder containing:
- PNG files for each cut
- Data arrays (C_tilde_DD, C_DG, geometry, tc_meV, v_offset, x_voltage, y_voltage, alpha, E_c)
"""

import json
import os
import numpy as np
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from gen_data_utils import QuantumDotModel, get_virtual_gate_transitions, get_coulomb_diamond_sizes
from qdarts.experiment import Experiment
from qdarts.plotting import plot_polytopes

# Physical constants
e = 1.602176634e-19  # Coulomb (elementary charge)


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
    # Convert config to params
    params = config_to_params(config)
    
    # Create output directory for this data point
    datapoint_dir = Path(output_dir) / f"datapoint_{datapoint_id:05d}"
    datapoint_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # Save PNG with cut index as filename
            fig, ax = plt.subplots(figsize=(8, 8))
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
            plt.tight_layout()
            plt.savefig(datapoint_dir / f"cut_{cut_idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
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
        
        # Save data as .npz file (includes cuts)
        np.savez(datapoint_dir / "data.npz", **data_dict)
        
        print(f"Generated datapoint {datapoint_id:05d} with {Ncuts} cuts")
        return True
        
    except Exception as e:
        print(f"Error generating datapoint {datapoint_id:05d}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to generate all data points."""
    # Load configuration
    config_path = "config_data_gen.json"
    config = load_config(config_path)
    
    # Output directory
    output_dir = "generated_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of data points to generate
    num_datapoints = 100
    
    print(f"Generating {num_datapoints} data points...")
    print(f"Output directory: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i in range(num_datapoints):
        success = generate_datapoint(config, i, output_dir)
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nGeneration complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()

