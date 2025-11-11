from __future__ import annotations

"""Utility classes for stochastic quantum-dot device generation."""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import inv

__all__ = ["QuantumDotModel", "PhysicalConstants", "e", "meV", "kb"]


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants (SI units) used by the data generator."""

    elementary_charge: float = 1.602176634e-19  # Coulomb
    milli_electron_volt: float = 1.602176634e-22  # Joule (e * 1e-3)
    boltzmann: float = 1.380649e-23  # Joule / Kelvin


CONSTS = PhysicalConstants()

# Backwards compatibility aliases
e = CONSTS.elementary_charge
meV = CONSTS.milli_electron_volt
kb = CONSTS.boltzmann


class QuantumDotModel:
    """Randomly generated multi-dot device with derived circuit parameters.

    Workflow
    --------
    1. Sample the charging energy matrix :math:`E` (Ec, Em) → derive ``C_DD_inv`` and ``C_DD``.
    2. Sample the canonical gate coupling matrix ``~C_DG`` with positive entries.
    3. Apply the device constraints (decouple sensor gate, fix sensor parameters).
    4. Set Maxwell ``C_DG = ~C_DG`` and compute the lever arm matrix ``alpha = C_DD_inv @ C_DG``.
    5. Derive a tunnel-coupling matrix correlated with ``C_tilde_DD``.

    Parameters
    ----------
    Nd:
        Number of dots (including the sensor).
    Ng:
        Number of gates (including the sensor gate).
    params:
        Dictionary with the statistical sampling parameters. The expected keys
        correspond to those used in the original notebook (``Ec_mean``, ``Ec_std``, …).
    kT_meV:
        Thermal energy in milli-electron volts.
    """

    def __init__(self, Nd: int, Ng: int, params: dict[str, float], kT_meV: float = 0.1):
        self.e = CONSTS.elementary_charge
        self.kT = kT_meV * CONSTS.milli_electron_volt / self.e  # convert to Volts (kT/e)
        self.Nd = Nd
        self.Ng = Ng
        self.params = params
        self.sensor_idx = params["sensor_idx"]
        self.sensor_gate_idx = params["sensor_gate_idx"]
        self.sensor_gate_voltage = 0.0

        self._generate_from_physics()
        self._set_sensor_bias(params["sensor_gate_tune_kT"])

    def _generate_from_physics(self) -> None:
        """Sample random device parameters and derive circuit matrices."""

        # 1. Charging-energy matrix in meV
        E_matrix_meV = np.zeros((self.Nd, self.Nd))
        sensor_idx = self.sensor_idx

        ec_diag = np.random.normal(
            self.params["Ec_mean"],
            self.params["Ec_std"],
            self.Nd,
        )
        ec_diag[sensor_idx] = self.params["Ec_sensor_fixed"]
        np.fill_diagonal(E_matrix_meV, ec_diag)

        for i in range(self.Nd):
            for j in range(i + 1, self.Nd):
                if i == sensor_idx or j == sensor_idx:
                    em_ij = np.random.normal(
                        self.params["Em_sensor_sys_mean"],
                        self.params["Em_sensor_sys_std"],
                    )
                else:
                    em_ij = np.random.normal(
                        self.params["Em_mean"],
                        self.params["Em_std"],
                    )
                E_matrix_meV[i, j] = E_matrix_meV[j, i] = max(0.0, em_ij)

        # 2. Canonical gate-capacitance matrix (Farads)
        C_tilde_DG = np.zeros((self.Nd, self.Ng))
        sensor_gate_idx = self.sensor_gate_idx

        diag_len = min(self.Nd, self.Ng)
        C_dg_diag = np.random.normal(
            self.params["C_dg_diag_mean"],
            self.params["C_dg_diag_std"],
            diag_len,
        )

        for i in range(self.Nd):
            for j in range(self.Ng):
                if i == j and j < diag_len:
                    C_tilde_DG[i, j] = max(0.0, C_dg_diag[i])
                else:
                    val = np.random.normal(
                        self.params["C_dg_cross_mean"],
                        self.params["C_dg_cross_std"],
                    )
                    C_tilde_DG[i, j] = max(0.0, val)

        # 3. Enforce sensor constraints
        C_tilde_DG[sensor_idx, sensor_gate_idx] = self.params["C_dg_sensor_gate_fixed"]
        C_tilde_DG[sensor_idx, :sensor_gate_idx] = 0.0
        C_tilde_DG[sensor_idx, sensor_gate_idx + 1 :] = 0.0
        C_tilde_DG[:sensor_idx, sensor_gate_idx] = 0.0
        C_tilde_DG[sensor_idx + 1 :, sensor_gate_idx] = 0.0

        # 4. Derive Maxwell matrices and lever arms
        self.E_matrix_meV = E_matrix_meV
        self.Ec_meV = np.diag(self.E_matrix_meV)
        self.C_tilde_DG = C_tilde_DG
        self.C_DG = self.C_tilde_DG.copy()

        E_matrix_J = self.E_matrix_meV * CONSTS.milli_electron_volt
        self.C_DD_inv = (2.0 / (self.e**2)) * E_matrix_J

        try:
            self.C_DD = inv(self.C_DD_inv)
        except np.linalg.LinAlgError:
            self.C_DD = np.eye(self.Nd)
            self.alpha = np.eye(self.Nd, self.Ng)
            return

        self.alpha = self.C_DD_inv @ self.C_DG

        # 5. Canonical C_DD and derived tunnel couplings
        self.C_tilde_DD = -self.C_DD.copy()
        np.fill_diagonal(self.C_tilde_DD, np.sum(self.C_DD, axis=1))

        self.tc_meV = np.zeros((self.Nd, self.Nd))
        tc_slope = self.params["tc_C_tilde_DD_slope"]
        tc_std = self.params["tc_std"]
        for i in range(self.Nd):
            for j in range(i + 1, self.Nd):
                c_tilde_ij = self.C_tilde_DD[i, j]
                tc_ij = tc_slope * c_tilde_ij + np.random.normal(0.0, tc_std)
                self.tc_meV[i, j] = self.tc_meV[j, i] = max(0.0, tc_ij)

    def _set_sensor_bias(self, sensor_gate_tune_kT: float) -> None:
        """Bias the sensor gate to sit on a Coulomb peak flank."""

        V_target = sensor_gate_tune_kT * self.kT
        alpha_ss = self.alpha[self.sensor_idx, self.sensor_gate_idx]
        if np.abs(alpha_ss) < 1.0e-12:
            self.sensor_gate_voltage = 0.0
            return
        self.sensor_gate_voltage = -V_target / alpha_ss

    # ---------------------------------------------------------------------
    # Convenience helpers reused in the notebook
    # ---------------------------------------------------------------------

    def get_gibbs_energy(self, N_D: np.ndarray, V_G: np.ndarray) -> float:
        """Return the Gibbs energy for a dot configuration and gate voltages."""

        N_D = np.asarray(N_D)
        V_G = np.asarray(V_G)
        term1 = 0.5 * (self.e**2) * (N_D.T @ self.C_DD_inv @ N_D)
        term2 = self.e * (N_D.T @ self.alpha @ V_G)
        return term1 + term2

    def find_ground_state(
        self,
        V_G: np.ndarray,
        N_search_space: list[list[int]] | np.ndarray,
    ) -> np.ndarray:
        """Brute-force search for the ground state configuration."""

        min_G = np.inf
        N_D_min = None
        for candidate in itertools.product(*N_search_space):
            G = self.get_gibbs_energy(candidate, V_G)
            if G < min_G:
                min_G = G
                N_D_min = candidate
        return np.array(N_D_min)

    def get_sensor_voltage(self, N_D: np.ndarray, V_G: np.ndarray) -> float:
        """Voltage at the sensor dot."""

        N_D = np.asarray(N_D)
        V_G = np.asarray(V_G)
        V_D = (self.e * self.C_DD_inv @ N_D) - (self.alpha @ V_G)
        return V_D[self.sensor_idx]

    def get_sensor_conductance(self, V_sensor: float) -> float:
        """Sensor conductance with thermal broadening."""

        arg = V_sensor / (2 * self.kT)
        if np.abs(arg) > 30:
            return 0.0
        return 1.0 / np.cosh(arg) ** 2

    def generate_csd_data(
        self,
        gate_x_idx: int,
        gate_y_idx: int,
        v_range: tuple[float, float],
        resolution: int,
    ):
        """Generate a capacitance stability diagram on a 2D voltage grid."""

        v_vec = np.linspace(v_range[0], v_range[1], resolution)
        V_x, V_y = np.meshgrid(v_vec, v_vec)

        N_ground_state = np.zeros((resolution, resolution, self.Nd), dtype=int)
        sensor_signal = np.zeros((resolution, resolution))

        V_G_base = np.zeros(self.Ng)
        V_G_base[self.sensor_gate_idx] = self.sensor_gate_voltage

        system_search = [0, 1, 2, 3]
        sensor_search = [0, 1, 2]
        N_search_space = [
            sensor_search if i == self.sensor_idx else system_search
            for i in range(self.Nd)
        ]

        for i in range(resolution):
            for j in range(resolution):
                V_G = V_G_base.copy()
                V_G[gate_x_idx] = V_x[i, j]
                V_G[gate_y_idx] = V_y[i, j]

                N_D_min = self.find_ground_state(V_G, N_search_space)
                N_ground_state[i, j, :] = N_D_min

                V_S = self.get_sensor_voltage(N_D_min, V_G)
                sensor_signal[i, j] = self.get_sensor_conductance(V_S)

        return V_x, V_y, N_ground_state, sensor_signal

    def get_virtual_gate_transitions(
        self,
        base_charge_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return gate-voltage vectors that realise 0→1 transitions per dot."""

        if base_charge_state is None:
            base_charge_state = np.zeros(self.Nd, dtype=int)
        else:
            base_charge_state = np.asarray(base_charge_state)
            if len(base_charge_state) != self.Nd:
                raise ValueError("base_charge_state must match the number of dots.")

        try:
            alpha_inv = inv(self.alpha)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Alpha matrix is singular; transitions are undefined.") from exc

        transition_voltages = np.zeros((self.Nd, self.Nd))
        for i in range(self.Nd):
            v_i_full = alpha_inv[:, i]
            term1_V = -self.e * (self.C_DD_inv[i, :] @ base_charge_state)
            term2_V = -(self.e / 2.0) * self.C_DD_inv[i, i]
            V_i_scalar = term1_V + term2_V
            transition_voltages[i] = v_i_full * V_i_scalar

        return transition_voltages


    def get_coulomb_diamond_sizes(self):
        C_DD_inv_diag = np.diag(self.C_DD_inv)
        # We use .copy() to create a writable array, not a read-only view
        alpha_diag = np.diag(self.alpha).copy()
        
        # Avoid division by zero if alpha is zero
        # (e.g., for a dot with no gate)
        alpha_diag[np.abs(alpha_diag) < 1e-15] = 1e-15
        
        # Charging energy (in Volts) = |e| * C_DD_inv[i,i]
        # This is the energy (in Joules) divided by 'e'
        charging_energy_V = self.e * C_DD_inv_diag
        
        # Diamond size (Volts) = Charging Energy (V) / Lever Arm (unitless)
        delta_V_o = charging_energy_V / alpha_diag
        
        return delta_V_o
