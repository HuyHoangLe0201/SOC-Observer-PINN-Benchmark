"""
Electro-Thermal-Aging 2nd-Order Dual Polarization (2-RC) Battery Cell Models
Journal of Energy Storage (Elsevier)

Single Source of Truth synchronized strictly with Table 10 (Appendix C.1) and Eqs. (1)-(22):
1. BatteryCellNMC: 2.00 Ah NMC-class cell (3-state [SOC, U1, U2], strictly monotonic 5th-order OCV).
2. BatteryCellLFP: 2.30 Ah LFP-class cell (4-state [SOC, U1, U2, V_h], flat plateau + ZOH dynamic hysteresis).

Both models incorporate:
- Piecewise cold/warm Arrhenius scaling of ohmic, polarization, and usable capacity parameters.
- Semi-empirical SOH degradation multipliers matching Table 10.
- Lumped electro-thermal energy balance ODE (C_th * dT/dt = Q_gen - (T - T_amb) / R_th).
"""

import numpy as np


class BatteryCellNMC:
    """
    Nickel Manganese Cobalt (NMC) 2-RC Equivalent Circuit Model calibrated to Table 10.
    Nominal Capacity: 2.00 Ah, OCV range: [3.212 V, 4.197 V].
    """

    def __init__(
        self,
        nominal_capacity_ah=2.00,
        soh=1.0,
        temperature_c=25.0,
        dt=1.0,
        thermal_coupled=False,
    ):
        self.chemistry = "NMC"
        self.q_nominal = float(nominal_capacity_ah)
        self.soh = float(soh)
        self.temperature_c = float(temperature_c)
        self.t_amb_c = float(temperature_c)
        self.dt = float(dt)
        self.thermal_coupled = bool(thermal_coupled)

        # Baseline fresh-cell ECM parameters at T_ref = 25 degC (Table 10)
        self.r0_base = 0.035    # Ohm (35.0 mOhm)
        self.r1_base = 0.015    # Ohm (15.0 mOhm), tau_1 = 24.0 s
        self.c1_base = 1600.0   # F
        self.r2_base = 0.025    # Ohm (25.0 mOhm), tau_2 = 112.5 s
        self.c2_base = 4500.0   # F

        # Piecewise Arrhenius constants (Appendix C.1)
        self.gas_constant = 8.314
        self.t_ref_k = 298.15
        self.ea_r0_cold = 20500.0
        self.ea_r0_hot = 14100.0
        self.ea_r1_cold = 28400.0
        self.ea_r1_hot = 16900.0
        self.ea_r2 = 28400.0
        self.ea_q = 3200.0
        self.kappa_t_hot = 0.0015

        # Semi-empirical SOH degradation multipliers (Table 10 / Appendix C.1)
        self.kappa_r0 = 2.50
        self.kappa_r1 = 7.50
        self.kappa_r2 = 4.00
        self.gamma_c1 = 1.25
        self.gamma_c2 = 0.78

        # Lumped thermal parameters (Appendix C.1)
        self.c_th = 35.8  # J/K
        self.r_th = 12.5  # K/W

        # Asymmetric Faradaic coulombic efficiency
        self.eta_charge = 0.995
        self.eta_discharge = 1.000

        # Strictly monotonic 5th-order polynomial OCV coefficients (z^5 -> z^0)
        self.ocv_coeffs = np.array([9.20379162, -28.84109993, 33.37645360, -17.41856030, 4.67403064, 3.21181343])

        # Internal state variables: [SOC, U1, U2]
        self.soc = 1.0
        self.up1 = 0.0
        self.up2 = 0.0

        self._update_parameters()

    def _f_t(self, temp_c, ea_cold, ea_hot):
        t_k = temp_c + 273.15
        ea = ea_cold if t_k < self.t_ref_k else ea_hot
        return float(np.exp((ea / self.gas_constant) * (1.0 / t_k - 1.0 / self.t_ref_k)))

    def _f_q(self, temp_c):
        t_k = temp_c + 273.15
        if t_k < self.t_ref_k:
            return float(np.exp(-(self.ea_q / self.gas_constant) * (1.0 / t_k - 1.0 / self.t_ref_k)))
        return float(1.0 + self.kappa_t_hot * (t_k - self.t_ref_k))

    def _update_parameters(self):
        deg = max(0.0, 1.0 - self.soh)
        ft0 = self._f_t(self.temperature_c, self.ea_r0_cold, self.ea_r0_hot)
        ft1 = self._f_t(self.temperature_c, self.ea_r1_cold, self.ea_r1_hot)
        ft2 = self._f_t(self.temperature_c, self.ea_r2, self.ea_r2)

        self.r0 = self.r0_base * ft0 * (1.0 + self.kappa_r0 * deg)
        self.r1 = self.r1_base * ft1 * (1.0 + self.kappa_r1 * deg)
        self.r2 = self.r2_base * ft2 * (1.0 + self.kappa_r2 * deg)
        self.c1 = max(100.0, self.c1_base * (1.0 - self.gamma_c1 * deg))
        self.c2 = max(100.0, self.c2_base * (1.0 - self.gamma_c2 * deg))

        self.q_actual = self.q_nominal * self.soh * self._f_q(self.temperature_c)
        self.tau1 = self.r1 * self.c1
        self.tau2 = self.r2 * self.c2

    def reset_state(self, initial_soc=1.0, temperature_c=None):
        self.soc = float(np.clip(initial_soc, 0.0, 1.0))
        self.up1 = 0.0
        self.up2 = 0.0
        if temperature_c is not None:
            self.temperature_c = float(temperature_c)
            self.t_amb_c = float(temperature_c)
        self._update_parameters()

    def get_ocv(self, soc):
        s = np.clip(soc, 0.0, 1.0)
        return float(np.polyval(self.ocv_coeffs, s))

    def get_docv_dsoc(self, soc):
        s = np.clip(soc, 0.0, 1.0)
        deriv_coeffs = np.polyder(self.ocv_coeffs)
        return float(max(0.05, np.polyval(deriv_coeffs, s)))

    def get_entropic_coeff(self, soc):
        # Negative over most of discharge, inverting sign above z > 0.75 (Appendix C.1)
        if soc > 0.75:
            return +0.08e-3
        return -0.065e-3

    def get_eta(self, current_a):
        return self.eta_discharge if current_a >= 0.0 else self.eta_charge

    def step(self, current_a):
        """
        Advance 2-RC discrete dynamics and optional lumped thermal ODE by dt.
        Positive current_a = discharge; negative = charge.
        Returns: (v_terminal, soc)
        """
        eta = self.get_eta(current_a)
        d_soc = (-eta * current_a * self.dt) / (3600.0 * self.q_actual)
        self.soc = float(np.clip(self.soc + d_soc, 0.0, 1.0))

        exp1 = np.exp(-self.dt / self.tau1)
        exp2 = np.exp(-self.dt / self.tau2)
        self.up1 = self.up1 * exp1 + current_a * self.r1 * (1.0 - exp1)
        self.up2 = self.up2 * exp2 + current_a * self.r2 * (1.0 - exp2)

        if self.thermal_coupled:
            t_k = self.temperature_c + 273.15
            q_irrev = (current_a ** 2) * self.r0 + (self.up1 ** 2) / self.r1 + (self.up2 ** 2) / self.r2
            q_rev = -current_a * t_k * self.get_entropic_coeff(self.soc)
            q_gen = q_irrev + q_rev
            dT = (q_gen - (self.temperature_c - self.t_amb_c) / self.r_th) * (self.dt / self.c_th)
            self.temperature_c += float(dT)
            self._update_parameters()

        ocv = self.get_ocv(self.soc)
        v_terminal = ocv - current_a * self.r0 - self.up1 - self.up2
        return float(v_terminal), float(self.soc)


class BatteryCellLFP:
    """
    LiFePO4 (LFP) 4-State Equivalent Circuit Model calibrated to Table 10.
    Nominal/Calibrated Capacity: 2.30 Ah, OCV range: [2.000 V, 3.600 V].
    Includes strictly monotonic analytical LFP OCV and ZOH dynamic hysteresis Eq. (18).
    """

    def __init__(
        self,
        nominal_capacity_ah=2.30,
        soh=1.0,
        temperature_c=25.0,
        dt=1.0,
        thermal_coupled=False,
    ):
        self.chemistry = "LFP"
        self.q_nominal = float(nominal_capacity_ah)
        self.soh = float(soh)
        self.temperature_c = float(temperature_c)
        self.t_amb_c = float(temperature_c)
        self.dt = float(dt)
        self.thermal_coupled = bool(thermal_coupled)

        # Baseline fresh-cell ECM parameters at T_ref = 25 degC (Table 10)
        self.r0_base = 0.022    # Ohm (22.0 mOhm)
        self.r1_base = 0.012    # Ohm (12.0 mOhm), tau_1 = 37.5 s
        self.c1_base = 3125.0   # F
        self.r2_base = 0.018    # Ohm (18.0 mOhm), tau_2 = 144.0 s
        self.c2_base = 8000.0   # F

        # Piecewise Arrhenius constants matching Appendix C.1
        self.gas_constant = 8.314
        self.t_ref_k = 298.15
        self.ea_r0_cold = 20500.0
        self.ea_r0_hot = 14100.0
        self.ea_r1_cold = 28400.0
        self.ea_r1_hot = 16900.0
        self.ea_r2 = 28400.0
        self.ea_q = 3200.0
        self.kappa_t_hot = 0.0015

        # Semi-empirical SOH degradation multipliers (Table 10 / Appendix C.1)
        self.kappa_r0 = 2.50
        self.kappa_r1 = 7.50
        self.kappa_r2 = 5.00
        self.gamma_c1 = 1.00
        self.gamma_c2 = 0.875

        # Lumped thermal parameters (Appendix C.1)
        self.c_th = 65.2  # J/K
        self.r_th = 8.4   # K/W

        # Dynamic hysteresis parameters (Eqs. 17-18, Table 10)
        self.beta_hyst = 45.0   # Dimensionless SOC transition factor beta
        self.m0_hyst = 0.030    # Hysteresis envelope magnitude M_0 = 30.0 mV
        self.v_hyst = 0.0

        # Asymmetric Faradaic coulombic efficiency
        self.eta_charge = 0.995
        self.eta_discharge = 1.000

        # Internal states: [SOC, U1, U2, V_h]
        self.soc = 1.0
        self.up1 = 0.0
        self.up2 = 0.0

        self._update_parameters()

    def _f_t(self, temp_c, ea_cold, ea_hot):
        t_k = temp_c + 273.15
        ea = ea_cold if t_k < self.t_ref_k else ea_hot
        return float(np.exp((ea / self.gas_constant) * (1.0 / t_k - 1.0 / self.t_ref_k)))

    def _f_q(self, temp_c):
        t_k = temp_c + 273.15
        if t_k < self.t_ref_k:
            return float(np.exp(-(self.ea_q / self.gas_constant) * (1.0 / t_k - 1.0 / self.t_ref_k)))
        return float(1.0 + self.kappa_t_hot * (t_k - self.t_ref_k))

    def _update_parameters(self):
        deg = max(0.0, 1.0 - self.soh)
        ft0 = self._f_t(self.temperature_c, self.ea_r0_cold, self.ea_r0_hot)
        ft1 = self._f_t(self.temperature_c, self.ea_r1_cold, self.ea_r1_hot)
        ft2 = self._f_t(self.temperature_c, self.ea_r2, self.ea_r2)

        self.r0 = self.r0_base * ft0 * (1.0 + self.kappa_r0 * deg)
        self.r1 = self.r1_base * ft1 * (1.0 + self.kappa_r1 * deg)
        self.r2 = self.r2_base * ft2 * (1.0 + self.kappa_r2 * deg)
        self.c1 = max(100.0, self.c1_base * (1.0 - self.gamma_c1 * deg))
        self.c2 = max(100.0, self.c2_base * (1.0 - self.gamma_c2 * deg))

        self.q_actual = self.q_nominal * self.soh * self._f_q(self.temperature_c)
        self.tau1 = self.r1 * self.c1
        self.tau2 = self.r2 * self.c2

    def reset_state(self, initial_soc=1.0, temperature_c=None):
        self.soc = float(np.clip(initial_soc, 0.0, 1.0))
        self.up1 = 0.0
        self.up2 = 0.0
        self.v_hyst = 0.0
        if temperature_c is not None:
            self.temperature_c = float(temperature_c)
            self.t_amb_c = float(temperature_c)
        self._update_parameters()

    def get_ocv(self, soc):
        """
        Strictly monotonic closed-form LFP OCV curve on [0, 1]:
        V_ocv(z) = 1.950 + 1.320 * sigmoid(26*(z - 0.06)) + 0.065*z + 0.250 * sigmoid(24*(z - 0.89))
        producing a flat 3.28-3.34 V two-phase plateau and 2.18-3.57 V dynamic span.
        """
        s = np.clip(soc, 0.0, 1.0)
        sig_low = 1.0 / (1.0 + np.exp(-26.0 * (s - 0.06)))
        sig_high = 1.0 / (1.0 + np.exp(-24.0 * (s - 0.89)))
        return float(1.950 + 1.320 * sig_low + 0.065 * s + 0.250 * sig_high)

    def get_docv_dsoc(self, soc):
        s = np.clip(soc, 0.0, 1.0)
        sig_low = 1.0 / (1.0 + np.exp(-26.0 * (s - 0.06)))
        sig_high = 1.0 / (1.0 + np.exp(-24.0 * (s - 0.89)))
        return float(1.320 * 26.0 * sig_low * (1.0 - sig_low) + 0.065 + 0.250 * 24.0 * sig_high * (1.0 - sig_high))

    def get_eta(self, current_a):
        return self.eta_discharge if current_a >= 0.0 else self.eta_charge

    def step(self, current_a):
        eta = self.get_eta(current_a)
        abs_dz = abs(eta * current_a * self.dt) / (3600.0 * self.q_actual)
        d_soc = (-eta * current_a * self.dt) / (3600.0 * self.q_actual)
        self.soc = float(np.clip(self.soc + d_soc, 0.0, 1.0))

        exp1 = np.exp(-self.dt / self.tau1)
        exp2 = np.exp(-self.dt / self.tau2)
        self.up1 = self.up1 * exp1 + current_a * self.r1 * (1.0 - exp1)
        self.up2 = self.up2 * exp2 + current_a * self.r2 * (1.0 - exp2)

        # Exact ZOH discrete hysteresis update Eq. (18)
        if abs(current_a) > 1e-6:
            a_h = np.exp(-self.beta_hyst * abs_dz)
            sgn_i = 1.0 if current_a > 0.0 else -1.0
            self.v_hyst = float(a_h * self.v_hyst + (1.0 - a_h) * self.m0_hyst * sgn_i)

        if self.thermal_coupled:
            t_k = self.temperature_c + 273.15
            q_irrev = (current_a ** 2) * self.r0 + (self.up1 ** 2) / self.r1 + (self.up2 ** 2) / self.r2
            q_rev = -current_a * t_k * (-0.065e-3)
            dT = ((q_irrev + q_rev) - (self.temperature_c - self.t_amb_c) / self.r_th) * (self.dt / self.c_th)
            self.temperature_c += float(dT)
            self._update_parameters()

        ocv = self.get_ocv(self.soc)
        v_terminal = ocv - self.v_hyst - current_a * self.r0 - self.up1 - self.up2
        return float(v_terminal), float(self.soc)


BatteryCell = BatteryCellNMC
