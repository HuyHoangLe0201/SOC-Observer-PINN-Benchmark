"""
Battery Electro-Thermal & Degradation Model (NMC and LiFePO4 / LFP)
Journal of Electrochemical Energy Conversion and Storage (ASME)
Benchmark Framework for State-of-Charge (SOC) Estimation

Features:
1. NMC Chemistry Cell (BatteryCellNMC) with strictly monotonic OCV
2. LiFePO4 / LFP Chemistry Cell (BatteryCellLFP) with Flat OCV Plateau and Dynamic One-State Hysteresis
3. Arrhenius Temperature Kinetics (-10°C to 45°C)
4. SOH Degradation Cross-Coupling (Capacity fade & SEI resistance increase)
"""

import numpy as np


class BatteryCellNMC:
    """
    NMC (Nickel Manganese Cobalt) Lithium-ion Cell (e.g. CALCE INR18650-20R / Panasonic 18650PF).
    Nominal Capacity: 2.0 Ah, Nominal Voltage: 3.7 V (Cut-off: 3.0V - 4.2V).
    High OCV slope dOCV/dSOC across entire range (high observability).
    """

    def __init__(
        self,
        nominal_capacity_ah=2.0,
        soh=1.0,
        temperature_c=25.0,
        dt=1.0,
    ):
        self.chemistry = "NMC"
        self.q_nominal = float(nominal_capacity_ah)  # Ah
        self.soh = float(soh)  # 1.0 (Fresh) to 0.8 (EOL)
        self.temperature_c = float(temperature_c)
        self.dt = float(dt)

        # Baseline ECM parameters at 25°C, SOH = 1.0
        self.r0_base = 0.035   # Ohms
        self.r1_base = 0.020   # Ohms
        self.c1_base = 1200.0  # Farads
        self.r2_base = 0.025   # Ohms
        self.c2_base = 4500.0  # Farads

        # Arrhenius constants
        self.ea_r0 = 25000.0
        self.ea_r_polar = 28000.0
        self.gas_constant = 8.314
        self.t_ref_k = 298.15

        # Degradation sensitivities
        self.alpha_r0_deg = 2.5
        self.alpha_r_polar_deg = 1.8

        # States: [SOC, Up1, Up2]
        self.soc = 1.0
        self.up1 = 0.0
        self.up2 = 0.0

        # Calibrated strictly monotonic 5th-order polynomial for NMC
        # OCV = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4 + c5*s^5
        self.ocv_coeffs = np.array([
            3.21181343,
            4.67403064,
            -17.41856030,
            33.37645360,
            -28.84109993,
            9.20379162
        ])

        self._update_parameters()

    def _update_parameters(self):
        t_kelvin = self.temperature_c + 273.15
        arrhenius_r0 = np.exp((self.ea_r0 / self.gas_constant) * (1.0 / t_kelvin - 1.0 / self.t_ref_k))
        arrhenius_polar = np.exp((self.ea_r_polar / self.gas_constant) * (1.0 / t_kelvin - 1.0 / self.t_ref_k))

        aging_loss = max(0.0, 1.0 - self.soh)
        deg_r0 = 1.0 + self.alpha_r0_deg * aging_loss
        deg_polar = 1.0 + self.alpha_r_polar_deg * aging_loss

        self.r0 = self.r0_base * arrhenius_r0 * deg_r0
        self.r1 = self.r1_base * arrhenius_polar * deg_polar
        self.r2 = self.r2_base * arrhenius_polar * deg_polar
        self.c1 = self.c1_base / (deg_polar ** 0.5)
        self.c2 = self.c2_base / (deg_polar ** 0.5)

        t_cap_factor = 1.0 - 0.006 * max(0.0, 25.0 - self.temperature_c)
        self.q_actual = self.q_nominal * self.soh * t_cap_factor

        self.tau1 = self.r1 * self.c1
        self.tau2 = self.r2 * self.c2

    def set_environment(self, temperature_c, soh=None):
        self.temperature_c = float(temperature_c)
        if soh is not None:
            self.soh = float(soh)
        self._update_parameters()

    def reset_state(self, initial_soc=1.0):
        self.soc = float(initial_soc)
        self.up1 = 0.0
        self.up2 = 0.0

    def get_ocv(self, soc_val):
        soc_clamped = np.clip(soc_val, 0.0, 1.0)
        ocv = np.polyval(self.ocv_coeffs[::-1], soc_clamped)
        temp_shift = (self.temperature_c - 25.0) * (-0.00012)
        return ocv + temp_shift

    def get_docv_dsoc(self, soc_val):
        soc_clamped = np.clip(soc_val, 0.0, 1.0)
        poly_der = np.polyder(self.ocv_coeffs[::-1])
        return np.polyval(poly_der, soc_clamped)

    def step(self, current_a):
        coulomb_eff = 0.995 if current_a < 0 else 1.0
        d_soc = -(coulomb_eff * current_a * self.dt) / (3600.0 * self.q_actual)
        self.soc = np.clip(self.soc + d_soc, 0.0, 1.0)

        exp1 = np.exp(-self.dt / self.tau1)
        exp2 = np.exp(-self.dt / self.tau2)
        self.up1 = self.up1 * exp1 + current_a * self.r1 * (1.0 - exp1)
        self.up2 = self.up2 * exp2 + current_a * self.r2 * (1.0 - exp2)

        ocv = self.get_ocv(self.soc)
        v_terminal = ocv - current_a * self.r0 - self.up1 - self.up2
        return v_terminal, self.soc


class BatteryCellLFP:
    """
    LiFePO4 (LFP) Cell (e.g. CALCE A123 26650 or CATL LFP Blade Cell).
    Nominal Capacity: 2.5 Ah, Nominal Voltage: 3.2 V - 3.3 V.
    Key characteristics for ASME JEECS:
    1. Very flat OCV plateau between 20% and 80% SOC (dOCV/dSOC < 0.1 V/SOC).
    2. Dynamic path-dependent thermodynamic hysteresis (dV_h ~ 30-50 mV).
    """

    def __init__(
        self,
        nominal_capacity_ah=2.5,
        soh=1.0,
        temperature_c=25.0,
        dt=1.0,
    ):
        self.chemistry = "LFP"
        self.q_nominal = float(nominal_capacity_ah)
        self.soh = float(soh)
        self.temperature_c = float(temperature_c)
        self.dt = float(dt)

        # Baseline ECM parameters for LFP 26650
        self.r0_base = 0.022   # Ohms
        self.r1_base = 0.015   # Ohms
        self.c1_base = 2500.0  # Farads (larger double-layer capacitance)
        self.r2_base = 0.018   # Ohms
        self.c2_base = 8000.0  # Farads

        # Arrhenius & aging parameters
        self.ea_r0 = 24000.0
        self.ea_r_polar = 26000.0
        self.gas_constant = 8.314
        self.t_ref_k = 298.15
        self.alpha_r0_deg = 2.0
        self.alpha_r_polar_deg = 1.5

        # Dynamic Hysteresis Model parameters (One-state differential equation)
        # d(Vh)/dt = -gamma * |I| * Vh + gamma * M(SOC) * I
        self.gamma_hyst = 0.75   # Hysteresis transition rate parameter (1/Ah)
        self.max_hyst_v = 0.035  # Maximum hysteresis voltage (35 mV)
        self.v_hyst = 0.0        # State: Hysteresis voltage (V)

        # Internal states: [SOC, Up1, Up2, V_hyst]
        self.soc = 1.0
        self.up1 = 0.0
        self.up2 = 0.0

        self._update_parameters()

    def _update_parameters(self):
        t_kelvin = self.temperature_c + 273.15
        arrhenius_r0 = np.exp((self.ea_r0 / self.gas_constant) * (1.0 / t_kelvin - 1.0 / self.t_ref_k))
        arrhenius_polar = np.exp((self.ea_r_polar / self.gas_constant) * (1.0 / t_kelvin - 1.0 / self.t_ref_k))

        aging_loss = max(0.0, 1.0 - self.soh)
        deg_r0 = 1.0 + self.alpha_r0_deg * aging_loss
        deg_polar = 1.0 + self.alpha_r_polar_deg * aging_loss

        self.r0 = self.r0_base * arrhenius_r0 * deg_r0
        self.r1 = self.r1_base * arrhenius_polar * deg_polar
        self.r2 = self.r2_base * arrhenius_polar * deg_polar
        self.c1 = self.c1_base / (deg_polar ** 0.5)
        self.c2 = self.c2_base / (deg_polar ** 0.5)

        t_cap_factor = 1.0 - 0.008 * max(0.0, 25.0 - self.temperature_c)
        self.q_actual = self.q_nominal * self.soh * t_cap_factor

        self.tau1 = self.r1 * self.c1
        self.tau2 = self.r2 * self.c2

    def reset_state(self, initial_soc=1.0):
        self.soc = float(initial_soc)
        self.up1 = 0.0
        self.up2 = 0.0
        self.v_hyst = 0.0

    def get_ocv(self, soc_val):
        """
        LFP Open-Circuit Voltage:
        Features sharp slope at edges (SOC < 15% and SOC > 85%)
        and very flat central plateau (~3.28V to 3.32V) in the 20% - 80% range.
        Modeled via hyperbolic tangent formulation commonly adopted in ASME JEECS.
        """
        s = np.clip(soc_val, 0.0, 1.0)
        # Baseline central plateau at 3.30V
        # Transitions at low end (s < 0.15) and high end (s > 0.85)
        v_low = -0.65 / (1.0 + np.exp((s - 0.08) / 0.025))
        v_high = 0.35 / (1.0 + np.exp(-(s - 0.92) / 0.025))
        v_linear = 0.08 * s  # Minimal slope in the plateau (only 80 mV over entire 0-1)
        ocv = 3.28 + v_low + v_high + v_linear

        temp_shift = (self.temperature_c - 25.0) * (-0.00010)
        return ocv + temp_shift

    def get_docv_dsoc(self, soc_val):
        """Numerical / analytical derivative of LFP OCV curve."""
        eps = 1e-4
        ocv_plus = self.get_ocv(soc_val + eps)
        ocv_minus = self.get_ocv(soc_val - eps)
        return (ocv_plus - ocv_minus) / (2.0 * eps)

    def step(self, current_a):
        """
        Advance LFP cell state including dynamic hysteresis.
        """
        coulomb_eff = 0.995 if current_a < 0 else 1.0
        d_soc = -(coulomb_eff * current_a * self.dt) / (3600.0 * self.q_actual)
        self.soc = np.clip(self.soc + d_soc, 0.0, 1.0)

        # Polarization voltages update
        exp1 = np.exp(-self.dt / self.tau1)
        exp2 = np.exp(-self.dt / self.tau2)
        self.up1 = self.up1 * exp1 + current_a * self.r1 * (1.0 - exp1)
        self.up2 = self.up2 * exp2 + current_a * self.r2 * (1.0 - exp2)

        # Hysteresis update (One-state differential equation)
        # dVh/dt = -gamma * |I| * Vh + gamma * M(SOC) * I
        # Discrete exact update:
        ah_step = (np.abs(current_a) * self.dt) / 3600.0
        exp_h = np.exp(-self.gamma_hyst * ah_step)
        h_target = np.sign(current_a) * self.max_hyst_v if abs(current_a) > 0.01 else 0.0
        self.v_hyst = self.v_hyst * exp_h + h_target * (1.0 - exp_h)

        # Terminal voltage: Vt = OCV(SOC) - I*R0 - Up1 - Up2 - V_hyst
        ocv = self.get_ocv(self.soc)
        v_terminal = ocv - current_a * self.r0 - self.up1 - self.up2 - self.v_hyst
        return v_terminal, self.soc


# Backward compatibility alias
BatteryCell = BatteryCellNMC
