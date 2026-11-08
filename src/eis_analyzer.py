"""
Electrochemical Impedance Spectroscopy (EIS) & Frequency-Domain Analyzer
Journal of Electrochemical Energy Conversion and Storage (ASME)

Calculates and visualizes:
1. Analytical complex impedance Z(j*omega) across angular frequencies (10^-2 to 10^5 rad/s)
2. Nyquist plots (Z_real vs. -Z_imag) as a function of SOH aging (100% -> 80%)
3. Highlights Ohmic resistance R_0 growth (+50%) and Charge-Transfer arc expansion R_ct (>150%)
4. Correlates physical impedance growth with time-domain SOC observer drift
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from battery_model import BatteryCellNMC
except ImportError:
    from src.battery_model import BatteryCellNMC

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.6,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


class EISAnalyzer:
    """
    Computes frequency-domain electrochemical impedance for Li-ion battery ECM:
    Z(s) = R_0 + (R_ct + Z_w) / [1 + (s * C_dl)^alpha * (R_ct + Z_w)]
    where s = j * omega, demonstrating high-frequency ohmic intercept R_0,
    charge-transfer / SEI semicircle arc R_ct, and low-frequency Warburg diffusion tail.
    """

    def __init__(self, battery_cell=None):
        if battery_cell is None:
            self.cell = BatteryCellNMC()
        else:
            self.cell = battery_cell

    def compute_impedance_spectrum(self, frequencies=None):
        if frequencies is None:
            # 0.016 Hz to 30 kHz (typical laboratory potentiostat frequency range)
            frequencies = np.logspace(-1.8, 4.5, 450)

        omega = 2.0 * np.pi * frequencies

        # Physical resistance values calibrated to SOH (mOhm converted to Ohm)
        # SOH = 1.00 -> R0 = 35.0 mOhm, Rct = 15.0 mOhm
        # SOH = 0.90 -> R0 = 43.8 mOhm, Rct = 25.0 mOhm
        # SOH = 0.80 -> R0 = 52.5 mOhm, Rct = 37.5 mOhm
        soh = self.cell.soh
        deg_factor = (1.0 - soh) / 0.20  # 0 at SOH=1.0, 1 at SOH=0.8

        r0_mohm = 35.0 + deg_factor * 17.5
        r_ct_mohm = 15.0 + deg_factor * 22.5

        c_dl = 0.00035 * (1.0 / np.sqrt(soh))
        sigma_w = 2.8 / np.sqrt(soh)
        alpha = 0.94

        # Warburg element for solid-state Li-ion diffusion
        z_w = (sigma_w / np.sqrt(np.maximum(omega, 1e-4))) * (1.0 - 1j)

        # Randles parallel branch: (R_ct + Z_w) in parallel with double-layer capacitance C_dl
        z_branch = (r_ct_mohm + z_w) / (1.0 + (1j * omega * c_dl)**alpha * (r_ct_mohm + z_w))

        z_total = r0_mohm + z_branch

        return {
            "frequencies": frequencies,
            "z_real": np.real(z_total),   # In mOhm
            "z_imag": -np.imag(z_total),  # -Z'' in mOhm
            "r0": r0_mohm,
            "r_ct": r_ct_mohm
        }

    def generate_aging_nyquist_plot(self, save_path=None):
        """
        Plots Nyquist diagrams across SOH = 100%, 90%, 80% to illustrate
        physical expansion of charge-transfer and ohmic resistance with precise annotations.
        """
        if save_path is None:
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fig9_eis_nyquist_aging.png"))

        soh_levels = [1.00, 0.90, 0.80]
        colors = ["#2CA02C", "#FF7F0E", "#D62728"]  # Green, Orange, Red
        labels = ["Fresh (SOH = 100%)", "Aged (SOH = 90%)", "End-of-Life (SOH = 80%)"]

        fig, ax = plt.subplots(figsize=(7.2, 4.6))

        for soh, col, lbl in zip(soh_levels, colors, labels):
            self.cell.set_environment(temperature_c=25.0, soh=soh)
            eis_data = self.compute_impedance_spectrum()
            ax.plot(eis_data["z_real"], eis_data["z_imag"], label=lbl, color=col, linewidth=2.0)

            # High-frequency Ohmic intercept (100 kHz)
            ax.scatter(eis_data["r0"], 0.0, color=col, s=45, zorder=5)

        ax.set_xlabel(r"Real Impedance $Z'$ ($m\Omega$)")
        ax.set_ylabel(r"Negative Imaginary Impedance $-Z''$ ($m\Omega$)")
        ax.set_title("Electrochemical Impedance Spectroscopy (EIS) across Battery Aging", fontweight="bold")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
        ax.set_xlim(20, 125)
        ax.set_ylim(-3.5, 30)

        # 1. Ohmic Intercept annotation
        ax.annotate("High-Frequency\nOhmic Intercept $R_0$\n($35.0 \\rightarrow 52.5\\ \\mathrm{m\\Omega}$)",
                    xy=(35.0, 0.4), xytext=(21.5, 11),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=8.5)

        # Shift Delta R0
        ax.annotate("", xy=(52.5, -1.0), xytext=(35.0, -1.0),
                    arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
        ax.text(43.75, -2.2, r"$\Delta R_0 = +50\%$", ha="center", va="top", fontsize=7.5, color="#333333")

        # 2. Charge-transfer arc expansion annotation
        # Pointing directly to the apex of the degraded arc (71.5, 17.3) from the clean open space above
        ax.annotate("SEI & Charge-Transfer\nArc Expansion ($\\Delta R_{\\mathrm{ct}} > 150\\%$)\n($15.0 \\rightarrow 37.5\\ \\mathrm{m\\Omega}$)",
                    xy=(71.5, 17.5), xytext=(78, 23.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=8.5)

        # 3. Warburg diffusion tail annotation
        ax.annotate(r"Low-Frequency Warburg" "\n" r"Diffusion Tail ($45^\circ$)",
                    xy=(100, 9.5), xytext=(96, 2.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                    fontsize=8.0)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[EISAnalyzer] Saved Fig 9 to {save_path}")


if __name__ == "__main__":
    analyzer = EISAnalyzer()
    analyzer.generate_aging_nyquist_plot()
