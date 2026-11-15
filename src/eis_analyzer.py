"""
Electrochemical Impedance Spectroscopy (EIS) & Frequency-Domain Analyzer
Journal of Energy Storage (Elsevier)

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
try:
    from pub_style import apply_pub_style, save_pub_figure, style_legend
except ImportError:
    from src.pub_style import apply_pub_style, save_pub_figure, style_legend

apply_pub_style()


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
        Plots Nyquist diagrams across SOH = 100%, 90%, 80% with characteristic
        frequency markers and an embedded Randles-CPE equivalent-circuit schematic.
        """
        if save_path is None:
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "fig8_eis_nyquist_aging.png"))

        soh_levels = [1.00, 0.90, 0.80]
        colors = ["#16A34A", "#EA580C", "#DC2626"]
        labels = ["Fresh (SOH = 100%)", "Aged (SOH = 90%)", "End-of-Life (SOH = 80%)"]

        fig, ax = plt.subplots(figsize=(7.4, 4.7))
        ax.axhline(0.0, color="#64748B", linewidth=0.85, linestyle="-", zorder=1)

        freq_targets = [(1000.0, "1 kHz"), (100.0, "100 Hz"), (10.0, "10 Hz"), (1.0, "1 Hz"), (0.05, "50 mHz")]

        for soh, col, lbl in zip(soh_levels, colors, labels):
            self.cell.set_environment(temperature_c=25.0, soh=soh)
            eis_data = self.compute_impedance_spectrum()
            freqs = eis_data["frequencies"]
            zr = eis_data["z_real"]
            zi = eis_data["z_imag"]
            ax.plot(zr, zi, label=lbl, color=col, linewidth=2.1, zorder=3)
            ax.scatter(eis_data["r0"], 0.0, color=col, s=50, edgecolor="white", linewidth=0.8, zorder=6)

            # Plot characteristic frequency markers on Fresh & End-of-Life arcs
            if soh in (1.00, 0.80):
                eol_offsets = {
                    1000.0: (4.8, 0.9, "left"),
                    100.0: (8.2, 0.2, "left"),
                    10.0: (0.0, 1.55, "center"),
                    1.0: (-3.2, 0.8, "right"),
                    0.05: (4.2, -0.4, "left"),
                }
                for f_val, f_lbl in freq_targets:
                    idx = int(np.argmin(np.abs(freqs - f_val)))
                    ax.scatter(zr[idx], zi[idx], color=col, marker="D", s=26, edgecolor="#111827", linewidth=0.6, zorder=6)
                    if soh == 0.80:
                        dx, dy, align = eol_offsets[f_val]
                        ax.annotate(
                            f_lbl,
                            xy=(zr[idx], zi[idx]),
                            xytext=(zr[idx] + dx, zi[idx] + dy),
                            arrowprops=dict(arrowstyle="-", color="#475569", lw=0.65) if abs(dx) > 2.0 else None,
                            fontsize=6.8,
                            fontweight="bold",
                            color="#1E293B",
                            ha=align,
                            va="center",
                            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="#E2E8F0", alpha=0.90, lw=0.4),
                            zorder=7,
                        )

        ax.set_xlabel(r"Real Impedance $Z'$ ($\mathrm{m}\Omega$)")
        ax.set_ylabel(r"Negative Imaginary Impedance $-Z''$ ($\mathrm{m}\Omega$)")
        ax.set_title("Electrochemical Impedance Spectroscopy (EIS) across Battery Aging", fontweight="bold")
        ax.grid(True)
        style_legend(ax.legend(loc="upper left", fontsize=7.9))
        ax.set_xlim(22.0, 122.0)
        ax.set_ylim(-3.0, 30.8)

        # Ohmic intercept shift
        ax.annotate(
            "High-Freq. Ohmic\nIntercept $R_0$\n($35.0 \\rightarrow 52.5\\;\\mathrm{m\\Omega}$)",
            xy=(35.0, 0.35),
            xytext=(24.0, 11.4),
            arrowprops=dict(arrowstyle="->", color="#111827", lw=1.05),
            fontsize=7.7,
            fontweight="bold",
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.94, lw=0.65),
        )
        ax.annotate("", xy=(52.5, -1.05), xytext=(35.0, -1.05), arrowprops=dict(arrowstyle="<->", color="#1E293B", lw=1.0))
        ax.text(43.75, -1.65, r"$\Delta R_0 = +50\%$", ha="center", va="top", fontsize=7.6, fontweight="bold", color="#1E293B")

        # Charge-transfer arc expansion
        ax.annotate(
            "SEI & Charge-Transfer\nArc Expansion ($\\Delta R_{\\mathrm{ct}} > 150\\%$)\n($15.0 \\rightarrow 37.5\\;\\mathrm{m\\Omega}$)",
            xy=(72.0, 17.5),
            xytext=(100.5, 24.4),
            arrowprops=dict(arrowstyle="->", color="#111827", lw=1.05),
            fontsize=7.7,
            fontweight="bold",
            color="#111827",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#CBD5E1", alpha=0.95, lw=0.65),
            zorder=8,
        )

        # Warburg diffusion tail
        ax.annotate(
            "Low-Freq. Warburg\n" + r"Diffusion Tail ($45^\circ$)",
            xy=(100.0, 9.8),
            xytext=(105.0, 2.2),
            arrowprops=dict(arrowstyle="->", color="#111827", lw=1.0),
            fontsize=7.5,
            fontweight="bold",
            color="#111827",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", alpha=0.94, lw=0.65),
            zorder=8,
        )

        # Inset Randles-CPE Schematic cleanly centered between Legend and Expansion Callout
        c_ax = ax.inset_axes([0.365, 0.72, 0.25, 0.22])
        c_ax.set_facecolor("#F8FAFC")
        c_ax.set_xticks([])
        c_ax.set_yticks([])
        for sp in c_ax.spines.values():
            sp.set_edgecolor("#475569")
            sp.set_linewidth(0.75)
        c_ax.set_xlim(0, 10)
        c_ax.set_ylim(0, 4.2)
        c_ax.plot([0.5, 1.4], [2.1, 2.1], "k-", lw=1.0)
        c_ax.add_patch(plt.Rectangle((1.4, 1.65), 1.6, 0.9, fc="#E2E8F0", ec="k", lw=0.8))
        c_ax.text(2.2, 2.1, r"$R_0$", ha="center", va="center", fontsize=7.0, fontweight="bold")
        c_ax.plot([3.0, 3.8], [2.1, 2.1], "k-", lw=1.0)
        c_ax.plot([3.8, 3.8], [1.0, 3.1], "k-", lw=1.0)
        c_ax.plot([3.8, 5.2], [3.1, 3.1], "k-", lw=1.0)
        c_ax.plot([5.2, 5.2], [2.65, 3.55], "k-", lw=1.2)
        c_ax.plot([5.7, 5.7], [2.65, 3.55], "k-", lw=1.2)
        c_ax.plot([5.7, 8.6], [3.1, 3.1], "k-", lw=1.0)
        c_ax.text(5.45, 3.72, r"$\mathrm{CPE}$ ($\alpha\!=\!0.94$)", ha="center", va="bottom", fontsize=6.2, fontweight="bold")
        c_ax.plot([3.8, 4.4], [1.0, 1.0], "k-", lw=1.0)
        c_ax.add_patch(plt.Rectangle((4.4, 0.55), 1.6, 0.9, fc="#DBEAFE", ec="k", lw=0.8))
        c_ax.text(5.2, 1.0, r"$R_{\mathrm{ct}}$", ha="center", va="center", fontsize=6.8, fontweight="bold")
        c_ax.plot([6.0, 6.6], [1.0, 1.0], "k-", lw=1.0)
        c_ax.add_patch(plt.Rectangle((6.6, 0.55), 1.4, 0.9, fc="#DCFCE7", ec="k", lw=0.8))
        c_ax.text(7.3, 1.0, r"$Z_W$", ha="center", va="center", fontsize=6.8, fontweight="bold")
        c_ax.plot([8.0, 8.6], [1.0, 1.0], "k-", lw=1.0)
        c_ax.plot([8.6, 8.6], [1.0, 3.1], "k-", lw=1.0)
        c_ax.plot([8.6, 9.5], [2.1, 2.1], "k-", lw=1.0)

        save_pub_figure(fig, save_path)
        print(f"[EISAnalyzer] Saved Fig 8 (PNG 600 DPI + PDF) to {save_path}")


if __name__ == "__main__":
    analyzer = EISAnalyzer()
    analyzer.generate_aging_nyquist_plot()
