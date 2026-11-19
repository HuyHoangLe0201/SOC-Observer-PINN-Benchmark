"""
Electrochemical Impedance Spectroscopy (EIS) Fitting & Aging Consistency Analyzer (Phase 8)
Journal of Energy Storage (Elsevier)

Features:
1. Direct Nonlinear Least-Squares (scipy.optimize.least_squares) Randles-CPE fitting on
   real experimental NASA Ames PCoE EIS spectra (B0007 / B0006, 0.1 Hz - 5 kHz, 39 frequencies)
   across aging checkpoints, reporting fitted parameters, +/-1-sigma uncertainty, and RMS fit error.
2. Unit-consistent model visualization linked to Table 10 (mOhm and mF*s^(alpha-1)), clarifying
   the ms-scale interfacial semicircle vs. 24-s ECM macroscopic polarization branch.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import nasa_data
try:
    from pub_style import apply_pub_style, save_pub_figure, style_legend
except ImportError:
    from src.pub_style import apply_pub_style, save_pub_figure, style_legend

apply_pub_style()


def randles_cpe_model(params, omega, include_inductance=False):
    """
    Randles-CPE-Warburg impedance (strictly consistent units: resistances in Ohm, Q_cpe in F*s^(alpha-1)).
    """
    if include_inductance:
        r0, l_ind, rct, q_cpe, alpha, sigma_w = params
        z_l = 1j * omega * l_ind
    else:
        r0, rct, q_cpe, alpha, sigma_w = params
        z_l = 0.0
    zw = (sigma_w / np.sqrt(np.maximum(omega, 1e-6))) * (1.0 - 1j)
    z_branch = (rct + zw) / (1.0 + ((1j * omega) ** alpha) * q_cpe * (rct + zw))
    return r0 + z_l + z_branch


def fit_experimental_nasa_eis(cell_id="B0006"):
    """
    Fit real experimental NASA EIS spectra (39 points, 0.1 Hz - 5 kHz) with parasitic
    cable/tab inductance L across fresh, mid-life, and end-of-test SOH.
    """
    dis = nasa_data.discharge_cycles(cell_id)
    imp = nasa_data.impedance_records(cell_id)
    freqs = np.logspace(np.log10(0.1), np.log10(5000.0), 39)
    omega = 2.0 * np.pi * freqs

    picks = [("Fresh (~100% SOH)", 0), ("Mid-Life (~90% SOH)", len(imp) // 2), ("End-of-Test (~75% SOH)", len(imp) - 1)]
    fitted_rows = []
    spectra = {}

    for lbl, ix in picks:
        rec = imp[ix]
        z_meas = rec["Z"]
        p0 = [max(0.01, rec["Re"]), 1.5e-7, max(0.01, rec["Rct"] - rec["Re"]), 1.2, 0.35, 0.0005]
        lb = [0.005, 0.0, 0.002, 1e-3, 0.15, 0.0]
        ub = [0.500, 1e-5, 0.500, 10.0, 0.99, 0.1]

        res = least_squares(
            lambda p: np.concatenate([
                np.real(randles_cpe_model(p, omega, include_inductance=True) - z_meas),
                np.imag(randles_cpe_model(p, omega, include_inductance=True) - z_meas),
            ]),
            p0,
            bounds=(lb, ub),
        )
        z_fit = randles_cpe_model(res.x, omega, include_inductance=True)
        dof = max(1, len(res.fun) - len(res.x))
        s2 = float(np.sum(res.fun ** 2)) / dof
        pcov = np.linalg.pinv(res.jac.T @ res.jac) * s2
        perr = np.sqrt(np.diag(pcov))

        rms_mohm = float(np.sqrt(np.mean(np.abs(z_fit - z_meas) ** 2))) * 1000.0
        rel_pct = rms_mohm / (float(np.mean(np.abs(z_meas))) * 1000.0) * 100.0

        fitted_rows.append({
            "Stage": lbl,
            "R0 (mOhm)": round(res.x[0] * 1000.0, 2),
            "R0_err (mOhm)": round(perr[0] * 1000.0, 2),
            "L (uH)": round(res.x[1] * 1e6, 3),
            "Rct (mOhm)": round(res.x[2] * 1000.0, 2),
            "Rct_err (mOhm)": round(perr[2] * 1000.0, 2),
            "Q_cpe (mF*s^(a-1))": round(res.x[3] * 1000.0, 1),
            "alpha": round(res.x[4], 3),
            "alpha_err": round(perr[4], 3),
            "sigma_w (mOhm*s^-1/2)": round(res.x[5] * 1000.0, 2),
            "RMS Fit Error (mOhm)": round(rms_mohm, 2),
            "Rel Fit Error (%)": round(rel_pct, 2),
        })
        spectra[lbl] = {"z_meas": z_meas, "z_fit": z_fit, "params": res.x, "rms_mohm": rms_mohm, "rel_pct": rel_pct}

    return pd.DataFrame(fitted_rows), spectra, freqs


def generate_eis_figures_and_tables():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    df_fit, spectra_nasa, freqs_nasa = fit_experimental_nasa_eis("B0006")
    df_fit.to_csv(os.path.join(results_dir, "eis_nasa_b0006_fit.csv"), index=False)
    print("=== Real NASA B0006 EIS Least-Squares Fitting Results ===")
    print(df_fit.to_string(index=False))

    # Plot 2-panel Figure 8 (Fig 10 in paper):
    # (a) Real NASA B0006 Experimental Nyquist Data vs Randles-CPE Least-Squares Fit
    # (b) Unit-Consistent Table 10 Plant Aging Law Impedance Trajectory (mOhm & mF*s^(alpha-1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"wspace": 0.26})

    colors = ["#15803D", "#D97706", "#DC2626"]
    markers = ["o", "s", "^"]
    for (lbl, d), col, mk in zip(spectra_nasa.items(), colors, markers):
        zm = d["z_meas"] * 1000.0
        zf = d["z_fit"] * 1000.0
        ax1.scatter(zm.real, -zm.imag, color=col, marker=mk, s=22, alpha=0.80, label=f"{lbl} (Meas.)")
        ax1.plot(zf.real, -zf.imag, color=col, linewidth=1.85, label=f"Fit (RMS: {d['rms_mohm']:.2f} m$\\Omega$, {d['rel_pct']:.1f}%)")

    ax1.axhline(0.0, color="#64748B", linestyle=":", linewidth=0.8)
    ax1.set_xlabel(r"Real Impedance $Z'$ ($\mathrm{m\Omega}$)")
    ax1.set_ylabel(r"Negative Imaginary $-Z''$ ($\mathrm{m\Omega}$)")
    ax1.set_title(r"(a) NASA B0006 Measured vs. Fitted EIS (0.1 Hz–5 kHz)", loc="left", fontweight="bold", fontsize=9.6)
    ax1.grid(True)
    style_legend(ax1.legend(loc="upper left", fontsize=6.6, ncol=1))

    # Panel (b): Plant Aging Consistency Check (Table 10 NMC: R0: 35 -> 52.5 mOhm, R1=Rct: 15 -> 37.5 mOhm)
    # Using strict consistent SI units: R in Ohm, Q_cpe = 0.564e-3 * (1 - 1.25*deg) F*s^(alpha-1) = 0.564 mF*s^(alpha-1)
    freqs_sim = np.logspace(-1.8, 6.0, 500)  # 16 mHz to 1 MHz so full semicircle touches R0=35 mOhm
    omega_sim = 2.0 * np.pi * freqs_sim
    for soh, lbl, col in [(1.00, "Fresh (100% SOH)", "#15803D"),
                          (0.90, "Aged (90% SOH)", "#D97706"),
                          (0.80, "End-of-Life (80% SOH)", "#DC2626")]:
        deg = 1.0 - soh
        r0_ohm = 0.035 * (1.0 + 2.5 * deg)      # 35.0 -> 52.5 mOhm
        rct_ohm = 0.015 * (1.0 + 7.5 * deg)     # 15.0 -> 37.5 mOhm (Table 10 R1!)
        q_cpe_si = (0.564e-3) * (1.0 - 1.25 * deg)
        sigma_w_si = 2.8e-3 / np.sqrt(soh)
        z_sim = randles_cpe_model([r0_ohm, rct_ohm, q_cpe_si, 0.94, sigma_w_si], omega_sim) * 1000.0
        ax2.plot(z_sim.real, -z_sim.imag, color=col, linewidth=2.1, label=lbl)

    ax2.axhline(0.0, color="#64748B", linestyle=":", linewidth=0.8)
    ax2.annotate("", xy=(52.5, -1.0), xytext=(35.0, -1.0), arrowprops=dict(arrowstyle="<->", color="#1E293B", lw=1.0))
    ax2.text(43.75, -2.5, r"$\Delta R_0 = +50\%$ ($35.0 \rightarrow 52.5\;\mathrm{m\Omega}$)", fontsize=7.0, fontweight="bold", ha="center", color="#1E293B")
    ax2.annotate(r"$\Delta R_{\mathrm{ct}} = +150\%$ ($15.0 \rightarrow 37.5\;\mathrm{m\Omega}$)" + "\n" + r"($f_c \approx 19\;\mathrm{Hz},\;\tau_{\mathrm{ct}} \sim 8.5\;\mathrm{ms}$)",
                 xy=(71.2, 17.3), xytext=(77.5, 24.5),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0),
                 fontsize=7.2, fontweight="bold", color="#DC2626", ha="center",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#DC2626", alpha=0.92, lw=0.7))
    ax2.set_xlabel(r"Real Impedance $Z'$ ($\mathrm{m\Omega}$)")
    ax2.set_ylabel(r"Negative Imaginary $-Z''$ ($\mathrm{m\Omega}$)")
    ax2.set_title(r"(b) Plant Aging-Law Spectra (16 mHz–1 MHz)", loc="left", fontweight="bold", fontsize=9.6)
    ax2.set_xlim(30.0, 105.0)
    ax2.set_ylim(-3.8, 30.0)
    ax2.grid(True)
    style_legend(ax2.legend(loc="upper left", fontsize=7.6))

    out_fig10 = os.path.join(results_dir, "fig10.png")
    save_pub_figure(fig, out_fig10)
    print(f"Saved Fig 10 to {out_fig10}")


if __name__ == "__main__":
    generate_eis_figures_and_tables()
