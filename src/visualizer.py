"""
Publication-Quality Visualization Generator
Journal of Energy Storage (Elsevier)

Generates clean, vector-quality, double-column figures for publication:
1. Fig 1: Dynamic Drive Cycle (UDDS) Tracking & Error Residuals
2. Fig 2: Degradation Robustness across Progressive Battery Aging (SOH: 100% -> 80%)
3. Fig 3: Thermal Robustness across Multi-Temperature Regimes (0°C, 25°C, 45°C)
4. Fig 4: Convergence Trajectories under Severe Initial SOC Uncertainty (Delta SOC = -40%)
5. Fig 5: Trade-off Radar Chart: Accuracy vs. Aging Robustness vs. Latency
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib for publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.6,
    "grid.alpha": 0.65,
    "grid.linestyle": "--",
    "grid.color": "#E2E8F0",
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#2B2B2B",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5
})

COLORS = {
    "True": "#111111",
    "EKF": "#E41A1C",        # Red
    "AEKF": "#377EB8",       # Blue
    "DEKF": "#4DAF4A",       # Green
    "SMO": "#984EA3",        # Purple
    "BiLSTM-Attn": "#FF7F00", # Orange
    "PINN-SOC": "#008080"    # Deep Teal
}


def plot_tracking_and_errors(data, results, save_path="../results/fig1_udds_tracking_errors.png"):
    """
    Fig 1: Upper plot: True vs Estimated SOC. Lower plot: SOC Error (%).
    """
    time_min = data["time"] / 60.0
    soc_true = data["soc_true"] * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={'height_ratios': [1.3, 1.0]})

    # Upper panel: SOC Trajectories
    ax1.plot(time_min, soc_true, label="Reference (True)", color=COLORS["True"], linewidth=2.2, linestyle="-")
    for name, res in results.items():
        ax1.plot(time_min, res["soc_est"] * 100.0, label=name, color=COLORS.get(name, "#888888"), alpha=0.85)

    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title("(a) SOC Tracking under Dynamic UDDS Cycle at 25°C", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_ylim(35, 105)
    ax1.legend(loc="lower left", ncol=3, framealpha=0.95, fontsize=8)

    # Lower panel: Error Residuals
    ax2.axhspan(-2.0, 2.0, color="#2CA02C", alpha=0.07, label=None, zorder=0)
    for name, res in results.items():
        err = (res["soc_est"] - data["soc_true"]) * 100.0
        ax2.plot(time_min, err, label=f"{name} (RMSE: {res['rmse_pct']:.2f}%)", color=COLORS.get(name, "#888888"), alpha=0.85)

    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.axhline(2.0, color="#555555", linestyle=":", linewidth=0.9)
    ax2.axhline(-2.0, color="#555555", linestyle=":", linewidth=0.9)
    ax2.text(1.0, 2.2, r"$\pm 2\%$ Target Corridor", fontsize=7.5, color="#444444", va="bottom", fontweight="bold")

    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("SOC Error (%)")
    ax2.set_title(r"(b) Estimation Error Residuals ($e_k = \hat{s}_k - s_k$)", loc="left", fontweight="bold")
    ax2.grid(True)
    ax2.set_ylim(-8.5, 13.5)
    ax2.legend(loc="upper right", ncol=3, framealpha=0.95, fontsize=7.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Fig 1 to {save_path}")


def plot_aging_degradation_trend(soh_levels, rmse_by_model, save_path="../results/fig2_aging_degradation.png"):
    """
    Fig 2: RMSE evolution as battery degrades from SOH = 100% to SOH = 80%.
    Demonstrates Aging Resilience of DEKF and PINN-SOC vs. degradation drift in standard EKF.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    markers = {"EKF": "s", "AEKF": "^", "DEKF": "o", "SMO": "d", "BiLSTM-Attn": "v", "PINN-SOC": "P"}

    # Target precision zone (<1% RMSE)
    ax.axhspan(0.0, 1.0, color="#2CA02C", alpha=0.08, label="Target Zone (<1% RMSE)", zorder=0)

    for name, rmse_vals in rmse_by_model.items():
        col = COLORS.get(name, "#888888")
        ax.plot(
            soh_levels * 100.0,
            rmse_vals,
            label=name,
            color=col,
            marker=markers.get(name, "o"),
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.9
        )

    # Informative annotations for standout phenomena with white background boxes to guarantee zero collision
    ax.annotate("+4473% Error Surge", xy=(80.0, 2.973), xytext=(83.0, 3.8),
                arrowprops=dict(arrowstyle="->", color="#E41A1C", lw=1.0),
                fontsize=8.0, color="#E41A1C", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#E41A1C", alpha=0.92, lw=0.7))
    ax.annotate("DEKF Robust (<0.6%)", xy=(80.0, 0.585), xytext=(83.5, 1.45),
                arrowprops=dict(arrowstyle="->", color="#2CA02C", lw=1.0),
                fontsize=8.0, color="#2CA02C", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2CA02C", alpha=0.92, lw=0.7))

    ax.invert_xaxis()  # Aging progresses from 100% down to 80%
    ax.set_xlabel("State of Health - SOH (%) [Fresh $\\rightarrow$ Aged]")
    ax.set_ylabel("SOC Estimation RMSE (%)")
    ax.set_title("Degradation Robustness: SOC Error across Cell Aging Stages", fontweight="bold")
    ax.grid(True)
    ax.set_ylim(-0.15, 7.5)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Fig 2 to {save_path}")


def plot_thermal_regimes_bar(temp_labels, rmse_data, save_path="../results/fig3_thermal_robustness.png"):
    """
    Fig 3: Grouped bar chart comparing models across 0°C, 25°C, 45°C.
    """
    models = list(rmse_data.keys())
    n_models = len(models)
    n_temps = len(temp_labels)

    x = np.arange(n_temps)
    width = 0.125
    hatches = ['', '//', '\\\\', 'xx', '..', '++']

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for idx, model_name in enumerate(models):
        offset = (idx - n_models / 2.0 + 0.5) * width
        vals = [rmse_data[model_name][t] for t in temp_labels]
        ax.bar(x + offset, vals, width, label=model_name,
               color=COLORS.get(model_name, "#888888"),
               edgecolor="#222222", linewidth=0.6,
               hatch=hatches[idx % len(hatches)], alpha=0.9)

    ax.axhline(2.0, color="crimson", linestyle="--", linewidth=0.9, alpha=0.6, label="2% Bound")
    ax.set_xticks(x)
    ax.set_xticklabels(temp_labels)
    ax.set_xlabel("Operating Ambient Temperature (°C)")
    ax.set_ylabel("SOC Estimation RMSE (%)")
    ax.set_title("Thermal Sensitivity Benchmark across Temperature Windows", fontweight="bold")
    ax.grid(True, axis="y")
    ax.set_ylim(0.0, 12.0)
    ax.legend(loc="upper center", ncol=4, framealpha=0.95, fontsize=8.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Fig 3 to {save_path}")


def plot_convergence_trajectories(data, results_converge, save_path="../results/fig4_initial_error_convergence.png"):
    """
    Fig 4: Convergence behavior from large initial error (Initial guess = 50%, True = 90%).
    """
    horizon = min(800, len(data["time"]))  # First 800 seconds to capture DEKF t_reach (725 s) & t_settle (732 s)
    time_sec = data["time"][:horizon]
    soc_true = data["soc_true"][:horizon] * 100.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    # Highlight fast settlement zone
    ax1.axvspan(0, 10, color="#2CA02C", alpha=0.07, label="Fast Convergence (<10 s)")
    ax1.plot(time_sec, soc_true, label="True SOC", color=COLORS["True"], linewidth=2.2, linestyle="-")
    for name, res in results_converge.items():
        ax1.plot(time_sec, res["soc_est"][:horizon] * 100.0, label=name, color=COLORS.get(name, "#888888"))

    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title(r"(a) Trajectory Recovery from Initial Error ($\Delta SOC_0 = -40\%$)", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", ncol=3, framealpha=0.92, fontsize=8)
    ax1.set_ylim(40, 105)
    ax1.set_xlim(0, horizon)

    # Panel b: Absolute error
    ax2.axhspan(0, 2.0, color="#2CA02C", alpha=0.08, label="±2% Target Band")
    for name, res in results_converge.items():
        err = (res["soc_est"][:horizon] - data["soc_true"][:horizon]) * 100.0
        ax2.plot(time_sec, np.abs(err), label=name, color=COLORS.get(name, "#888888"))

    ax2.axhline(2.0, color="crimson", linestyle="--", linewidth=1.1, label="±2% Tolerance Bound")
    ax2.axvline(725.0, color="#4DAF4A", linestyle=":", linewidth=1.0, alpha=0.85)
    ax2.annotate("DEKF Entry (725 s)\n& Settle (732 s)", xy=(725.0, 2.0), xytext=(535.0, 13.5),
                 arrowprops=dict(arrowstyle="->", color="#4DAF4A", lw=0.9),
                 fontsize=7.5, color="#2E7D32", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#4DAF4A", alpha=0.92, lw=0.6))
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Absolute Error (%)")
    ax2.set_title("(b) Convergence Dynamics into ±2% Error Boundary (0–800 s Window)", loc="left", fontweight="bold")
    ax2.grid(True)
    ax2.set_ylim(0, 48)
    ax2.set_xlim(0, horizon)
    ax2.legend(loc="upper right", ncol=3, framealpha=0.92, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Fig 4 to {save_path}")


def plot_radar_summary(metrics_dict, save_path="../results/fig5_radar_tradeoff.png"):
    """
    Fig 5: Multi-attribute Radar Chart (0 to 10 score):
    - Accuracy (1 / RMSE)
    - Aging Resilience (1 / ASF)
    - Thermal Robustness
    - Convergence Speed
    - Embedded Efficiency (1 / Latency)
    """
    categories = ['Nominal\nAccuracy', 'Aging\nRobustness', 'Thermal\nRobustness', 'Convergence\nSpeed', 'Computational\nEfficiency']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 5.5), subplot_kw=dict(polar=True))

    for name, scores in metrics_dict.items():
        values = scores + scores[:1]
        col = COLORS.get(name, "#888888")
        ax.plot(angles, values, linewidth=1.8, linestyle='solid', label=name, color=col,
                marker='o', markersize=4.2, markeredgecolor="white", markeredgewidth=0.6)
        ax.fill(angles, values, color=col, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.tick_params(axis='x', pad=18)
    ax.set_xticklabels(categories, fontsize=9.0, fontweight="bold")
    ax.set_ylim(0, 11.5)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8.0, color="#64748B")
    ax.grid(True, color="#CBD5E1", linestyle="--", linewidth=0.7)
    ax.spines['polar'].set_visible(False)
    ax.set_title("Multi-Attribute Performance Trade-Off Benchmark", y=1.12, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.08), fontsize=8.5, framealpha=0.94)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Fig 5 to {save_path}")
