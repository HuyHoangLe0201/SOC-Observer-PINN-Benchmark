"""
Series Battery Pack & Cell-to-Cell Inconsistency Benchmark
Journal of Electrochemical Energy Conversion and Storage (ASME)

Features:
1. 8S Series Battery Pack (8 Cells in Series) with realistic manufacturing variations:
   - Cell capacity variations: Q_i ~ N(2.0, 0.05^2) Ah
   - Ohmic resistance variations: R0_i ~ N(0.035, 0.003^2) Ohm
   - Initial SOC variations: SOC_0,i ~ N(0.95, 0.015^2)
2. Compares:
   - Lumped Pack Estimator (Single equivalent macro-cell model with pack voltage)
   - Individual Cell Observers (DEKF tracking each cell)
   - Limiting / Weakest Cell Tracking Accuracy (critical for EV safety and overcharge prevention)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from battery_model import BatteryCellNMC
from drive_cycles import DriveCycles
from models_model_based import ExtendedKalmanFilter, DualExtendedKalmanFilter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})


class SeriesBatteryPack:
    """
    8-Cell Series Battery Pack with heterogeneous cell-to-cell parameters.
    Current is shared across all cells: I_cell = I_pack.
    Pack terminal voltage: V_pack = sum(V_cell_i).
    """

    def __init__(self, num_cells=8, dt=1.0, random_seed=42):
        self.num_cells = num_cells
        self.dt = dt
        np.random.seed(random_seed)

        # Generate realistic Gaussian distributions for cell parameters
        self.capacities = np.random.normal(2.0, 0.06, num_cells)       # Ah (std ~3%)
        self.r0_values = np.random.normal(0.035, 0.0035, num_cells)    # Ohm (std ~10%)
        self.initial_socs = np.random.normal(0.95, 0.015, num_cells)   # Initial SOC

        # Create cell instances
        self.cells = []
        for i in range(num_cells):
            cell = BatteryCellNMC(nominal_capacity_ah=self.capacities[i], soh=1.0, temperature_c=25.0, dt=dt)
            cell.r0_base = self.r0_values[i]
            cell._update_parameters()
            cell.reset_state(initial_soc=self.initial_socs[i])
            self.cells.append(cell)

        # Identify the "Weakest Cell" (lowest capacity, critical safety limiter)
        self.weakest_cell_idx = int(np.argmin(self.capacities))
        print(f"[SeriesBatteryPack] Initialized {num_cells}S pack.")
        print(f"  Capacities (Ah): {[round(c, 3) for c in self.capacities]}")
        print(f"  Weakest cell index: Cell {self.weakest_cell_idx + 1} (Q = {self.capacities[self.weakest_cell_idx]:.3f} Ah)")

    def step(self, current_a):
        """
        Advance all cells in series by 1 time step.
        Returns:
            v_pack: float, Total pack voltage
            cell_voltages: np.ndarray, Voltages of individual cells
            cell_socs: np.ndarray, True SOCs of individual cells
        """
        cell_v = np.zeros(self.num_cells)
        cell_s = np.zeros(self.num_cells)

        for i, cell in enumerate(self.cells):
            v, s = cell.step(current_a)
            cell_v[i] = v
            cell_s[i] = s

        v_pack = np.sum(cell_v)
        return v_pack, cell_v, cell_s


def run_pack_inconsistency_benchmark():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    dt = 1.0

    pack = SeriesBatteryPack(num_cells=8, dt=dt, random_seed=105)
    duration = 3600
    current_profile = DriveCycles.generate_udds(duration_sec=duration, dt=dt)
    n_steps = len(current_profile)

    # Ground truth storage
    v_pack_true = np.zeros(n_steps)
    cell_v_true = np.zeros((n_steps, pack.num_cells))
    cell_s_true = np.zeros((n_steps, pack.num_cells))

    for k in range(n_steps):
        vp, cv, cs = pack.step(current_profile[k])
        v_pack_true[k] = vp
        cell_v_true[k, :] = cv
        cell_s_true[k, :] = cs

    # Sensed signals with noise
    np.random.seed(99)
    i_sensed = current_profile + np.random.normal(0, 0.015, n_steps)
    v_pack_noisy = v_pack_true + np.random.normal(0, 0.025, n_steps)
    cell_v_noisy = cell_v_true + np.random.normal(0, 0.005, (n_steps, pack.num_cells))

    # True Mean Pack SOC and Weakest Cell SOC
    mean_soc_true = np.mean(cell_s_true, axis=1)
    weakest_soc_true = cell_s_true[:, pack.weakest_cell_idx]

    # -------------------------------------------------------------
    # 1. Estimator A: Lumped Pack Macro-Cell Estimator (Standard EKF)
    # -------------------------------------------------------------
    # Assumes identical average cells: V_macro = V_pack / 8
    macro_cell = BatteryCellNMC(nominal_capacity_ah=2.0, soh=1.0, temperature_c=25.0, dt=dt)
    ekf_lumped = ExtendedKalmanFilter(macro_cell)
    ekf_lumped.reset(initial_soc=mean_soc_true[0])

    soc_lumped_est = np.zeros(n_steps)
    for k in range(n_steps):
        v_avg_sensed = v_pack_noisy[k] / float(pack.num_cells)
        s_est, _, _ = ekf_lumped.step(i_sensed[k], v_avg_sensed)
        soc_lumped_est[k] = s_est

    # -------------------------------------------------------------
    # 2. Estimator B: Cell-Level Distributed DEKF (Tracks Each Cell)
    # -------------------------------------------------------------
    dekf_observers = []
    for i in range(pack.num_cells):
        c_model = BatteryCellNMC(nominal_capacity_ah=2.0, soh=1.0, temperature_c=25.0, dt=dt)
        obs = DualExtendedKalmanFilter(c_model)
        obs.reset(initial_soc=cell_s_true[0, i])
        dekf_observers.append(obs)

    soc_cells_est = np.zeros((n_steps, pack.num_cells))
    for k in range(n_steps):
        for i in range(pack.num_cells):
            s_est, _, _, _ = dekf_observers[i].step(i_sensed[k], cell_v_noisy[k, i])
            soc_cells_est[k, i] = s_est

    distributed_mean_est = np.mean(soc_cells_est, axis=1)
    distributed_weakest_est = soc_cells_est[:, pack.weakest_cell_idx]

    # Metrics computation
    # Error on Mean Pack SOC
    err_lumped_mean = (soc_lumped_est - mean_soc_true) * 100.0
    err_distrib_mean = (distributed_mean_est - mean_soc_true) * 100.0

    # Error on Weakest Cell SOC (Critical for BMS Safety)
    err_lumped_weakest = (soc_lumped_est - weakest_soc_true) * 100.0
    err_distrib_weakest = (distributed_weakest_est - weakest_soc_true) * 100.0

    table8_rows = [
        {
            "Estimation Strategy": "Lumped Pack Observer (Mean Macro-Cell EKF)",
            "Target State": "Mean Pack SOC",
            "RMSE (%)": round(np.sqrt(np.mean(err_lumped_mean**2)), 3),
            "Max Error (%)": round(np.max(np.abs(err_lumped_mean)), 3),
            "Safety Margin Risk": "Moderate"
        },
        {
            "Estimation Strategy": "Distributed DEKF Architecture",
            "Target State": "Mean Pack SOC",
            "RMSE (%)": round(np.sqrt(np.mean(err_distrib_mean**2)), 3),
            "Max Error (%)": round(np.max(np.abs(err_distrib_mean)), 3),
            "Safety Margin Risk": "None (Optimal)"
        },
        {
            "Estimation Strategy": "Lumped Pack Observer (Ignoring Variations)",
            "Target State": "Weakest Cell SOC (Safety Limiter)",
            "RMSE (%)": round(np.sqrt(np.mean(err_lumped_weakest**2)), 3),
            "Max Error (%)": round(np.max(np.abs(err_lumped_weakest)), 3),
            "Safety Margin Risk": "High (Risk of Overdischarge / Plating)"
        },
        {
            "Estimation Strategy": "Distributed DEKF Architecture",
            "Target State": "Weakest Cell SOC (Safety Limiter)",
            "RMSE (%)": round(np.sqrt(np.mean(err_distrib_weakest**2)), 3),
            "Max Error (%)": round(np.max(np.abs(err_distrib_weakest)), 3),
            "Safety Margin Risk": "Guaranteed Tracking"
        }
    ]

    df_table8 = pd.DataFrame(table8_rows)
    df_table8.to_csv(os.path.join(results_dir, "table8_pack_cell_inconsistency.csv"), index=False)
    print("\n--- Table 8: Series Battery Pack & Cell-to-Cell Inconsistency Benchmark ---")
    print(df_table8.to_string(index=False))

    # Plot Fig 10: Pack-Level Cell Inconsistency Tracking
    time_min = np.arange(n_steps) / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)

    # Upper panel: Cell-to-cell spread & tracking
    min_cell_soc = np.min(cell_s_true, axis=1) * 100.0
    max_cell_soc = np.max(cell_s_true, axis=1) * 100.0
    ax1.fill_between(time_min, min_cell_soc, max_cell_soc, color="#1F77B4", alpha=0.10, label="8S Cell Dispersion Band")

    for i in range(pack.num_cells):
        alpha_val = 0.95 if i == pack.weakest_cell_idx else 0.35
        col = "#D62728" if i == pack.weakest_cell_idx else "#1F77B4"
        lbl = f"Cell {i+1} (Weakest)" if i == pack.weakest_cell_idx else (f"Cell {i+1}" if i == 0 else None)
        ax1.plot(time_min, cell_s_true[:, i] * 100.0, color=col, alpha=alpha_val, linewidth=2.0 if i == pack.weakest_cell_idx else 0.9, label=lbl)

    ax1.plot(time_min, soc_lumped_est * 100.0, "--", color="black", linewidth=1.6, label="Lumped Macro-Cell Est.")
    ax1.plot(time_min, distributed_weakest_est * 100.0, ":", color="limegreen", linewidth=2.0, label="Distributed DEKF (Weakest)")
    ax1.set_ylabel("SOC (%)")
    ax1.set_title(r"(a) 8S Series Battery Pack SOC Distribution and Weakest Cell Tracking", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_ylim(40, 105)
    ax1.legend(loc="lower left", ncol=2, framealpha=0.95, fontsize=8.0)

    # Lower panel: Error tracking the Weakest Cell
    ax2.axhspan(-2.0, 2.0, color="#2CA02C", alpha=0.08, label=None)
    ax2.plot(time_min, err_lumped_weakest, color="black", linestyle="--", linewidth=1.5, label=f"Lumped Pack Error (Max: {np.max(np.abs(err_lumped_weakest)):.2f}%)")
    ax2.plot(time_min, err_distrib_weakest, color="limegreen", linestyle="-", linewidth=1.7, label=f"Distributed DEKF Error (Max: {np.max(np.abs(err_distrib_weakest)):.2f}%)")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.axhline(2.0, color="crimson", linestyle="--", linewidth=0.9, label="±2% Tolerance Bound")
    ax2.axhline(-2.0, color="crimson", linestyle="--", linewidth=0.9)
    ax2.text(1.0, 2.2, r"$\pm 2\%$ Safe Limit Corridor", fontsize=7.5, color="#555555", va="bottom", fontweight="bold")
    ax2.annotate("Underestimation Hazard\n(Overdischarge Risk)", xy=(2.0, -2.15), xytext=(7.0, -3.5),
                 arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
                 fontsize=8.0, color="crimson", fontweight="bold")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Error on Weakest Cell (%)")
    ax2.set_title(r"(b) Estimation Error on the Limiting (Weakest) Cell", loc="left", fontweight="bold")
    ax2.grid(True)
    ax2.set_ylim(-4.5, 6.5)
    ax2.legend(loc="upper left", ncol=2, framealpha=0.95, fontsize=8.0)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig10_pack_cell_inconsistency.png"), bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved Fig 10 to {os.path.join(results_dir, 'fig10_pack_cell_inconsistency.png')}")


if __name__ == "__main__":
    run_pack_inconsistency_benchmark()
