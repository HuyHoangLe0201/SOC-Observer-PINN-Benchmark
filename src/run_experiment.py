"""
Master Experiment Runner & Benchmark Orchestrator (Phases 1-5: Tier 2 Co-Simulation)
Journal of Energy Storage (Elsevier)

Strict Zero-Leakage Protocol:
- All model-based observers (EKF, AEKF, UKF, DEKF, SMO, PIDKF) initialize from
  fresh room-temperature nominal parameters (100% SOH, 25 deg C).
- All neural networks (BiLSTM-Attn, PINN-SOC, Unconstrained MLP, PIDKF compensator)
  are trained strictly on Batch A DST across 5 random seeds (42..46) and evaluated
  on strictly unseen UDDS profiles, reporting Mean +/- SD.
- Phase 5 Radar scores are deterministically computed from Tables 1-4 and MCU FLOPs.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

from benchmark_engine import BenchmarkEngine
from pidkf_hybrid import train_residual_compensator
import visualizer


def compute_radar_scores(df1, df2, df3, df4, models_radar):
    """Deterministic log/linear 1-10 normalization from Tables 1-4 + analytical MCU FLOPs."""
    # Exact single-step arithmetic FLOPs/step matching Table 8 instruction analysis
    flops_map = {
        "SMO": 165.0,
        "EKF": 410.0,
        "AEKF": 540.0,
        "DEKF": 960.0,
        "UKF": 1420.0,
        "PIDKF": 1540.0,
        "PINN-SOC": 13500.0,
        "BiLSTM-Attn": 1680000.0,
    }
    r1 = df1.set_index("Algorithm")["RMSE (%)"].to_dict()
    r2 = df2.set_index("Algorithm")["SOH=80%"].to_dict()
    r3_0 = df3.set_index("Algorithm")["0°C"].to_dict()
    r3_45 = df3.set_index("Algorithm")["45°C"].to_dict()
    r4_ts = df4.set_index("Algorithm")["t_settle_5 (s)"].to_dict()
    r4_dyn = df4.set_index("Algorithm")["MaxE_dyn (%)"].to_dict()

    scores = {}
    for m in models_radar:
        # 1. Nominal Accuracy: 10 at 0.05%, decays smoothly with RMSE
        s_nom = float(np.clip(10.0 - 2.0 * np.log10(max(r1[m], 0.05) / 0.05), 1.0, 10.0))
        # 2. Aging Robustness: based on worst aged RMSE at 80% SOH
        s_age = float(np.clip(10.0 - 1.6 * r2[m], 1.0, 10.0))
        # 3. Thermal Robustness: mean error across 0 C and 45 C extremes
        r_th = 0.5 * (r3_0[m] + r3_45[m])
        s_thm = float(np.clip(10.0 - 1.25 * r_th, 1.0, 10.0))
        # 4. Convergence Speed & Post-Entry Stability: t_settle^(5%) and MaxE_dyn
        ts_val = r4_ts[m] if not np.isnan(r4_ts[m]) else 1800.0
        dyn_val = r4_dyn[m]
        s_conv = float(np.clip(10.0 - (ts_val / 220.0) - 0.35 * dyn_val, 1.0, 10.0))
        # 5. Embedded Efficiency: log10 scale vs BiLSTM/MLP (165 FLOPs -> 9.8, 1.68M FLOPs -> 1.5)
        fl = flops_map.get(m, 2000.0)
        s_eff = float(np.clip(10.0 - 2.1 * np.log10(fl / 150.0), 1.0, 10.0))

        scores[m] = [round(s_nom, 2), round(s_age, 2), round(s_thm, 2), round(s_conv, 2), round(s_eff, 2)]
    return scores


def main():
    print("=" * 84)
    print("BATTERY SOC BENCHMARK EXPERIMENT RUNNER (PHASES 1-5, TIER 2 CO-SIMULATION)")
    print("=" * 84)

    seeds = (42, 43, 44, 45, 46)
    np.random.seed(42)
    torch.manual_seed(42)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    engine = BenchmarkEngine(dt=1.0, results_dir=results_dir)

    # 1. Train 5 seeds of neural baselines (BiLSTM, PINN-SOC, Unconstrained MLP) + PIDKF compensator strictly on Batch A DST
    engine.train_deep_models(epochs=30, lr=0.002, seeds=seeds, chemistry="NMC")
    pidkf_models = []
    print("[Runner] Training 5 seeds of PIDKF causal residual compensator on Batch A DST (NMC)...")
    for s in seeds:
        pm, _ = train_residual_compensator(chemistry="NMC", seed=s, epochs=600)
        pidkf_models.append(pm)

    models_order = ["EKF", "AEKF", "UKF", "DEKF", "SMO", "BiLSTM-Attn", "PINN-SOC", "MLP (Unconstrained)", "PIDKF"]

    # -------------------------------------------------------------
    # PHASE 1: Dynamic Cycle Benchmark (UDDS @ 25 deg C, SOH=100%)
    # -------------------------------------------------------------
    print("\n[Phase 1] Dynamic Cycle Benchmark (UDDS @ 25 C, SOH=100%)...")
    udds_data = engine.generate_simulation_data(cycle_name="udds", duration=3600, soh=1.0, temp_c=25.0, chemistry="NMC", initial_soc=0.95)
    exp1_res = engine.run_benchmark_on_dataset(udds_data, include_ukf=True, include_mlp=True, pidkf_models=pidkf_models)

    table1_rows = []
    for m in models_order:
        r = exp1_res[m]
        table1_rows.append({
            "Algorithm": m,
            "RMSE (%)": round(r["rmse_pct"], 3),
            "RMSE SD (%)": round(r.get("rmse_sd", 0.0), 3),
            "MAE (%)": round(r["mae_pct"], 3),
            "MaxE (%)": round(r["max_err_pct"], 3),
        })
    df_exp1 = pd.DataFrame(table1_rows)
    df_exp1.to_csv(os.path.join(results_dir, "table1_dynamic_udds.csv"), index=False)
    print("\n--- Table 1: Performance Summary on Unseen UDDS Cycle at 25 C (NMC) ---")
    print(df_exp1.to_string(index=False))

    visualizer.plot_tracking_and_errors(udds_data, exp1_res, save_path=os.path.join(results_dir, "fig3.png"))

    # -------------------------------------------------------------
    # PHASE 2: Degradation & Aging Robustness (SOH: 1.0 -> 0.8)
    # -------------------------------------------------------------
    print("\n[Phase 2] Progressive Aging Degradation Benchmark...")
    soh_levels = np.array([1.00, 0.95, 0.90, 0.85, 0.80])
    rmse_aging = {m: [] for m in models_order}
    sd_aging = {m: [] for m in models_order}

    for soh in soh_levels:
        print(f"  Testing SOH = {soh * 100:.0f}%...")
        data_aging = engine.generate_simulation_data(cycle_name="udds", duration=3600, soh=soh, temp_c=25.0, chemistry="NMC", initial_soc=0.95)
        res_aging = engine.run_benchmark_on_dataset(data_aging, include_ukf=True, include_mlp=True, pidkf_models=pidkf_models)
        for m in models_order:
            rmse_aging[m].append(round(res_aging[m]["rmse_pct"], 3))
            sd_aging[m].append(round(res_aging[m].get("rmse_sd", 0.0), 3))

    table2_rows = []
    for m in models_order:
        fresh_r = rmse_aging[m][0]
        aged_r = rmse_aging[m][-1]
        asf = ((aged_r - fresh_r) / max(fresh_r, 1e-6)) * 100.0
        table2_rows.append({
            "Algorithm": m,
            "SOH=100%": fresh_r,
            "SOH=95%": rmse_aging[m][1],
            "SOH=90%": rmse_aging[m][2],
            "SOH=85%": rmse_aging[m][3],
            "SOH=80%": aged_r,
            "80% SD": sd_aging[m][-1],
            "ASF (%)": round(asf, 1),
        })
    df_exp2 = pd.DataFrame(table2_rows)
    df_exp2.to_csv(os.path.join(results_dir, "table2_aging_degradation.csv"), index=False)
    print("\n--- Table 2: Degradation Robustness across Cell Aging (RMSE %) ---")
    print(df_exp2.to_string(index=False))

    visualizer.plot_aging_degradation_trend(soh_levels, rmse_aging, save_path=os.path.join(results_dir, "fig4.png"))

    # -------------------------------------------------------------
    # PHASE 3: Thermal Stress Robustness (0 C, 25 C, 45 C)
    # -------------------------------------------------------------
    print("\n[Phase 3] Thermal Stress Benchmark (0 C, 25 C, 45 C)...")
    temp_levels = ["0°C", "25°C", "45°C"]
    temp_values = [0.0, 25.0, 45.0]
    rmse_thermal = {m: {} for m in models_order}
    sd_thermal = {m: {} for m in models_order}

    for t_str, t_val in zip(temp_levels, temp_values):
        print(f"  Testing Temp = {t_str}...")
        data_temp = engine.generate_simulation_data(cycle_name="udds", duration=3600, soh=1.0, temp_c=t_val, chemistry="NMC", initial_soc=0.95)
        res_temp = engine.run_benchmark_on_dataset(data_temp, include_ukf=True, include_mlp=True, pidkf_models=pidkf_models)
        for m in models_order:
            rmse_thermal[m][t_str] = round(res_temp[m]["rmse_pct"], 3)
            sd_thermal[m][t_str] = round(res_temp[m].get("rmse_sd", 0.0), 3)

    table3_rows = []
    for m in models_order:
        table3_rows.append({
            "Algorithm": m,
            "0°C": rmse_thermal[m]["0°C"],
            "0°C SD": sd_thermal[m]["0°C"],
            "25°C": rmse_thermal[m]["25°C"],
            "45°C": rmse_thermal[m]["45°C"],
            "45°C SD": sd_thermal[m]["45°C"],
        })
    df_exp3 = pd.DataFrame(table3_rows)
    df_exp3.to_csv(os.path.join(results_dir, "table3_thermal_robustness.csv"), index=False)
    print("\n--- Table 3: Thermal Stress Robustness across Temperatures (RMSE %) ---")
    print(df_exp3.to_string(index=False))

    visualizer.plot_thermal_regimes_bar(temp_levels, rmse_thermal, save_path=os.path.join(results_dir, "fig5.png"))

    # -------------------------------------------------------------
    # PHASE 4: Initial SOC Uncertainty Convergence (Delta SOC = -40%)
    # -------------------------------------------------------------
    print("\n[Phase 4] Initial SOC Uncertainty Convergence Test (Delta SOC = -40%)...")
    data_conv = engine.generate_simulation_data(cycle_name="udds", duration=1800, soh=1.0, temp_c=25.0, chemistry="NMC", initial_soc=0.95)
    res_conv = engine.run_benchmark_on_dataset(data_conv, initial_soc_guess=0.55, include_ukf=True, include_mlp=True, pidkf_models=pidkf_models)

    table4_rows = []
    for m in models_order:
        r = res_conv[m]
        table4_rows.append({
            "Algorithm": m,
            "t_reach_2 (s)": r["t_reach_2"],
            "t_settle_5 (s)": r["t_settle_5"],
            "RMSE (%)": round(r["rmse_pct"], 3),
            "MaxE (%)": round(r["max_err_pct"], 2),
            "MaxE_dyn (%)": round(r["max_dyn_pct"], 2),
        })
    df_exp4 = pd.DataFrame(table4_rows)
    df_exp4.to_csv(os.path.join(results_dir, "table4_convergence.csv"), index=False)
    print("\n--- Table 4: Convergence Recovery from -40% Initial Offset ---")
    print(df_exp4.to_string(index=False))

    visualizer.plot_convergence_trajectories(data_conv, res_conv, save_path=os.path.join(results_dir, "fig6.png"))

    # -------------------------------------------------------------
    # PHASE 5: Deterministically Computed Multi-Attribute Radar Scores
    # -------------------------------------------------------------
    print("\n[Phase 5] Computing Data-Driven Multi-Attribute Radar Benchmark...")
    radar_models = ["EKF", "AEKF", "UKF", "DEKF", "SMO", "BiLSTM-Attn", "PINN-SOC", "PIDKF"]
    radar_scores = compute_radar_scores(df_exp1, df_exp2, df_exp3, df_exp4, radar_models)
    df_exp5 = pd.DataFrame([
        {
            "Algorithm": m,
            "Nominal Accuracy": v[0],
            "Aging Robustness": v[1],
            "Thermal Robustness": v[2],
            "Convergence Speed": v[3],
            "Embedded Efficiency": v[4],
        }
        for m, v in radar_scores.items()
    ])
    df_exp5.to_csv(os.path.join(results_dir, "table5_radar_scores.csv"), index=False)
    print(df_exp5.to_string(index=False))
    visualizer.plot_radar_summary(radar_scores, save_path=os.path.join(results_dir, "fig7.png"))

    print("\n" + "=" * 84)
    print("PHASES 1-5 COMPLETED CLEANLY!")
    print("=" * 84)


if __name__ == "__main__":
    main()
