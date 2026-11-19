"""
Advanced Benchmarks Runner: Phases 6, 7 & 10 (Tier 2 Calibrated Co-Simulation)
Journal of Energy Storage (Elsevier)

Executes strictly fair, zero-leakage multi-seed benchmarks:
- Phase 6: Cross-Chemistry NMC vs. LFP Comparison on IDENTICAL unseen UDDS & FUDS
           schedules (nominal 100% SOH @ 25 C & aged 85% SOH @ 25 C) with
           4-state dynamic hysteresis active on LFP.
- Phase 7: Sensor Bias Drift (+0.25 A Hall offset) & 25 mV Inverter EMI Stress
           across Coulomb Counting, EKF, AEKF, DEKF, SMO, and PIDKF
           (reporting RMSE, MaxE, and final-60s window-averaged drift).
- Phase 10: Comprehensive 5-Seed Ablation (Batch A DST -> unseen Batch B UDDS/FUDS)
            on Aged NMC & Aged LFP (85% SOH), plus exact vs. exogenous Jacobian
            discrepancy verification and Fig. 11 generation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from battery_model import BatteryCellNMC, BatteryCellLFP
from benchmark_engine import BenchmarkEngine, compute_metrics
from models_model_based import (
    ExtendedKalmanFilter,
    AdaptiveEKF,
    UnscentedKalmanFilter,
    DualExtendedKalmanFilter,
    SlidingModeObserver,
)
from pidkf_hybrid import PhysicsInformedDualKalmanFilter, train_residual_compensator
from pub_style import PALETTE, MARKERS, apply_pub_style, save_pub_figure, style_legend

apply_pub_style()


def run_phase6_cross_chemistry(engine, pidkf_nmc, pidkf_lfp, results_dir):
    print("\n[Phase 6] Cross-Chemistry NMC vs. LFP Benchmark on Unseen UDDS & FUDS (5-Seed)...")
    # Generate identical unseen UDDS drive cycles at 25 C (100% and 85% SOH)
    nmc_100 = engine.generate_simulation_data("udds", duration=3600, soh=1.00, temp_c=25.0, chemistry="NMC", initial_soc=0.95)
    lfp_100 = engine.generate_simulation_data("udds", duration=3600, soh=1.00, temp_c=25.0, chemistry="LFP", initial_soc=0.95)
    lfp_fuds = engine.generate_simulation_data("fuds", duration=3600, soh=1.00, temp_c=25.0, chemistry="LFP", initial_soc=0.95)

    # Evaluate fresh benchmarks cleanly
    res_nmc = engine.run_benchmark_on_dataset(nmc_100, include_ukf=True, pidkf_models=pidkf_nmc)
    res_lfp = engine.run_benchmark_on_dataset(lfp_100, include_ukf=True, pidkf_models=pidkf_lfp)

    # Also evaluate LFP EKF & DEKF WITHOUT hysteresis compensation as diagnostic control
    ekf_nohyst = ExtendedKalmanFilter(BatteryCellLFP(2.30, soh=1.0, temperature_c=25.0), use_hysteresis=False)
    ekf_nohyst.reset(0.95)
    t_ekf_nh = np.array([ekf_nohyst.step(lfp_100["current_sensed"][k], lfp_100["v_noisy"][k])[0] for k in range(len(lfp_100["time"]))])
    m_ekf_nh = compute_metrics(t_ekf_nh, lfp_100["soc_true"])

    res_lfp_fuds = engine.run_benchmark_on_dataset(lfp_fuds, include_ukf=True, pidkf_models=pidkf_lfp)

    models_p6 = ["EKF", "AEKF", "UKF", "DEKF", "SMO", "BiLSTM-Attn", "PINN-SOC", "PIDKF"]
    rows = []
    for m in models_p6:
        rows.append({
            "Algorithm": m,
            "NMC UDDS RMSE (%)": round(res_nmc[m]["rmse_pct"], 3),
            "NMC UDDS SD": round(res_nmc[m].get("rmse_sd", 0.0), 3),
            "LFP UDDS RMSE (%)": round(res_lfp[m]["rmse_pct"], 3),
            "LFP UDDS SD": round(res_lfp[m].get("rmse_sd", 0.0), 3),
            "LFP UDDS MaxE (%)": round(res_lfp[m]["max_err_pct"], 3),
            "LFP FUDS RMSE (%)": round(res_lfp_fuds[m]["rmse_pct"], 3),
        })
    rows.append({
        "Algorithm": "EKF (No Hyst. Control)",
        "NMC UDDS RMSE (%)": np.nan,
        "NMC UDDS SD": 0.0,
        "LFP UDDS RMSE (%)": round(m_ekf_nh["rmse_pct"], 3),
        "LFP UDDS SD": 0.0,
        "LFP UDDS MaxE (%)": round(m_ekf_nh["max_err_pct"], 3),
        "LFP FUDS RMSE (%)": np.nan,
    })

    df_p6 = pd.DataFrame(rows)
    df_p6.to_csv(os.path.join(results_dir, "table6_cross_chemistry.csv"), index=False)
    print(df_p6.to_string(index=False))

    # Plot Fig 6: Cross-Chemistry LFP Plateau Tracking vs NMC on Unseen UDDS
    time_min = lfp_100["time"] / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.6), gridspec_kw={"height_ratios": [1.05, 1.05], "hspace": 0.28})

    # Top panel: LFP UDDS Error trajectories (PIDKF, DEKF, EKF 4-state, EKF No-Hyst, BiLSTM, PINN)
    ax1.axhspan(-2.0, 2.0, color="#16A34A", alpha=0.075, label=r"$\pm 2\%$ Corridor", zorder=0)
    ax1.axhline(0.0, color="#111827", linestyle="--", linewidth=0.75, zorder=1)
    ax1.plot(time_min, (t_ekf_nh - lfp_100["soc_true"]) * 100.0, label=f"EKF (No Hyst: {m_ekf_nh['rmse_pct']:.2f}%)", color="#9CA3AF", linestyle="-.", linewidth=1.25, zorder=2)
    for m in ["BiLSTM-Attn", "PINN-SOC", "SMO", "AEKF", "EKF", "DEKF", "PIDKF"]:
        err = (res_lfp[m]["soc_est"] - lfp_100["soc_true"]) * 100.0
        ax1.plot(
            time_min, err,
            label=f"{m} ({res_lfp[m]['rmse_pct']:.2f}%)",
            color=PALETTE.get(m, "#64748B"),
            linewidth=1.8 if m == "PIDKF" else (1.1 if m in ("BiLSTM-Attn", "PINN-SOC") else 1.4),
            alpha=0.72 if m in ("BiLSTM-Attn", "PINN-SOC") else 0.94,
            zorder=7 if m == "PIDKF" else 4,
        )
    ax1.set_ylabel("SOC Error (%)")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_title(r"(a) Flat-Plateau LFP Tracking Error on Unseen UDDS ($25^\circ\mathrm{C}$, SOH $= 100\%$, 4-State Hysteresis)", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_xlim(0.0, 60.5)
    style_legend(ax1.legend(loc="upper right", ncol=3, fontsize=6.6, columnspacing=0.7, handlelength=1.4))

    # Bottom panel: Paired Bar Chart NMC vs LFP UDDS RMSE across estimators
    xm = np.arange(len(models_p6))
    w = 0.36
    nmc_vals = [res_nmc[m]["rmse_pct"] for m in models_p6]
    lfp_vals = [res_lfp[m]["rmse_pct"] for m in models_p6]
    b1 = ax2.bar(xm - w / 2, nmc_vals, w, label="NMC (Sloping OCV)", color="#2563EB", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    b2 = ax2.bar(xm + w / 2, lfp_vals, w, label="LFP (Flat Plateau + Hysteresis)", color="#0F766E", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    max_b = max(max(nmc_vals), max(lfp_vals))
    for bar, val in list(zip(b1, nmc_vals)) + list(zip(b2, lfp_vals)):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + max_b * 0.02, f"{val:.2f}%", ha="center", va="bottom", rotation=90, fontsize=6.8, fontweight="bold")
    ax2.set_xticks(xm)
    ax2.set_xticklabels(models_p6, rotation=15, ha="right", fontweight="bold", fontsize=8.0)
    ax2.set_ylabel("UDDS RMSE (%)")
    ax2.set_ylim(0.0, max_b * 1.32 + 0.4)
    ax2.set_title("(b) Direct Cross-Chemistry Error Comparison on Identical UDDS Schedule", loc="left", fontweight="bold")
    ax2.grid(True, axis="y")
    style_legend(ax2.legend(loc="upper left", ncol=2, fontsize=7.6))

    save_pub_figure(fig, os.path.join(results_dir, "fig8.png"))
    print("[Phase 6] Saved fig8.png")
    return df_p6


def run_phase7_sensor_bias(engine, pidkf_nmc, results_dir):
    print("\n[Phase 7] Sensor Bias Drift (+0.25 A Hall Bias) + 25 mV EMI Stress (2-Hour UDDS)...")
    data_bias = engine.generate_simulation_data(
        cycle_name="udds", duration=7200, soh=1.0, temp_c=25.0, chemistry="NMC",
        initial_soc=0.95, i_bias=0.25, v_noise_std=0.025, i_noise_std=0.020, seed=77
    )
    currents = data_bias["current_sensed"]
    voltages = data_bias["v_noisy"]
    temps = data_bias["temp_c"]
    soc_true = data_bias["soc_true"]
    n = len(currents)

    # 1. Open-loop Coulomb Counting (drifts steadily via dt * +0.25 A bias)
    cc_est = np.zeros(n)
    cc_est[0] = 0.95
    for k in range(1, n):
        eta = 1.0 if currents[k] >= 0.0 else 0.995
        cc_est[k] = np.clip(cc_est[k - 1] - (eta * currents[k] / (3600.0 * 2.00)), 0.0, 1.0)

    def mk_cell():
        return BatteryCellNMC(nominal_capacity_ah=2.00, soh=1.0, temperature_c=25.0, dt=1.0)

    obs_map = {
        "Coulomb Counting": cc_est,
    }
    for name, obs in [
        ("EKF", ExtendedKalmanFilter(mk_cell())),
        ("AEKF", AdaptiveEKF(mk_cell())),
        ("DEKF", DualExtendedKalmanFilter(mk_cell())),
        ("SMO", SlidingModeObserver(mk_cell())),
    ]:
        obs.reset(0.95)
        obs_map[name] = np.array([obs.step(currents[k], voltages[k])[0] for k in range(n)])

    # Multi-seed PIDKF under bias + EMI
    pid_trajs = []
    pid_rows = []
    for pnn in pidkf_nmc:
        p = PhysicsInformedDualKalmanFilter(mk_cell(), nn_model=pnn)
        p.reset(0.95)
        tr = np.array([p.step(currents[k], voltages[k], temp_c=float(temps[k]))[0] for k in range(n)])
        pid_trajs.append(tr)
        m = compute_metrics(tr, soc_true)
        m["end60_pct"] = float(np.mean(np.abs((tr[-60:] - soc_true[-60:]) * 100.0)))
        pid_rows.append(m)
    obs_map["PIDKF"] = pid_trajs[0]

    rows = []
    for name in ["Coulomb Counting", "EKF", "AEKF", "DEKF", "SMO"]:
        tr = obs_map[name]
        m = compute_metrics(tr, soc_true)
        end60 = float(np.mean(np.abs((tr[-60:] - soc_true[-60:]) * 100.0)))
        rows.append({
            "Algorithm": name,
            "RMSE (%)": round(m["rmse_pct"], 3),
            "RMSE SD": 0.0,
            "MaxE (%)": round(m["max_err_pct"], 3),
            "Final-60s Drift (%)": round(end60, 3),
        })
    rows.append({
        "Algorithm": "PIDKF",
        "RMSE (%)": round(float(np.mean([r["rmse_pct"] for r in pid_rows])), 3),
        "RMSE SD": round(float(np.std([r["rmse_pct"] for r in pid_rows])), 3),
        "MaxE (%)": round(float(np.mean([r["max_err_pct"] for r in pid_rows])), 3),
        "Final-60s Drift (%)": round(float(np.mean([r["end60_pct"] for r in pid_rows])), 3),
    })

    df_p7 = pd.DataFrame(rows)
    df_p7.to_csv(os.path.join(results_dir, "table7_sensor_corruption.csv"), index=False)
    print(df_p7.to_string(index=False))

    # Plot Fig 7: Error trajectories under Hall +0.25A bias & 25mV EMI
    t_min = data_bias["time"] / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.4), gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.28})
    ax1.plot(t_min, soc_true * 100.0, label="True SOC", color="#111827", lw=2.0, zorder=7)
    for name in ["Coulomb Counting", "SMO", "AEKF", "EKF", "DEKF", "PIDKF"]:
        ax1.plot(t_min, obs_map[name] * 100.0, label=name, color=PALETTE.get(name, "#64748B"), lw=1.7 if name == "PIDKF" else 1.25, zorder=6 if name == "PIDKF" else 3)
    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title(r"(a) Closed-Loop Rejection vs. Open-Loop Runaway ($+0.25\;\mathrm{A}$ Bias + $25\;\mathrm{mV}$ EMI, 2 h)", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_xlim(0.0, 120.5)
    style_legend(ax1.legend(loc="lower left", ncol=4, fontsize=6.9))

    ax2.axhspan(-2.0, 2.0, color="#16A34A", alpha=0.075, zorder=0)
    ax2.axhline(0.0, color="#111827", linestyle="--", lw=0.75)
    for name in ["Coulomb Counting", "SMO", "AEKF", "EKF", "DEKF", "PIDKF"]:
        err = (obs_map[name] - soc_true) * 100.0
        lbl = f"{name} ({df_p7.set_index('Algorithm').loc[name, 'RMSE (%)']:.2f}%)"
        ax2.plot(t_min, err, label=lbl, color=PALETTE.get(name, "#64748B"), lw=1.7 if name == "PIDKF" else 1.25, zorder=6 if name == "PIDKF" else 3)
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("SOC Error (%)")
    ax2.set_title("(b) Bias Drift Containment across Closed-Loop Estimators (Bounded vs. Divergent Integration)", loc="left", fontweight="bold")
    ax2.grid(True)
    ax2.set_xlim(0.0, 120.5)
    style_legend(ax2.legend(loc="lower left", ncol=3, fontsize=6.8))

    save_pub_figure(fig, os.path.join(results_dir, "fig9.png"))
    print("[Phase 7] Saved fig9.png")
    return df_p7


def run_phase10_ablation_and_jacobian(engine, seeds, results_dir):
    print("\n[Phase 10] Full 5-Seed PIDKF Ablation Study (Aged 85% SOH NMC & LFP, Batch A -> Batch B)...")

    # Pretrain all ablation NN variants strictly on Batch A DST across 5 seeds
    variants_nn = {}
    for chem in ("NMC", "LFP"):
        variants_nn[chem] = {"blackbox": [], "phys_reg": [], "full_pidkf": []}
        for s in seeds:
            m_bb, _ = train_residual_compensator(chemistry=chem, seed=s, epochs=600, unconstrained_blackbox=True)
            m_pr, _ = train_residual_compensator(chemistry=chem, seed=s, epochs=600, lambda_phys=0.50, unconstrained_blackbox=False)
            variants_nn[chem]["blackbox"].append(m_bb)
            variants_nn[chem]["phys_reg"].append(m_pr)
            variants_nn[chem]["full_pidkf"].append(m_pr)  # identical trained physics weights, paired with hard passivity at inference

    # Strictly unseen Batch B tests: UDDS (2h) at Aged 85% SOH & 25 C (plus 0 C NMC cold-aged stress check)
    nmc_85 = engine.generate_simulation_data("udds", duration=7200, soh=0.85, temp_c=25.0, chemistry="NMC", initial_soc=0.95)
    lfp_85 = engine.generate_simulation_data("udds", duration=7200, soh=0.85, temp_c=25.0, chemistry="LFP", initial_soc=0.95)
    nmc_cold_85 = engine.generate_simulation_data("udds", duration=3600, soh=0.85, temp_c=0.0, chemistry="NMC", initial_soc=0.95)

    def eval_variant(data, variant_name, chem):
        def mk_nom():
            if chem == "LFP":
                return BatteryCellLFP(2.30, soh=1.0, temperature_c=25.0, dt=1.0)
            return BatteryCellNMC(2.00, soh=1.0, temperature_c=25.0, dt=1.0)

        curr, volt, temp, s_true = data["current_sensed"], data["v_noisy"], data["temp_c"], data["soc_true"]
        n = len(curr)

        if variant_name == "Standard EKF":
            obs = ExtendedKalmanFilter(mk_nom(), use_hysteresis=(chem == "LFP"))
            obs.reset(0.95)
            tr = np.array([obs.step(curr[k], volt[k])[0] for k in range(n)])
            m = compute_metrics(tr, s_true)
            return m["rmse_pct"], 0.0, m["max_err_pct"]

        if variant_name == "Baseline DEKF (No Hyst.)":
            obs = DualExtendedKalmanFilter(mk_nom(), use_hysteresis=False)
            obs.reset(0.95)
            tr = np.array([obs.step(curr[k], volt[k])[0] for k in range(n)])
            m = compute_metrics(tr, s_true)
            return m["rmse_pct"], 0.0, m["max_err_pct"]

        if variant_name == "DEKF + Hysteresis":
            obs = DualExtendedKalmanFilter(mk_nom(), use_hysteresis=True)
            obs.reset(0.95)
            tr = np.array([obs.step(curr[k], volt[k])[0] for k in range(n)])
            m = compute_metrics(tr, s_true)
            return m["rmse_pct"], 0.0, m["max_err_pct"]

        rmses, maxes = [], []
        key_nn = "blackbox" if "Black-Box" in variant_name else "phys_reg"
        hard_p = "PIDKF" in variant_name
        exact_j = "Exact" in variant_name
        for pm in variants_nn[chem][key_nn]:
            obs = PhysicsInformedDualKalmanFilter(
                mk_nom(), nn_model=pm, use_hysteresis=True, exact_jacobian=exact_j, hard_passivity=hard_p
            )
            obs.reset(0.95)
            tr = np.array([obs.step(curr[k], volt[k], temp_c=float(temp[k]))[0] for k in range(n)])
            m = compute_metrics(tr, s_true)
            rmses.append(m["rmse_pct"])
            maxes.append(m["max_err_pct"])
        return float(np.mean(rmses)), float(np.std(rmses)), float(np.mean(maxes))

    ablation_configs = [
        "Standard EKF",
        "Baseline DEKF (No Hyst.)",
        "DEKF + Hysteresis",
        "DEKF + Black-Box MLP",
        "DEKF + Phys-Reg MLP (No Clamp)",
        "Full PIDKF (Exact Jac.)",
        "Full PIDKF (Exog. Approx.)",
    ]

    rows = []
    for cfg in ablation_configs:
        nmc_r, nmc_sd, nmc_mx = eval_variant(nmc_85, cfg, "NMC")
        lfp_r, lfp_sd, lfp_mx = eval_variant(lfp_85, cfg, "LFP")
        c_r, c_sd, _ = eval_variant(nmc_cold_85, cfg, "NMC")
        rows.append({
            "Variant": cfg,
            "NMC 85% RMSE (%)": round(nmc_r, 3),
            "NMC SD": round(nmc_sd, 3),
            "NMC MaxE (%)": round(nmc_mx, 3),
            "LFP 85% RMSE (%)": round(lfp_r, 3),
            "LFP SD": round(lfp_sd, 3),
            "LFP MaxE (%)": round(lfp_mx, 3),
            "NMC 85% 0C RMSE (%)": round(c_r, 3),
            "NMC 0C SD": round(c_sd, 3),
        })

    df_p10 = pd.DataFrame(rows)
    df_p10.to_csv(os.path.join(results_dir, "table9_pidkf_ablation.csv"), index=False)
    print(df_p10.to_string(index=False))

    # Extract exact autograd chain-rule vs. exogenous Jacobian discrepancy directly from PhysicsInformedDualKalmanFilter
    p_ex = PhysicsInformedDualKalmanFilter(BatteryCellNMC(2.0, soh=1.0, temperature_c=25.0), nn_model=variants_nn["NMC"]["full_pidkf"][0], exact_jacobian=True)
    p_ex.reset(0.95)
    for k in range(len(nmc_85["time"])):
        p_ex.step(nmc_85["current_sensed"][k], nmc_85["v_noisy"][k], temp_c=float(nmc_85["temp_c"][k]))
    jac_ratios = np.array(p_ex.rel_jac_diffs, dtype=float)
    print(f"\n[Phase 10 Jacobian Check] Relative ||J_NN|| / ||H_ECM||: mean={np.mean(jac_ratios):.4f}, max={np.max(jac_ratios):.4f}")

    # Plot Fig 11 strictly from df_p10 (grouped bar ablation with error bars + thermal cold-aged inset panel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"wspace": 0.30})
    short_lbls = ["EKF", "DEKF\nBase", "DEKF+\nHyst.", "DEKF+\nBB MLP", "DEKF+\nPhysReg", "PIDKF\nExact", "PIDKF\nExog."]
    x = np.arange(len(short_lbls))
    w = 0.36
    y_nmc = df_p10["NMC 85% RMSE (%)"].values
    sd_nmc = df_p10["NMC SD"].values
    y_lfp = df_p10["LFP 85% RMSE (%)"].values
    sd_lfp = df_p10["LFP SD"].values

    b1 = ax1.bar(x - w / 2, y_nmc, w, yerr=sd_nmc, capsize=2.5, label="Aged NMC (85% SOH, UDDS)", color="#2563EB", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    b2 = ax1.bar(x + w / 2, y_lfp, w, yerr=sd_lfp, capsize=2.5, label="Aged LFP (85% SOH, UDDS)", color="#0F766E", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    max_y1 = max(np.max(y_nmc + sd_nmc), np.max(y_lfp + sd_lfp))
    for bar, val, sd in list(zip(b1, y_nmc, sd_nmc)) + list(zip(b2, y_lfp, sd_lfp)):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + sd + max_y1 * 0.03, f"{val:.2f}%", ha="center", va="bottom", rotation=90, fontsize=6.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_lbls, fontsize=6.9, fontweight="bold")
    ax1.set_ylabel("Unseen UDDS RMSE (%) [5-Seed Mean +/- SD]")
    ax1.set_ylim(0.0, max_y1 * 1.38 + 0.35)
    ax1.set_title("(a) Ablation on Aged NMC & LFP (85% SOH @ 25 C)", fontweight="bold", fontsize=8.8)
    ax1.grid(True, axis="y")
    style_legend(ax1.legend(loc="upper right", fontsize=7.0))

    y_c = df_p10["NMC 85% 0C RMSE (%)"].values
    sd_c = df_p10["NMC 0C SD"].values
    bc = ax2.bar(x, y_c, 0.56, yerr=sd_c, capsize=3.0, color=["#DC2626", "#64748B", "#16A34A", "#EA580C", "#854D0E", "#0284C7", "#0F766E"], edgecolor="#1F2937", lw=0.65, alpha=0.90)
    max_y2 = float(np.max(y_c + sd_c))
    for bar, val, sd in zip(bc, y_c, sd_c):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + sd + max_y2 * 0.03, f"{val:.2f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax2.axhline(2.0, color="#DC2626", linestyle="--", lw=1.0, label="2.0% BMS Bound")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_lbls, fontsize=6.9, fontweight="bold")
    ax2.set_ylabel("Cold-Aged UDDS RMSE (%) [0 C, 85% SOH]")
    ax2.set_ylim(0.0, max_y2 * 1.25 + 0.4)
    ax2.set_title("(b) Combined Cold + Aging Stress (0 C, 85% SOH)", fontweight="bold", fontsize=8.8)
    ax2.grid(True, axis="y")
    style_legend(ax2.legend(loc="upper right", fontsize=7.2))

    save_pub_figure(fig, os.path.join(results_dir, "fig13.png"))
    print("[Phase 10] Saved fig13.png")
    return df_p10


def main():
    print("=" * 84)
    print("ADVANCED BENCHMARKS RUNNER (PHASES 6, 7 & 10, ZERO-LEAKAGE PROTOCOL)")
    print("=" * 84)
    seeds = (42, 43, 44, 45, 46)
    np.random.seed(42)
    torch.manual_seed(42)

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    engine = BenchmarkEngine(dt=1.0, results_dir=results_dir)

    # Train 5 seeds of neural baselines & PIDKF strictly on Batch A DST (NMC & LFP independently)
    engine.train_deep_models(epochs=30, lr=0.002, seeds=seeds, chemistry="NMC")
    b_lfp, p_lfp, m_lfp = engine.train_deep_models(epochs=30, lr=0.002, seeds=seeds, chemistry="LFP")

    pidkf_nmc = [train_residual_compensator("NMC", seed=s, epochs=600)[0] for s in seeds]
    pidkf_lfp = [train_residual_compensator("LFP", seed=s, epochs=600)[0] for s in seeds]

    # Phase 6 uses chemistry-specific trained models
    # Store NMC inside engine for Phase 6 NMC call, pass LFP cleanly via temporary swap
    nmc_b, nmc_p, nmc_m = engine.bilstm_models, engine.pinn_models, engine.mlp_models

    # Evaluate Phase 6 manually handling both chemistries cleanly
    print("\n[Phase 6] Evaluating Cross-Chemistry (NMC vs. LFP)...")
    nmc_100 = engine.generate_simulation_data("udds", duration=3600, soh=1.00, temp_c=25.0, chemistry="NMC", initial_soc=0.95)
    lfp_100 = engine.generate_simulation_data("udds", duration=3600, soh=1.00, temp_c=25.0, chemistry="LFP", initial_soc=0.95)
    lfp_fuds = engine.generate_simulation_data("fuds", duration=3600, soh=1.00, temp_c=25.0, chemistry="LFP", initial_soc=0.95)

    res_nmc = engine.run_benchmark_on_dataset(nmc_100, include_ukf=True, pidkf_models=pidkf_nmc)
    engine.bilstm_models, engine.pinn_models, engine.mlp_models = b_lfp, p_lfp, m_lfp
    res_lfp = engine.run_benchmark_on_dataset(lfp_100, include_ukf=True, pidkf_models=pidkf_lfp)
    res_lfp_fuds = engine.run_benchmark_on_dataset(lfp_fuds, include_ukf=True, pidkf_models=pidkf_lfp)
    engine.bilstm_models, engine.pinn_models, engine.mlp_models = nmc_b, nmc_p, nmc_m

    ekf_nohyst = ExtendedKalmanFilter(BatteryCellLFP(2.30, soh=1.0, temperature_c=25.0), use_hysteresis=False)
    ekf_nohyst.reset(0.95)
    t_ekf_nh = np.array([ekf_nohyst.step(lfp_100["current_sensed"][k], lfp_100["v_noisy"][k])[0] for k in range(len(lfp_100["time"]))])
    m_ekf_nh = compute_metrics(t_ekf_nh, lfp_100["soc_true"])

    models_p6 = ["EKF", "AEKF", "UKF", "DEKF", "SMO", "BiLSTM-Attn", "PINN-SOC", "PIDKF"]
    rows_p6 = []
    for m in models_p6:
        rows_p6.append({
            "Algorithm": m,
            "NMC UDDS RMSE (%)": round(res_nmc[m]["rmse_pct"], 3),
            "NMC UDDS SD": round(res_nmc[m].get("rmse_sd", 0.0), 3),
            "LFP UDDS RMSE (%)": round(res_lfp[m]["rmse_pct"], 3),
            "LFP UDDS SD": round(res_lfp[m].get("rmse_sd", 0.0), 3),
            "LFP UDDS MaxE (%)": round(res_lfp[m]["max_err_pct"], 3),
            "LFP FUDS RMSE (%)": round(res_lfp_fuds[m]["rmse_pct"], 3),
        })
    rows_p6.append({
        "Algorithm": "EKF (No Hyst. Control)",
        "NMC UDDS RMSE (%)": np.nan,
        "NMC UDDS SD": 0.0,
        "LFP UDDS RMSE (%)": round(m_ekf_nh["rmse_pct"], 3),
        "LFP UDDS SD": 0.0,
        "LFP UDDS MaxE (%)": round(m_ekf_nh["max_err_pct"], 3),
        "LFP FUDS RMSE (%)": np.nan,
    })
    df_p6 = pd.DataFrame(rows_p6)
    df_p6.to_csv(os.path.join(results_dir, "table6_cross_chemistry.csv"), index=False)
    print(df_p6.to_string(index=False))

    # Plot Fig 6 using res_nmc, res_lfp, m_ekf_nh
    time_min = lfp_100["time"] / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.9), gridspec_kw={"height_ratios": [1.08, 1.05], "hspace": 0.42})
    ax1.axhspan(-2.0, 2.0, color="#16A34A", alpha=0.075, label=r"$\pm 2\%$ Corridor", zorder=0)
    ax1.axhline(0.0, color="#111827", linestyle="--", linewidth=0.75, zorder=1)
    ax1.plot(time_min, (t_ekf_nh - lfp_100["soc_true"]) * 100.0, label=f"EKF (No Hyst: {m_ekf_nh['rmse_pct']:.2f}%)", color="#9CA3AF", linestyle="-.", linewidth=1.25, zorder=2)
    for m in ["BiLSTM-Attn", "PINN-SOC", "SMO", "AEKF", "UKF", "EKF", "DEKF", "PIDKF"]:
        err = (res_lfp[m]["soc_est"] - lfp_100["soc_true"]) * 100.0
        ax1.plot(time_min, err, label=f"{m} ({res_lfp[m]['rmse_pct']:.2f}%)", color=PALETTE.get(m, "#64748B"), linewidth=1.8 if m == "PIDKF" else (1.05 if m in ("BiLSTM-Attn", "PINN-SOC") else 1.35), alpha=0.72 if m in ("BiLSTM-Attn", "PINN-SOC") else 0.94, zorder=7 if m == "PIDKF" else 4)
    ax1.set_ylabel("SOC Error (%)")
    ax1.set_xlabel("Time (minutes)")
    ax1.set_title(r"(a) Flat-Plateau LFP Tracking Error on Unseen UDDS ($25^\circ\mathrm{C}$, SOH $= 100\%$, 4-State Hysteresis)", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_xlim(0.0, 60.5)
    ax1.set_ylim(-2.5, 9.2)
    style_legend(ax1.legend(loc="upper right", ncol=3, fontsize=6.4, columnspacing=0.65, handlelength=1.35))

    xm = np.arange(len(models_p6))
    w = 0.36
    nmc_vals = [res_nmc[m]["rmse_pct"] for m in models_p6]
    lfp_vals = [res_lfp[m]["rmse_pct"] for m in models_p6]
    b1 = ax2.bar(xm - w / 2, nmc_vals, w, label="NMC (Sloping OCV)", color="#2563EB", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    b2 = ax2.bar(xm + w / 2, lfp_vals, w, label="LFP (Flat Plateau + Hysteresis)", color="#0F766E", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    max_b = max(max(nmc_vals), max(lfp_vals))
    for bar, val in list(zip(b1, nmc_vals)) + list(zip(b2, lfp_vals)):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val + max_b * 0.02, f"{val:.2f}%", ha="center", va="bottom", rotation=90, fontsize=6.8, fontweight="bold")
    ax2.set_xticks(xm)
    ax2.set_xticklabels(models_p6, rotation=15, ha="right", fontweight="bold", fontsize=8.0)
    ax2.set_ylabel("UDDS RMSE (%)")
    ax2.set_ylim(0.0, max_b * 1.32 + 0.4)
    ax2.set_title("(b) Direct Cross-Chemistry Error Comparison on Identical UDDS Schedule", loc="left", fontweight="bold")
    ax2.grid(True, axis="y")
    style_legend(ax2.legend(loc="upper left", ncol=2, fontsize=7.6))
    save_pub_figure(fig, os.path.join(results_dir, "fig8.png"))

    run_phase7_sensor_bias(engine, pidkf_nmc, results_dir)
    run_phase10_ablation_and_jacobian(engine, seeds, results_dir)

    print("\n" + "=" * 84)
    print("PHASES 6, 7 & 10 COMPLETED CLEANLY!")
    print("=" * 84)


if __name__ == "__main__":
    main()
