"""
Publication-Quality Visualization Generator (Unified Q1 Editorial Design System)
Journal of Energy Storage (Elsevier)

Generates 600-DPI PNG + lossless vector PDF figures synchronized across
results/, manuscript/, and submission_journal_of_energy_storage/:
1. Fig 1: Dynamic Drive Cycle (UDDS) Tracking & Error Residuals (with tight inset)
2. Fig 2: Degradation Robustness across Progressive Battery Aging (SOH: 100% -> 80%)
3. Fig 3: Thermal Robustness across Multi-Temperature Regimes (direct bar value tags)
4. Fig 4: Convergence Trajectories under Initial Uncertainty (dual zoomed insets)
5. Fig 5: Multi-Attribute Radar Trade-Off Benchmark
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pub_style import PALETTE, MARKERS, apply_pub_style, save_pub_figure, style_legend, add_zoom_inset

apply_pub_style()


def plot_tracking_and_errors(data, results, save_path="../results/fig1_udds_tracking_errors.png"):
    """Fig 1: True vs Estimated SOC and Error Residuals with high-precision tracking inset."""
    time_min = data["time"] / 60.0
    soc_true = data["soc_true"] * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.15], "hspace": 0.24}
    )

    # Order plotting back-to-front so wide-swinging DL traces never obscure tight Kalman filters
    draw_order = [n for n in ["BiLSTM-Attn", "PINN-SOC", "SMO", "AEKF", "UKF", "EKF", "DEKF", "PIDKF"] if n in results]

    # Panel (a): SOC Trajectories
    handles_a, labels_a = [], []
    h_ref, = ax1.plot(time_min, soc_true, label="Reference (True)", color=PALETTE["True"], linewidth=2.1, zorder=8)
    handles_a.append(h_ref)
    labels_a.append("Reference (True)")

    for name in draw_order:
        res = results[name]
        is_dl = name in ("BiLSTM-Attn", "PINN-SOC", "MLP (Unconstrained)")
        h, = ax1.plot(
            time_min,
            res["soc_est"] * 100.0,
            label=name,
            color=PALETTE.get(name, "#64748B"),
            linewidth=1.15 if is_dl else (1.75 if name == "PIDKF" else 1.45),
            alpha=0.70 if is_dl else 0.92,
            zorder=2 if is_dl else (7 if name == "PIDKF" else 5),
        )
        handles_a.append(h)
        labels_a.append(name)

    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title(r"(a) SOC Tracking under Dynamic UDDS Cycle ($25^\circ\mathrm{C}$, SOH $= 100\%$)", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_xlim(0.0, 60.5)
    ax1.set_ylim(float(np.min(soc_true)) - 4.0, 102.0)
    style_legend(ax1.legend(handles_a, labels_a, loc="lower left", ncol=4, fontsize=7.0, columnspacing=0.85, handlelength=1.55))

    # Panel (b): Error Residuals
    ax2.axhspan(-2.0, 2.0, color="#16A34A", alpha=0.075, label=r"$\pm 2\%$ Corridor", zorder=0)
    ax2.axhline(0.0, color="#111827", linestyle="--", linewidth=0.75, zorder=1)
    ax2.axhline(2.0, color="#15803D", linestyle=":", linewidth=0.95, zorder=1)
    ax2.axhline(-2.0, color="#15803D", linestyle=":", linewidth=0.95, zorder=1)

    max_err_val = 2.0
    min_err_val = -2.0
    for name in draw_order:
        res = results[name]
        err = (res["soc_est"] - data["soc_true"]) * 100.0
        max_err_val = max(max_err_val, float(np.max(err)))
        min_err_val = min(min_err_val, float(np.min(err)))
        is_dl = name in ("BiLSTM-Attn", "PINN-SOC", "MLP (Unconstrained)")
        ax2.plot(
            time_min,
            err,
            label=f"{name} ({res['rmse_pct']:.2f}%)",
            color=PALETTE.get(name, "#64748B"),
            linewidth=1.10 if is_dl else (1.70 if name == "PIDKF" else 1.40),
            alpha=0.68 if is_dl else 0.92,
            zorder=2 if is_dl else (7 if name == "PIDKF" else 5),
        )

    axins = add_zoom_inset(ax2, [0.055, 0.56, 0.32, 0.37], xlim=(14.0, 22.0), ylim=(-1.6, 1.8), loc1=3, loc2=4)
    axins.axhspan(-1.6, 1.8, color="#16A34A", alpha=0.06, zorder=0)
    axins.axhline(0.0, color="#111827", linestyle="--", linewidth=0.6)
    for name in ["SMO", "AEKF", "UKF", "DEKF", "EKF", "PIDKF"]:
        if name in results:
            err = (results[name]["soc_est"] - data["soc_true"]) * 100.0
            axins.plot(time_min, err, color=PALETTE[name], linewidth=1.3 if name == "PIDKF" else (0.75 if name == "SMO" else 1.05), alpha=0.55 if name == "SMO" else 0.92)
    axins.set_xticks([15, 18, 21])
    axins.set_yticks([-1, 0, 1])
    axins.text(
        0.5, 0.87, r"Sub-2% Regime (14–22 min)",
        transform=axins.transAxes, ha="center", va="center",
        fontsize=6.2, fontweight="bold", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor="#CBD5E1", alpha=0.95, lw=0.5),
    )

    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("SOC Error (%)")
    ax2.set_title(r"(b) Estimation Error Residuals ($e_k = \hat{s}_k - s_k$) & RMSE Summary", loc="left", fontweight="bold")
    ax2.grid(True)
    ax2.set_xlim(0.0, 60.5)
    ax2.set_ylim(min(min_err_val - 1.2, -3.5), max(max_err_val * 1.38 + 2.5, 8.0))
    style_legend(ax2.legend(loc="upper right", ncol=3, fontsize=6.6, columnspacing=0.70, handlelength=1.30))

    save_pub_figure(fig, save_path)
    print(f"[Visualizer] Saved Fig 1 (PNG 600 DPI + PDF) to {save_path}")


def plot_aging_degradation_trend(soh_levels, rmse_by_model, save_path="../results/fig2_aging_degradation.png"):
    """Fig 2: RMSE evolution from SOH = 100% to 80% with dynamic non-overlapping right-margin callouts."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    ax.axhspan(0.0, 1.0, color="#16A34A", alpha=0.075, label="Target Zone (<1.0% RMSE)", zorder=0)
    ax.axhline(1.0, color="#15803D", linestyle=":", linewidth=0.9, zorder=1)

    soh_pct = soh_levels * 100.0
    plot_models = [m for m in rmse_by_model if m != "MLP (Unconstrained)"]
    for name in plot_models:
        rmse_vals = rmse_by_model[name]
        col = PALETTE.get(name, "#64748B")
        ax.plot(
            soh_pct,
            rmse_vals,
            label=name,
            color=col,
            marker=MARKERS.get(name, "o"),
            markersize=6.2 if name != "PIDKF" else 7.2,
            markeredgecolor="white",
            markeredgewidth=0.75,
            linewidth=2.2 if name == "PIDKF" else 1.85,
            zorder=6 if name == "PIDKF" else 4,
        )

    # Sort endpoints vertically at SOH = 80% and space badges cleanly to eliminate any collision
    endpoints = sorted([(name, float(rmse_by_model[name][-1])) for name in plot_models], key=lambda x: x[1])
    max_y = max(max(rmse_by_model[m]) for m in plot_models)
    min_gap = max(0.32, (max_y + 0.5) / (len(endpoints) + 2.5))
    y_placed = []
    for name, y_val in endpoints:
        y_lbl = y_val
        if y_placed and y_lbl - y_placed[-1] < min_gap:
            y_lbl = y_placed[-1] + min_gap
        y_placed.append(y_lbl)
        col = PALETTE.get(name, "#64748B")
        if abs(y_val - y_lbl) > 0.04:
            ax.plot([80.0, 79.15], [y_val, y_lbl], color=col, linewidth=0.75, linestyle="-", alpha=0.8)
        ax.text(
            79.0, y_lbl, f"{name}: {y_val:.2f}%",
            va="center", ha="left", fontsize=7.2, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
        )

    if "EKF" in rmse_by_model:
        ekf0, ekf80 = float(rmse_by_model["EKF"][0]), float(rmse_by_model["EKF"][-1])
        mult = ekf80 / max(ekf0, 1e-6)
        ax.annotate(
            f"EKF {mult:.1f}x Surge\n(Frozen Aging Prior)",
            xy=(80.0, ekf80), xytext=(86.5, 2.75),
            arrowprops=dict(arrowstyle="->", color=PALETTE["EKF"], lw=1.0),
            fontsize=7.2, color=PALETTE["EKF"], fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=PALETTE["EKF"], alpha=0.95, lw=0.7),
            zorder=12,
        )
    if "PIDKF" in rmse_by_model:
        p85 = float(rmse_by_model["PIDKF"][3])
        ax.annotate(
            "PIDKF + DEKF Track\n" + r"$\hat{R}_0, \hat{Q}_{\mathrm{act}}$ & Overpotential",
            xy=(85.0, p85), xytext=(89.5, 0.28),
            arrowprops=dict(arrowstyle="->", color=PALETTE["PIDKF"], lw=1.0),
            fontsize=7.2, color=PALETTE["PIDKF"], fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=PALETTE["PIDKF"], alpha=0.95, lw=0.7),
            zorder=12,
        )

    ax.set_xlim(101.2, 72.5)
    ax.set_xticks([100, 95, 90, 85, 80])
    ax.set_xlabel(r"State of Health --- SOH (%) [Fresh $\rightarrow$ End-of-Life]")
    ax.set_ylabel("SOC Estimation RMSE (%)")
    ax.set_title("Degradation Robustness: SOC Error across Cell Aging Stages", fontweight="bold")
    ax.grid(True)
    ax.set_ylim(-0.30, max(max_y, y_placed[-1]) + 1.45)
    style_legend(ax.legend(loc="upper left", ncol=3, framealpha=0.96, fontsize=7.0, columnspacing=0.9))

    save_pub_figure(fig, save_path)
    print(f"[Visualizer] Saved Fig 2 (PNG 600 DPI + PDF) to {save_path}")


def plot_thermal_regimes_bar(temp_labels, rmse_data, save_path="../results/fig3_thermal_robustness.png"):
    """Fig 3: Clean grouped bar chart across 0, 25, 45 C with exact numeric labels above every bar."""
    models = [m for m in rmse_data.keys() if m != "MLP (Unconstrained)"]
    n_models = len(models)
    n_temps = len(temp_labels)

    x = np.arange(n_temps)
    width = min(0.132, 0.78 / max(n_models, 1))

    fig, ax = plt.subplots(figsize=(7.6, 4.7))

    max_v = 2.5
    for idx, model_name in enumerate(models):
        offset = (idx - n_models / 2.0 + 0.5) * width
        vals = [rmse_data[model_name][t] for t in temp_labels]
        max_v = max(max_v, max(vals))
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model_name,
            color=PALETTE.get(model_name, "#64748B"),
            edgecolor="#1F2937",
            linewidth=0.65,
            alpha=0.90,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + max(0.12, max_v * 0.015),
                f"{v:.2f}%",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.8,
                fontweight="bold",
                color="#111827",
                zorder=5,
            )

    ax.axhline(2.0, color="#DC2626", linestyle="--", linewidth=1.05, alpha=0.78, label="2.0% BMS Bound", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$0^\circ\mathrm{C}$ (Cold)", r"$25^\circ\mathrm{C}$ (Nominal)", r"$45^\circ\mathrm{C}$ (Hot)"], fontweight="bold")
    ax.set_xlabel("Operating Ambient Temperature")
    ax.set_ylabel("SOC Estimation RMSE (%)")
    ax.set_title("Thermal Sensitivity Benchmark across Temperature Regimes", fontweight="bold")
    ax.grid(True, axis="y", zorder=0)
    ax.set_ylim(0.0, max_v * 1.28 + 1.2)
    style_legend(ax.legend(loc="upper center", ncol=4, fontsize=7.4, columnspacing=1.0, handlelength=1.45))

    save_pub_figure(fig, save_path)
    print(f"[Visualizer] Saved Fig 3 (PNG 600 DPI + PDF) to {save_path}")


def plot_convergence_trajectories(data, results_converge, save_path="../results/fig4_initial_error_convergence.png"):
    """Fig 4: Initial uncertainty recovery (-40% offset) with dynamic data-driven zoom insets."""
    horizon = min(1300, len(data["time"]))
    time_sec = data["time"][:horizon]
    soc_true = data["soc_true"][:horizon] * 100.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 5.6), sharex=True, gridspec_kw={"hspace": 0.24}
    )

    draw_order = [n for n in ["BiLSTM-Attn", "PINN-SOC", "SMO", "AEKF", "UKF", "EKF", "DEKF", "PIDKF"] if n in results_converge]

    ax1.axvspan(0, 20, color="#16A34A", alpha=0.08, label="Fast Window (<20 s)")
    ax1.plot(time_sec, soc_true, label="True SOC", color=PALETTE["True"], linewidth=2.1, zorder=8)
    for name in draw_order:
        res = results_converge[name]
        ax1.plot(
            time_sec,
            res["soc_est"][:horizon] * 100.0,
            label=name,
            color=PALETTE.get(name, "#64748B"),
            linewidth=1.25 if name in ("BiLSTM-Attn", "PINN-SOC") else (1.75 if name == "PIDKF" else 1.45),
            alpha=0.88,
            zorder=7 if name == "PIDKF" else (3 if name in ("BiLSTM-Attn", "PINN-SOC") else 5),
        )

    ax1_ins = add_zoom_inset(ax1, [0.12, 0.10, 0.44, 0.54], xlim=(0.0, 50.0), ylim=(52.0, 100.0), loc1=2, loc2=3)
    ax1_ins.axvspan(0, 20, color="#16A34A", alpha=0.08)
    ax1_ins.plot(time_sec, soc_true, color=PALETTE["True"], linewidth=1.8, zorder=8)
    for name in draw_order:
        ax1_ins.plot(time_sec, results_converge[name]["soc_est"][:horizon] * 100.0, color=PALETTE.get(name, "#64748B"), linewidth=1.25)

    fast_tags = []
    for m_chk in ["AEKF", "UKF", "EKF", "SMO", "PIDKF", "DEKF"]:
        if m_chk in results_converge:
            tr = results_converge[m_chk].get("t_reach_2", np.nan)
            if not np.isnan(tr) and tr <= 60:
                fast_tags.append(f"{m_chk}: {int(round(tr))} s")
    box_str = "Transient Recovery (0–50 s)\n" + (" | ".join(fast_tags[:3]) if fast_tags else "Closed-Loop Contraction")
    if len(fast_tags) > 3:
        box_str += "\n" + " | ".join(fast_tags[3:])
    ax1_ins.text(
        0.96, 0.12, box_str,
        transform=ax1_ins.transAxes, ha="right", va="bottom",
        fontsize=6.0, fontweight="bold", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#CBD5E1", alpha=0.95, lw=0.5),
    )

    ax1.set_ylabel("State of Charge (%)")
    ax1.set_title(r"(a) Trajectory Recovery from Severe Initial Offset ($\Delta\mathrm{SOC}_0 = -40\%$)", loc="left", fontweight="bold")
    ax1.grid(True)
    ax1.set_xlim(0.0, horizon)
    ax1.set_ylim(48.0, 102.5)
    style_legend(ax1.legend(loc="lower right", ncol=2, fontsize=6.9, columnspacing=0.8))

    # Panel (b)
    ax2.axhspan(0.0, 2.0, color="#16A34A", alpha=0.085, label=r"$\pm 2\%$ Target Band", zorder=0)
    ax2.axhline(2.0, color="#DC2626", linestyle="--", linewidth=1.05, label=r"$\pm 2\%$ Tolerance Bound", zorder=2)
    for name in draw_order:
        err = np.abs((results_converge[name]["soc_est"][:horizon] - data["soc_true"][:horizon]) * 100.0)
        ax2.plot(
            time_sec,
            err,
            label=name,
            color=PALETTE.get(name, "#64748B"),
            linewidth=1.25 if name in ("BiLSTM-Attn", "PINN-SOC") else (1.75 if name == "PIDKF" else 1.45),
            alpha=0.88,
            zorder=7 if name in ("PIDKF", "DEKF") else 4,
        )

    # Dynamically annotate PIDKF / DEKF reach & settle times
    hero = "PIDKF" if "PIDKF" in results_converge else "DEKF"
    tr_hero = results_converge[hero].get("t_reach_2", np.nan)
    ts_hero = results_converge[hero].get("t_settle_5", np.nan)
    if not np.isnan(tr_hero) and tr_hero < horizon:
        ax2.axvline(tr_hero, color=PALETTE[hero], linestyle=":", linewidth=1.2, zorder=6)
        txt_ts = f"{int(round(ts_hero))} s" if not np.isnan(ts_hero) else "N/A"
        ax2.annotate(
            f"{hero} Recovery\nReach 2%: {int(round(tr_hero))} s | Settle 5%: {txt_ts}",
            xy=(tr_hero, 2.0),
            xytext=(920.0, 22.0),
            arrowprops=dict(arrowstyle="->", color=PALETTE[hero], lw=1.05),
            fontsize=7.3,
            color=PALETTE[hero],
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=PALETTE[hero], alpha=0.95, lw=0.7),
            zorder=12,
        )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Absolute Error (%)")
    ax2.set_title(f"(b) Convergence Dynamics into $\\pm 2\\%$ Error Boundary (0–{horizon} s Window)", loc="left", fontweight="bold")
    ax2.grid(True)
    ax2.set_ylim(-0.5, 49.5)
    style_legend(ax2.legend(loc="upper right", ncol=3, fontsize=6.9, columnspacing=0.80))

    save_pub_figure(fig, save_path)
    print(f"[Visualizer] Saved Fig 4 (PNG 600 DPI + PDF) to {save_path}")


def plot_radar_summary(metrics_dict, save_path="../results/fig5_radar_tradeoff.png"):
    """Fig 5: Clean non-muddy radar trade-off profile with haloed radial ticks and alternating rings."""
    categories = [
        "Nominal\nAccuracy",
        "Aging\nRobustness",
        "Thermal\nRobustness",
        "Convergence\nSpeed",
        "Computational\nEfficiency",
    ]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.4, 5.4), subplot_kw=dict(polar=True))
    ax.set_theta_offset(0.0)
    ax.set_rlabel_position(36)

    # Subtle alternating radial rings for crisp visual depth
    theta_grid = np.linspace(0, 2 * np.pi, 180)
    for r_in, r_out, tint in [(0, 2, "#F8FAFC"), (4, 6, "#F8FAFC"), (8, 10, "#F8FAFC")]:
        ax.fill_between(theta_grid, r_in, r_out, color=tint, zorder=0)

    for name, scores in metrics_dict.items():
        values = scores + scores[:1]
        col = PALETTE.get(name, "#64748B")
        ax.plot(
            angles,
            values,
            linewidth=2.0,
            linestyle="-",
            label=name,
            color=col,
            marker=MARKERS.get(name, "o"),
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
        ax.fill(angles, values, color=col, alpha=0.04, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Drawn radially outward with explicit alignment so zero label grazes the outer ring
    out_cfg = [
        (0.0, categories[0], "left", "center"),
        (angles[1], categories[1], "center", "bottom"),
        (angles[2], categories[2], "right", "center"),
        (angles[3], categories[3], "right", "center"),
        (angles[4], categories[4], "center", "top"),
    ]
    for ang, lbl_txt, ha_v, va_v in out_cfg:
        ax.text(ang, 11.45, lbl_txt, ha=ha_v, va=va_v, fontsize=9.1, fontweight="bold", color="#111827")

    ax.set_ylim(0, 10.6)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels([])
    for rv in [2, 4, 6, 8, 10]:
        ax.text(
            np.deg2rad(54.0),
            rv,
            str(rv),
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.13", facecolor="white", edgecolor="#CBD5E1", alpha=0.95, lw=0.45),
            zorder=15,
        )

    ax.grid(True, color="#94A3B8", linestyle="--", linewidth=0.65, alpha=0.65)
    ax.spines["polar"].set_color("#64748B")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title("Multi-Attribute Performance Trade-Off Benchmark (Score / 10)", y=1.18, fontweight="bold")
    style_legend(ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8.4, columnspacing=1.3))

    save_pub_figure(fig, save_path)
    print(f"[Visualizer] Saved Fig 5 (PNG 600 DPI + PDF) to {save_path}")
