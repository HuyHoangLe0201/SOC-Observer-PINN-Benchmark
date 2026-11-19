"""
Pareto Efficiency & Embedded Resource Frontier (Fig. 10)
Journal of Energy Storage (Elsevier)

Strictly data-driven from executed CSV outputs:
  - results/table8_embedded_profiling.csv (Analytical ARMv7E-M cycles, latency, RAM, Flash)
  - results/table2_aging_degradation.csv (80% SOH degraded RMSE on unseen UDDS)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pub_style import PALETTE, MARKERS, apply_pub_style, save_pub_figure, style_legend

apply_pub_style()


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(root_dir, "results")

    df_t8 = pd.read_csv(os.path.join(results_dir, "table8_embedded_profiling.csv"))
    df_t2 = pd.read_csv(os.path.join(results_dir, "table2_aging_degradation.csv"))

    # Harmonize names between Table 8 ("PIDKF (Proposed)") and Table 2 ("PIDKF")
    t2_map = df_t2.set_index("Algorithm")["SOH=80%"].to_dict()

    records = []
    for _, row in df_t8.iterrows():
        alg = str(row["Algorithm"])
        key_t2 = "PIDKF" if "PIDKF" in alg else alg
        if key_t2 not in t2_map:
            continue
        records.append({
            "Algorithm": alg,
            "Short": "PIDKF" if "PIDKF" in alg else alg,
            "Latency_us": float(row["M4 Latency @168MHz (us)"]),
            "RMSE_80": float(t2_map[key_t2]),
            "RAM_B": float(row["Static+Stack RAM (B)"]),
            "Flash_KB": float(row["Weight+Kernel Flash (B)"]) / 1024.0,
        })

    df = pd.DataFrame(records).sort_values("Latency_us").reset_index(drop=True)

    # Compute Pareto-efficient subset minimizing (Latency_us, RMSE_80)
    pareto_pts = []
    best_rmse = np.inf
    for _, r in df.iterrows():
        if r["RMSE_80"] < best_rmse:
            best_rmse = r["RMSE_80"]
            pareto_pts.append((r["Latency_us"], r["RMSE_80"]))
    px = [p[0] for p in pareto_pts]
    py = [p[1] for p in pareto_pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 4.3), gridspec_kw={"width_ratios": [1.22, 0.95], "wspace": 0.28})

    # Panel (a): Pareto Frontier (Log-Scale Analytical Cortex-M4F Latency vs 80% SOH RMSE)
    ax1.axhspan(0.0, 1.0, color="#16A34A", alpha=0.08, zorder=0, label="<1.0% Target Zone")
    ax1.axhline(1.0, color="#15803D", linestyle=":", lw=0.95, zorder=1)
    ax1.plot(px, py, color="#0F766E", linestyle="--", lw=1.6, zorder=3, label="Pareto Efficiency Frontier")

    # Non-overlapping label offsets tailored cleanly in log-x / linear-y coordinates
    offsets = {
        "SMO": (1.28, -0.45, "left"),
        "EKF": (1.42, 0.30, "left"),
        "AEKF": (1.42, 0.36, "left"),
        "DEKF": (1.42, -0.05, "left"),
        "UKF": (1.45, -0.45, "left"),
        "PIDKF": (1.35, -0.15, "left"),
        "PINN-SOC": (1.30, 0.35, "left"),
        "BiLSTM-Attn": (0.55, -0.60, "right"),
    }
    for _, r in df.iterrows():
        short = r["Short"]
        col = PALETTE.get(short, "#475569")
        mk = MARKERS.get(short, "o")
        sz = 120 if short == "PIDKF" else 75
        ax1.scatter(r["Latency_us"], r["RMSE_80"], s=sz, color=col, marker=mk, edgecolor="white", linewidth=0.9, zorder=6)
        xf, dy, ha_v = offsets.get(short, (1.1, 0.2, "left"))
        ax1.annotate(
            f"{short}\n({r['Latency_us']:.1f} us, {r['RMSE_80']:.2f}%)",
            xy=(r["Latency_us"], r["RMSE_80"]),
            xytext=(r["Latency_us"] * xf, r["RMSE_80"] + dy),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.6),
            fontsize=6.4, fontweight="bold", color=col, ha=ha_v, va="center",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor=col, alpha=0.94, lw=0.6),
            zorder=8,
        )

    ax1.set_xscale("log")
    ax1.set_xlim(0.75, 45000.0)
    ax1.set_ylim(-0.2, max(df["RMSE_80"]) + 1.35)
    ax1.set_xlabel(r"Analytical Cortex-M4F Latency @ $168\;\mathrm{MHz}$ ($\mu$s/step, log)")
    ax1.set_ylabel("Aged RMSE at 80% SOH on UDDS (%)")
    ax1.set_title("(a) Accuracy--Compute Pareto Frontier (80% SOH)", fontweight="bold", fontsize=9.2)
    ax1.grid(True, which="both", alpha=0.45)
    style_legend(ax1.legend(loc="upper right", fontsize=7.0))

    # Panel (b): Static+Stack RAM (KB) & Weight+Kernel Flash (KB)
    y_pos = np.arange(len(df))
    h = 0.36
    ram_kb = df["RAM_B"].values / 1024.0
    flash_kb = df["Flash_KB"].values
    ax2.barh(y_pos - h / 2, ram_kb, h, label="Static + Stack RAM (KB)", color="#2563EB", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    ax2.barh(y_pos + h / 2, flash_kb, h, label="Weight + Kernel Flash (KB)", color="#0F766E", alpha=0.88, edgecolor="#1F2937", lw=0.6)
    for yp, rk, fk in zip(y_pos, ram_kb, flash_kb):
        ax2.text(rk * 1.15, yp - h / 2, f"{rk*1024:.0f} B" if rk < 1.0 else f"{rk:.2f} KB", va="center", ha="left", fontsize=6.2, fontweight="bold", color="#1E3A8A")
        ax2.text(fk * 1.15, yp + h / 2, f"{fk:.2f} KB", va="center", ha="left", fontsize=6.2, fontweight="bold", color="#0F766E")

    ax2.set_xscale("log")
    ax2.set_xlim(0.05, 5000.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df["Short"], fontweight="bold", fontsize=7.6)
    ax2.set_ylim(len(df) - 0.35, -1.65)
    ax2.set_xlabel("MCU Memory Footprint (KB, log scale)")
    ax2.set_title("(b) Static C Memory Sizing (ARMv7E-M)", fontweight="bold", fontsize=9.2)
    ax2.grid(True, axis="x", which="both", alpha=0.45)
    style_legend(ax2.legend(loc="upper right", fontsize=6.9))

    out_png = os.path.join(results_dir, "fig10_embedded_resource_tradeoff.png")
    save_pub_figure(fig, out_png)
    print(f"[Fig 10] Saved data-driven Pareto plot to {out_png}")


if __name__ == "__main__":
    main()
