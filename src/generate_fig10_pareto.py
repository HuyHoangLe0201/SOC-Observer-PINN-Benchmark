"""
Fig 10: Pareto Efficiency Frontier (Estimation Accuracy vs. ARM Cortex-M4 Latency)
Unified Q1 publication palette & collision-free direct labels + 600-DPI PNG & vector PDF export.
"""

import os
import matplotlib.pyplot as plt
try:
    from pub_style import PALETTE, MARKERS, apply_pub_style, save_pub_figure, style_legend
except ImportError:
    from src.pub_style import PALETTE, MARKERS, apply_pub_style, save_pub_figure, style_legend

apply_pub_style()

algs = ["SMO", "EKF", "AEKF", "DEKF", "PIDKF (Proposed)", "PINN-SOC", "BiLSTM-Attn"]
latencies = [1.67, 3.87, 5.24, 8.63, 14.20, 127.98, 2440.48]
aged_rmse = [4.254, 1.985, 3.834, 0.452, 0.414, 4.177, 4.202]
fresh_rmse = [0.941, 0.065, 0.759, 0.191, 0.185, 2.781, 2.813]

fig, ax = plt.subplots(figsize=(7.4, 4.6))

# Highlight Real-Time BMS Window (<20 us) cleanly
ax.axvspan(0.8, 20.0, alpha=0.075, color="#0F766E", zorder=0)
ax.axvline(20.0, color="#0F766E", linestyle="--", linewidth=1.1, alpha=0.85, zorder=2)
ax.text(
    4.0,
    4.86,
    r"Real-Time BMS Window ($<20\;\mu\mathrm{s}$)",
    ha="center",
    va="top",
    fontsize=7.9,
    fontweight="bold",
    color="#0F766E",
    bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#99F6E4", alpha=0.95, lw=0.75),
    zorder=6,
)

# Degradation arrows from fresh (100%) to aged (85%)
for i in range(len(algs)):
    ax.annotate(
        "",
        xy=(latencies[i], aged_rmse[i]),
        xytext=(latencies[i], fresh_rmse[i]),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.0, ls=":"),
        zorder=3,
    )

# Fresh Pareto Frontier (100% SOH)
ax.plot(
    [1.67, 3.87],
    [0.941, 0.065],
    color="#64748B",
    linestyle="--",
    linewidth=1.25,
    alpha=0.85,
    label="Fresh Pareto Frontier (100% SOH)",
    zorder=3,
)

# Aged Pareto Frontier (85% SOH)
pareto_lat = [1.67, 3.87, 8.63, 14.20]
pareto_rmse = [4.254, 1.985, 0.452, 0.414]
ax.plot(
    pareto_lat,
    pareto_rmse,
    color=PALETTE["PIDKF (Proposed)"],
    linestyle="-",
    linewidth=2.1,
    alpha=0.92,
    label="Aged Pareto Frontier (85% SOH)",
    zorder=5,
)

# Fresh hollow markers
ax.scatter(
    latencies,
    fresh_rmse,
    s=62,
    facecolors="white",
    edgecolors="#64748B",
    linewidth=1.2,
    label="Fresh Baseline (100% SOH)",
    zorder=4,
)

# Aged solid markers using unified project-wide colors
for i, a in enumerate(algs):
    m = "*" if "PIDKF" in a else MARKERS.get(a, "o")
    s = 250 if "PIDKF" in a else 105
    ax.scatter(
        latencies[i],
        aged_rmse[i],
        s=s,
        marker=m,
        color=PALETTE[a],
        edgecolors="#111827",
        linewidth=1.05,
        zorder=7 if "PIDKF" in a else 6,
    )

# Strictly de-conflicted white-pill annotations
annot_cfg = {
    "SMO": (1.67, 4.254, (1.67, 4.58), "center"),
    "EKF": (3.87, 1.985, (3.25, 1.85), "right"),
    "AEKF": (5.24, 3.834, (6.30, 3.83), "left"),
    "DEKF": (8.63, 0.452, (7.45, 0.62), "right"),
    "PIDKF (Proposed)": (14.20, 0.414, (22.5, 0.44), "left"),
    "PINN-SOC": (127.98, 4.177, (127.98, 3.82), "center"),
    "BiLSTM-Attn": (2440.48, 4.202, (2440.48, 3.82), "center"),
}

for a, (xv, yv, (xt, yt), ha) in annot_cfg.items():
    col = PALETTE[a] if "PIDKF" in a else "#111827"
    txt = r"$\mathbf{PIDKF\;(Proposed)}$" + f"\n(0.414%, 14.20 $\\mu$s)" if "PIDKF" in a else f"{a} ({yv:.2f}%)"
    kw = dict(arrowprops=dict(arrowstyle="->", color=col, lw=0.85)) if ("PIDKF" in a or a == "DEKF") else {}
    ax.annotate(
        txt,
        xy=(xv, yv),
        xytext=(xt, yt),
        ha=ha,
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=col,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#CBD5E1" if "PIDKF" not in a else col, alpha=0.94, lw=0.65),
        zorder=8,
        **kw,
    )

ax.set_xscale("log")
ax.set_xlabel(r"ARM Cortex-M4 Execution Latency ($\mu\mathrm{s/step}$) [Log Scale]")
ax.set_ylabel("SOC Estimation RMSE (%) [Lower is Better]")
ax.set_title("Pareto Frontier: Estimation Accuracy vs. Cortex-M4 Latency on Aged Cells (85% SOH)", fontweight="bold")
ax.grid(True, which="both")
ax.set_xlim(0.85, 6500.0)
ax.set_ylim(-0.25, 5.15)
style_legend(ax.legend(loc="upper right", fontsize=8.0))

out_path = os.path.join("results", "fig10_embedded_resource_tradeoff.png")
save_pub_figure(fig, out_path)
print("Saved Fig 10 (PNG 600 DPI + PDF) to results/fig10_embedded_resource_tradeoff.png")
