"""
Unified Q1 Publication Typography, Palette & Export Utilities
Ensures consistent Computer Modern typography, colorblind-safe method colors,
600-DPI PNG + lossless vector PDF export, and automated manuscript sync across
all 11 quantitative figures (fig1 through fig11).
"""

import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

PALETTE = {
    "Reference": "#111827",
    "True": "#111827",
    "Coulomb Counting": "#475569",
    "EKF": "#DC2626",               # Crimson Red
    "Standard EKF": "#DC2626",
    "AEKF": "#2563EB",              # Royal Cobalt Blue
    "DEKF": "#16A34A",              # Emerald Green
    "Dual EKF (DEKF)": "#16A34A",
    "SMO": "#7C3AED",               # Deep Violet
    "BiLSTM-Attn": "#EA580C",       # Burnished Orange
    "PINN-SOC": "#854D0E",          # Rich Ochre-Bronze (distinct across all baselines)
    "PIDKF (Proposed)": "#0F766E",  # Hero Deep Teal (matches Fig. 2 framework accent)
    "Novel PIDKF (Hybrid)": "#0F766E",
    "PIDKF": "#0F766E",
}

MARKERS = {
    "EKF": "s",
    "AEKF": "^",
    "DEKF": "o",
    "SMO": "D",
    "BiLSTM-Attn": "v",
    "PINN-SOC": "P",
    "PIDKF (Proposed)": "*",
}


def apply_pub_style():
    """Configure Matplotlib to mirror LaTeX Computer Modern + Nature/Elsevier house style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.labelsize": 10.2,
        "axes.titlesize": 10.6,
        "axes.titlepad": 7.0,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "lines.linewidth": 1.55,
        "grid.alpha": 0.72,
        "grid.linestyle": "--",
        "grid.linewidth": 0.55,
        "grid.color": "#E2E8F0",
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#1F2937",
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.color": "#1F2937",
        "ytick.color": "#1F2937",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_pub_figure(fig, png_path, dpi=600):
    """Save 600-DPI PNG + lossless vector PDF and sync both into manuscript folders."""
    abs_png = os.path.abspath(png_path)
    results_dir = os.path.dirname(abs_png)
    os.makedirs(results_dir, exist_ok=True)
    stem, _ = os.path.splitext(abs_png)
    abs_pdf = f"{stem}.pdf"

    fig.savefig(abs_png, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(abs_pdf, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

    root_dir = os.path.dirname(results_dir)
    base_png = os.path.basename(abs_png)
    base_pdf = os.path.basename(abs_pdf)
    for sub in ("manuscript", "submission_journal_of_energy_storage"):
        target_dir = os.path.join(root_dir, sub)
        if os.path.isdir(target_dir):
            shutil.copy2(abs_png, os.path.join(target_dir, base_png))
            shutil.copy2(abs_pdf, os.path.join(target_dir, base_pdf))


def style_legend(leg):
    """Apply clean Q1 styling to legend boxes."""
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("#FFFFFF")
        frame.set_edgecolor("#CBD5E1")
        frame.set_linewidth(0.75)
        frame.set_alpha(0.96)
    return leg


def add_zoom_inset(ax, bounds, xlim, ylim, loc1=2, loc2=4, fc="#F8FAFC"):
    """Create a framed zoom inset axis with crisp dashed connector guides."""
    axins = ax.inset_axes(bounds)
    axins.set_facecolor(fc)
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.tick_params(labelsize=7.2, length=2.4, width=0.65, pad=1.5)
    for sp in axins.spines.values():
        sp.set_edgecolor("#334155")
        sp.set_linewidth(0.75)
    axins.grid(True, linestyle=":", linewidth=0.45, color="#CBD5E1", alpha=0.8)
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="#475569", lw=0.7, ls="--", alpha=0.85)
    return axins
