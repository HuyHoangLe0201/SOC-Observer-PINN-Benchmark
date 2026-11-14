"""
Generate two publication-grade conceptual figures for Journal of Energy Storage:
1. fig0_electrochemical_mechanism.png: Coupled Electro-Thermal-Aging Physical Mechanism & 2nd-Order ECM with Hysteresis and OCV-SOC Observability Analysis
2. fig0_pidkf_framework.png: Proposed Physics-Informed Dual Kalman Filter (PIDKF) Hybrid Observer & 10-Phase Benchmarking Framework
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#1e293b",
    "axes.linewidth": 1.1,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def draw_box(ax, x, y, w, h, title="", subtitle="", fc="#f8fafc", ec="#334155",
             lw=1.5, radius=0.02, title_color="#0f172a", sub_color="#334155",
             title_fs=10.5, sub_fs=8.8, alpha=1.0, zorder=3, header_bg=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)

    if header_bg is not None:
        hh = min(h * 0.34, 0.055)
        hbox = FancyBboxPatch(
            (x, y + h - hh), w, hh,
            boxstyle=f"round,pad=0.002,rounding_size={radius*0.8}",
            facecolor=header_bg, edgecolor=ec, linewidth=lw * 0.9, zorder=zorder + 0.5
        )
        ax.add_patch(hbox)
        if title:
            ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=title_color, zorder=zorder + 1)
        if subtitle:
            ax.text(x + w / 2, y + (h - hh) / 2, subtitle, ha="center", va="center",
                    fontsize=sub_fs, color=sub_color, zorder=zorder + 1, linespacing=1.28)
    else:
        if title and subtitle:
            ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=title_color, zorder=zorder + 1)
            ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                    fontsize=sub_fs, color=sub_color, zorder=zorder + 1, linespacing=1.25)
        elif title:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=title_fs, fontweight="bold", color=title_color, zorder=zorder + 1)
        elif subtitle:
            ax.text(x + w / 2, y + h / 2, subtitle, ha="center", va="center",
                    fontsize=sub_fs, color=sub_color, zorder=zorder + 1, linespacing=1.25)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.6, style="-|>", rad=0.0, ls="-", zorder=5, mutation=14):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, linestyle=ls,
        mutation_scale=mutation, zorder=zorder
    )
    ax.add_patch(arr)
    return arr


def draw_resistor_h(ax, x_start, x_end, y, height=0.024, n_zigs=5, color="#0f172a", lw=1.8, zorder=5):
    xs = np.linspace(x_start, x_end, n_zigs * 2 + 1)
    ys = np.zeros_like(xs) + y
    for i in range(1, len(xs) - 1):
        ys[i] = y + (height if i % 2 == 1 else -height)
    ax.plot(xs, ys, color=color, lw=lw, zorder=zorder)


def draw_capacitor_h(ax, x_mid, y, gap=0.012, plate_h=0.036, color="#0f172a", lw=2.0, zorder=5):
    ax.plot([x_mid - gap / 2, x_mid - gap / 2], [y - plate_h, y + plate_h], color=color, lw=lw, zorder=zorder)
    ax.plot([x_mid + gap / 2, x_mid + gap / 2], [y - plate_h, y + plate_h], color=color, lw=lw, zorder=zorder)


def generate_mechanism_figure(save_paths):
    fig = plt.figure(figsize=(15.5, 9.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.12, 1.0],
                           hspace=0.25, wspace=0.18, left=0.03, right=0.98, top=0.93, bottom=0.06)

    fig.suptitle(
        "Coupled Electro-Thermal-Aging Physical Mechanism, 2nd-Order Hysteresis ECM, and Observability Landscape",
        fontsize=13.8, fontweight="bold", color="#0f172a", y=0.975
    )

    # =========================================================================
    # PANEL (a): Microscopic Degradation & Electro-Thermal Coupling (Top-Left)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")

    draw_box(ax_a, 0.005, 0.01, 0.99, 0.97, fc="#ffffff", ec="#cbd5e1", lw=1.2, radius=0.02, zorder=1)
    ax_a.text(0.025, 0.935, "(a) Microscopic Electrode Degradation & Coupled Electro-Thermal Physics",
              fontsize=11.2, fontweight="bold", color="#0f172a", zorder=6)

    # Anode zone (left)
    draw_box(ax_a, 0.03, 0.36, 0.27, 0.51, fc="#eff6ff", ec="#3b82f6", lw=1.4, radius=0.02,
             header_bg="#dbeafe", title="Graphite Anode (Negative)", title_color="#1e3a8a", title_fs=9.5)
    # Draw graphite particle + SEI layer
    outer_sei = Circle((0.165, 0.60), 0.115, facecolor="#fde68a", edgecolor="#d97706", lw=1.6, ls="--", zorder=4)
    inner_part = Circle((0.165, 0.60), 0.082, facecolor="#3b82f6", edgecolor="#1d4ed8", lw=1.5, zorder=5)
    ax_a.add_patch(outer_sei)
    ax_a.add_patch(inner_part)
    ax_a.text(0.165, 0.60, r"$\mathrm{Li_xC_6}$" + "\nSolid\nDiffusion\n" + r"$D_s(T)$",
              ha="center", va="center", fontsize=8.0, fontweight="bold", color="white", zorder=6)
    ax_a.text(0.165, 0.745, "SEI Film Growth", ha="center", va="center",
              fontsize=8.0, fontweight="bold", color="#b45309", zorder=6)
    ax_a.text(0.165, 0.42, r"$\bullet$ Traps cyclable $\mathrm{Li^+}$ (LLI)" + "\n" +
              r"$\bullet$ Film resistance $\uparrow \Delta R_{\mathrm{SEI}}$" + "\n" +
              r"$\bullet$ Cold plating risk ($T \to 0^\circ\mathrm{C}$)",
              ha="center", va="center", fontsize=7.8, color="#1e293b", zorder=6)

    # Separator & Electrolyte (center)
    draw_box(ax_a, 0.33, 0.36, 0.30, 0.51, fc="#f8fafc", ec="#64748b", lw=1.3, radius=0.02,
             header_bg="#e2e8f0", title="Electrolyte & Double-Layer Interface", title_color="#1e293b", title_fs=9.2)
    # Li+ ion migration arrows
    draw_arrow(ax_a, 0.35, 0.70, 0.60, 0.70, color="#0284c7", lw=2.0, mutation=13)
    ax_a.text(0.475, 0.735, r"Discharge: $\mathrm{Li^+}$ Migration ($I > 0$)",
              ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0369a1", zorder=6)
    draw_arrow(ax_a, 0.60, 0.62, 0.35, 0.62, color="#059669", lw=1.6, ls="--", mutation=12)
    ax_a.text(0.475, 0.585, r"Charge: $\mathrm{Li^+}$ Intercalation ($I < 0$)",
              ha="center", va="center", fontsize=7.8, color="#047857", zorder=6)

    # Butler-Volmer & Desolvation box inside center
    draw_box(ax_a, 0.35, 0.39, 0.26, 0.16, fc="#fffbeb", ec="#f59e0b", lw=1.1, radius=0.015,
             title="Butler–Volmer Kinetics",
             subtitle=r"$\eta_{\mathrm{act}} = \frac{2RT}{F}\mathrm{arcsinh}\!\left(\frac{I}{2Ai_0(T,\mathrm{SOH})}\right)$" + "\n" +
                      r"Desolvation barrier: $E_{a,1}^{\mathrm{cold}} = 28.4\;\mathrm{kJ/mol}$",
             title_fs=8.3, sub_fs=7.4, title_color="#92400e", sub_color="#78350f")

    # Cathode zone (right)
    draw_box(ax_a, 0.66, 0.36, 0.31, 0.51, fc="#fef2f2", ec="#ef4444", lw=1.4, radius=0.02,
             header_bg="#fee2e2", title="Cathode Host (NMC / LFP)", title_color="#991b1b", title_fs=9.5)
    # Two sub-cards: NMC vs LFP
    draw_box(ax_a, 0.68, 0.59, 0.27, 0.19, fc="#ffffff", ec="#f87171", lw=1.1, radius=0.015,
             title="NMC811 / INR18650-20R",
             subtitle=r"Solid-solution single-phase:" + "\n" +
                      r"Steep slope $V_{\mathrm{ocv}}'(z) \in [0.48, 4.67]\;\mathrm{V}$" + "\n" +
                      r"High observability ($\mathrm{rank}\,\mathcal{O}_k = 3$)",
             title_fs=8.3, sub_fs=7.4, title_color="#b91c1c")
    draw_box(ax_a, 0.68, 0.38, 0.27, 0.19, fc="#ffffff", ec="#a855f7", lw=1.1, radius=0.015,
             title=r"$\mathrm{LiFePO_4}$ / ANR26650M1-B",
             subtitle=r"Biphasic $\mathrm{LiFePO_4 \leftrightarrow FePO_4}$ ($F'=0$):" + "\n" +
                      r"Flat plateau $V_{\mathrm{ocv}}' \approx 0.08\;\mathrm{V}$ + Hysteresis $V_h$" + "\n" +
                      r"Severe observability collapse ($\kappa(\mathcal{O}_k) \uparrow 6\times$)",
             title_fs=8.3, sub_fs=7.4, title_color="#6b21a8")

    # Bottom bar in Panel (a): Coupled Electro-Thermal-Aging Loop
    draw_box(ax_a, 0.03, 0.04, 0.94, 0.28, fc="#f0fdf4", ec="#16a34a", lw=1.4, radius=0.02,
             header_bg="#dcfce7",
             title="Bidirectional Electro-Thermal-Aging Coupling Loop (Arrhenius & Lumped Energy Balance)",
             title_color="#14532d", title_fs=9.3)

    draw_box(ax_a, 0.05, 0.065, 0.28, 0.16, fc="#ffffff", ec="#22c55e", lw=1.1, radius=0.012,
             title="Heat Generation & Balance",
             subtitle=r"$C_{\mathrm{th}}\frac{dT}{dt} = I^2 R_0 + I(U_1+U_2)$" + "\n" +
                      r"$- IT\frac{\partial V_{\mathrm{ocv}}}{\partial T} - \frac{T-T_{\mathrm{amb}}}{R_{\mathrm{th}}}$",
             title_fs=8.3, sub_fs=7.6, title_color="#166534")

    draw_box(ax_a, 0.365, 0.065, 0.28, 0.16, fc="#ffffff", ec="#22c55e", lw=1.1, radius=0.012,
             title="Piecewise Arrhenius Scaling",
             subtitle=r"$R_i(T,\mathrm{SOH}) = R_{i,\mathrm{nom}}\,f_{T,i}(T)$" + "\n" +
                      r"$\times\,[1 + \kappa_{R_i}(1-\mathrm{SOH})]$",
             title_fs=8.3, sub_fs=7.6, title_color="#166534")

    draw_box(ax_a, 0.68, 0.065, 0.27, 0.16, fc="#ffffff", ec="#22c55e", lw=1.1, radius=0.012,
             title="Usable Capacity Modulation",
             subtitle=r"$Q_{\mathrm{act}}(T,\mathrm{SOH}) = \mathrm{SOH}\cdot Q_{\mathrm{init}}\,f_Q(T)$" + "\n" +
                      r"$0.89\times\;\mathrm{at}\;0^\circ\mathrm{C},\;\;1.03\times\;\mathrm{at}\;45^\circ\mathrm{C}$",
             title_fs=8.3, sub_fs=7.6, title_color="#166534")

    draw_arrow(ax_a, 0.33, 0.145, 0.365, 0.145, color="#15803d", lw=1.6, mutation=12)
    draw_arrow(ax_a, 0.645, 0.145, 0.68, 0.145, color="#15803d", lw=1.6, mutation=12)

    # =========================================================================
    # PANEL (b): 2nd-Order ECM with Hysteresis & Unmodeled Residual (Top-Right)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")

    draw_box(ax_b, 0.005, 0.01, 0.99, 0.97, fc="#ffffff", ec="#cbd5e1", lw=1.2, radius=0.02, zorder=1)
    ax_b.text(0.025, 0.935, "(b) Augmented 2nd-Order RC Equivalent Circuit & Neural Residual Formulation",
              fontsize=11.2, fontweight="bold", color="#0f172a", zorder=6)

    # Schematic diagram box
    draw_box(ax_b, 0.03, 0.38, 0.94, 0.50, fc="#f8fafc", ec="#94a3b8", lw=1.2, radius=0.02, zorder=2)

    # Circuit coordinates
    y_top = 0.68
    y_bot = 0.43
    x_src = 0.09

    # Left branch: Vocv + Vh
    ax_b.plot([x_src, x_src], [y_bot, 0.51], color="#0f172a", lw=1.8, zorder=4)
    ocv_circle = Circle((x_src, 0.555), 0.038, facecolor="#dbeafe", edgecolor="#1d4ed8", lw=1.8, zorder=5)
    ax_b.add_patch(ocv_circle)
    ax_b.text(x_src, 0.555, r"$V_{\mathrm{ocv}}(z)$", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#1e3a8a", zorder=6)
    ax_b.plot([x_src, x_src], [0.593, 0.615], color="#0f172a", lw=1.8, zorder=4)

    # Hysteresis block in series on left branch
    vh_box = FancyBboxPatch((x_src - 0.038, 0.615), 0.076, 0.048,
                            boxstyle="round,pad=0.002,rounding_size=0.008",
                            facecolor="#f3e8ff", edgecolor="#7e22ce", lw=1.5, zorder=5)
    ax_b.add_patch(vh_box)
    ax_b.text(x_src, 0.639, r"$\pm V_{h,k}$", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#6b21a8", zorder=6)
    ax_b.plot([x_src, x_src], [0.663, y_top], color="#0f172a", lw=1.8, zorder=4)

    # R0 resistor
    ax_b.plot([x_src, 0.16], [y_top, y_top], color="#0f172a", lw=1.8, zorder=4)
    draw_resistor_h(ax_b, 0.16, 0.26, y_top, height=0.022, n_zigs=4, color="#b91c1c", lw=1.8)
    ax_b.text(0.21, y_top + 0.048, r"$R_0(T,\mathrm{SOH})$" + "\nOhmic", ha="center", va="center",
              fontsize=8.0, fontweight="bold", color="#991b1b", zorder=6)
    ax_b.plot([0.26, 0.31], [y_top, y_top], color="#0f172a", lw=1.8, zorder=4)

    # RC1 loop (Charge Transfer: R1, C1, U1)
    x_rc1_l, x_rc1_r = 0.31, 0.49
    y_rc_up, y_rc_dn = y_top + 0.065, y_top - 0.065
    ax_b.plot([x_rc1_l, x_rc1_l], [y_rc_dn, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    ax_b.plot([x_rc1_r, x_rc1_r], [y_rc_dn, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    # top arm: R1
    ax_b.plot([x_rc1_l, x_rc1_l + 0.03], [y_rc_up, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    draw_resistor_h(ax_b, x_rc1_l + 0.03, x_rc1_r - 0.03, y_rc_up, height=0.018, n_zigs=4, color="#0369a1", lw=1.6)
    ax_b.plot([x_rc1_r - 0.03, x_rc1_r], [y_rc_up, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    ax_b.text((x_rc1_l + x_rc1_r) / 2, y_rc_up + 0.035, r"$R_1$ (Charge Transfer)", ha="center", va="center",
              fontsize=7.8, fontweight="bold", color="#0369a1", zorder=6)
    # bottom arm: C1
    x_c1 = (x_rc1_l + x_rc1_r) / 2
    ax_b.plot([x_rc1_l, x_c1 - 0.010], [y_rc_dn, y_rc_dn], color="#0f172a", lw=1.6, zorder=4)
    draw_capacitor_h(ax_b, x_c1, y_rc_dn, gap=0.018, plate_h=0.024, color="#0369a1", lw=1.8)
    ax_b.plot([x_c1 + 0.010, x_rc1_r], [y_rc_dn, y_rc_dn], color="#0f172a", lw=1.6, zorder=4)
    ax_b.text(x_c1, y_rc_dn - 0.062, r"$C_1$ ($\tau_1 = R_1 C_1$)" + "\n" + r"$+ \;\; U_{1,k} \;\; -$",
              ha="center", va="center", fontsize=7.6, color="#0369a1", zorder=6)

    ax_b.plot([x_rc1_r, 0.54], [y_top, y_top], color="#0f172a", lw=1.8, zorder=4)

    # RC2 loop (Diffusion: R2, C2, U2)
    x_rc2_l, x_rc2_r = 0.54, 0.72
    ax_b.plot([x_rc2_l, x_rc2_l], [y_rc_dn, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    ax_b.plot([x_rc2_r, x_rc2_r], [y_rc_dn, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    ax_b.plot([x_rc2_l, x_rc2_l + 0.03], [y_rc_up, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    draw_resistor_h(ax_b, x_rc2_l + 0.03, x_rc2_r - 0.03, y_rc_up, height=0.018, n_zigs=4, color="#047857", lw=1.6)
    ax_b.plot([x_rc2_r - 0.03, x_rc2_r], [y_rc_up, y_rc_up], color="#0f172a", lw=1.6, zorder=4)
    ax_b.text((x_rc2_l + x_rc2_r) / 2, y_rc_up + 0.035, r"$R_2$ (Warburg Diffusion)", ha="center", va="center",
              fontsize=7.8, fontweight="bold", color="#047857", zorder=6)
    x_c2 = (x_rc2_l + x_rc2_r) / 2
    ax_b.plot([x_rc2_l, x_c2 - 0.010], [y_rc_dn, y_rc_dn], color="#0f172a", lw=1.6, zorder=4)
    draw_capacitor_h(ax_b, x_c2, y_rc_dn, gap=0.018, plate_h=0.024, color="#047857", lw=1.8)
    ax_b.plot([x_c2 + 0.010, x_rc2_r], [y_rc_dn, y_rc_dn], color="#0f172a", lw=1.6, zorder=4)
    ax_b.text(x_c2, y_rc_dn - 0.062, r"$C_2$ ($\tau_2 = R_2 C_2$)" + "\n" + r"$+ \;\; U_{2,k} \;\; -$",
              ha="center", va="center", fontsize=7.6, color="#047857", zorder=6)

    # Neural Overpotential Compensator element in series before terminal
    ax_b.plot([x_rc2_r, 0.76], [y_top, y_top], color="#0f172a", lw=1.8, zorder=4)
    nn_box = FancyBboxPatch((0.76, y_top - 0.036), 0.11, 0.072,
                            boxstyle="round,pad=0.002,rounding_size=0.01",
                            facecolor="#fef2f2", edgecolor="#dc2626", lw=1.6, ls="-", zorder=5)
    ax_b.add_patch(nn_box)
    ax_b.text(0.815, y_top, r"$\delta\hat{V}_{t,\mathrm{comp},k}$" + "\n(PIDKF MLP)",
              ha="center", va="center", fontsize=7.6, fontweight="bold", color="#b91c1c", zorder=6)
    ax_b.plot([0.87, 0.93], [y_top, y_top], color="#0f172a", lw=1.8, zorder=4)

    # Bottom wire and terminals
    ax_b.plot([x_src, 0.93], [y_bot, y_bot], color="#0f172a", lw=1.8, zorder=4)
    term_pos = Circle((0.93, y_top), 0.010, facecolor="white", edgecolor="#0f172a", lw=1.8, zorder=6)
    term_neg = Circle((0.93, y_bot), 0.010, facecolor="white", edgecolor="#0f172a", lw=1.8, zorder=6)
    ax_b.add_patch(term_pos)
    ax_b.add_patch(term_neg)
    ax_b.text(0.93, y_top + 0.028, r"$+$", fontsize=11, fontweight="bold", ha="center", va="center")
    ax_b.text(0.93, y_bot - 0.028, r"$-$", fontsize=11, fontweight="bold", ha="center", va="center")
    ax_b.text(0.93, (y_top + y_bot) / 2, r"$V_{t,k}$", fontsize=10.5, fontweight="bold", color="#0f172a", ha="center", va="center")
    draw_arrow(ax_b, 0.875, y_top + 0.045, 0.935, y_top + 0.045, color="#dc2626", lw=1.6, mutation=11)
    ax_b.text(0.905, y_top + 0.072, r"$I_k > 0$", fontsize=8.2, fontweight="bold", color="#dc2626", ha="center")

    # Parameter Allocation Table / Cards at bottom of Panel (b)
    draw_box(ax_b, 0.03, 0.04, 0.94, 0.30, fc="#eff6ff", ec="#2563eb", lw=1.3, radius=0.02,
             header_bg="#dbeafe",
             title="Observer Parameter Allocation & Structural Identifiability Principle",
             title_color="#1e3a8a", title_fs=9.3)

    draw_box(ax_b, 0.05, 0.065, 0.28, 0.17, fc="#ffffff", ec="#64748b", lw=1.1, radius=0.012,
             title="Fixed-Parameter Filters",
             subtitle="EKF, AEKF, SMO:\nFreeze nominal fresh\n" + r"$(R_0, R_1, R_2, C_1, C_2, Q_{\mathrm{init}})$",
             title_fs=8.2, sub_fs=7.5, title_color="#334155")

    draw_box(ax_b, 0.36, 0.065, 0.28, 0.17, fc="#ffffff", ec="#16a34a", lw=1.1, radius=0.012,
             title="Dual EKF (DEKF)",
             subtitle=r"Co-estimates $\hat{\theta}_k = [\hat{R}_{0,k}, 1/\hat{Q}_{\mathrm{act},k}]^T$" + "\n" +
                      r"Freezes $(R_1, R_2, C_1, C_2)$ to prevent" + "\nstructural collinearity",
             title_fs=8.2, sub_fs=7.5, title_color="#166534")

    draw_box(ax_b, 0.67, 0.065, 0.28, 0.17, fc="#fef2f2", ec="#dc2626", lw=1.3, radius=0.012,
             title="Proposed Hybrid PIDKF",
             subtitle=r"DEKF tracks $(\hat{R}_{0,k}, \hat{Q}_{\mathrm{act},k})$ + MLP" + "\n" +
                      r"compensates $\delta V_{\mathrm{pol},k}^*$ under" + "\n" +
                      r"passivity $I_k \delta\hat{V}_{t,\mathrm{comp},k} \leq 0$",
             title_fs=8.2, sub_fs=7.5, title_color="#991b1b")

    # =========================================================================
    # PANEL (c): OCV-SOC Curve & Hysteresis Loop (Bottom-Left, Quantitative)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    z = np.linspace(0.0, 1.0, 400)
    # Exact Eq. (11) NMC 5th-order polynomial
    vocv_nmc = 9.204 * z**5 - 28.841 * z**4 + 33.376 * z**3 - 17.419 * z**2 + 4.674 * z + 3.212
    # Exact Section 2.4 LFP 5th-order polynomial over operational domain [0.12, 0.95]
    z_lfp = np.linspace(0.12, 0.95, 350)
    vocv_lfp = 3.415 - 0.286 * z_lfp + 1.831 * z_lfp**2 - 4.120 * z_lfp**3 + 4.185 * z_lfp**4 - 1.545 * z_lfp**5
    M_hyst = 0.030  # 30 mV hysteresis envelope

    ax_c.plot(z * 100, vocv_nmc, color="#2563eb", lw=2.4, label=r"NMC811 $V_{\mathrm{ocv}}(z)$ (Eq. 11, Monotonic 5th-Order)")
    ax_c.plot(z_lfp * 100, vocv_lfp, color="#16a34a", lw=2.2, label=r"$\mathrm{LiFePO_4}$ Equilibrium $V_{\mathrm{ocv}}^{\mathrm{LFP}}(z)$")
    ax_c.fill_between(z_lfp * 100, vocv_lfp - M_hyst, vocv_lfp + M_hyst,
                      color="#16a34a", alpha=0.22, label=r"LFP Hysteresis Envelope $\pm M_0$ ($\pm 30\;\mathrm{mV}$)")

    # Highlight LFP flat plateau region (20% - 80%)
    ax_c.axvspan(20, 80, color="#fef08a", alpha=0.28, zorder=0)
    ax_c.annotate("Biphasic Plateau (20%–80% SOC)\n" + r"$\mathrm{LiFePO_4 \leftrightarrow FePO_4}$ ($F' = C - P = 0$)" + "\n" +
                  r"Flat slope $\approx 0.08\;\mathrm{V}$ & $\pm 30\;\mathrm{mV}$ Hysteresis",
                  xy=(50, 3.31), xytext=(44, 3.66),
                  fontsize=8.5, fontweight="bold", color="#854d0e",
                  bbox=dict(boxstyle="round,pad=0.35", fc="#fefce8", ec="#eab308", lw=1.1),
                  arrowprops=dict(arrowstyle="->", color="#ca8a04", lw=1.5))

    ax_c.set_title("(c) Calibrated Open-Circuit Voltage Curves & LFP Hysteresis Loop",
                   fontsize=11.2, fontweight="bold", color="#0f172a", pad=8)
    ax_c.set_xlabel("State of Charge, SOC (%)", fontsize=10.2, fontweight="bold")
    ax_c.set_ylabel("Open-Circuit Potential, $V_{\\mathrm{ocv}}$ (V)", fontsize=10.2, fontweight="bold")
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(3.05, 4.30)
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.legend(loc="upper left", fontsize=8.3, framealpha=0.92, edgecolor="#cbd5e1")

    # =========================================================================
    # PANEL (d): OCV Slope dVocv/dSOC & Observability Condition Number (Bottom-Right)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    dvocv_nmc = 46.020 * z**4 - 115.364 * z**3 + 100.128 * z**2 - 34.838 * z + 4.674
    dvocv_lfp = -0.286 + 3.662 * z_lfp - 12.36 * z_lfp**2 + 16.74 * z_lfp**3 - 7.725 * z_lfp**4

    ax_d.semilogy(z * 100, dvocv_nmc, color="#2563eb", lw=2.4,
                  label=r"NMC Slope $V_{\mathrm{ocv}}'(z) \in [0.481, 4.674]\;\mathrm{V}$")
    ax_d.semilogy(z_lfp * 100, dvocv_lfp, color="#dc2626", lw=2.4, ls="-",
                  label=r"LFP Slope $V_{\mathrm{ocv}}'(z)$ (Drops to $0.065\;\mathrm{to}\;0.08\;\mathrm{V}$ on Plateau)")

    ax_d.axvspan(20, 80, color="#fee2e2", alpha=0.30, zorder=0)
    ax_d.axhline(0.08, color="#991b1b", ls=":", lw=1.3)
    ax_d.text(50, 0.21, r"LFP Plateau Observability Bottleneck ($V_{\mathrm{ocv}}' \approx 0.08\;\mathrm{V}$)",
              fontsize=8.3, fontweight="bold", color="#991b1b", ha="center",
              bbox=dict(boxstyle="round,pad=0.22", fc="#fff1f2", ec="#fca5a5", lw=0.9))

    # Annotate Observability Determinant & Condition Number
    ax_d.text(50, 0.010,
              r"Observability Determinant: $|\det(\mathcal{O}_k)| \propto V_{\mathrm{ocv}}'(z_k)$ (Eqs. 12, 16)" + "\n" +
              r"$\bullet$ NMC: $\kappa(\mathcal{O}_k) \approx 3.2 \times 10^4 \;\Rightarrow\;$ Strong Innovation Gain" + "\n" +
              r"$\bullet$ LFP Plateau: $\kappa(\mathcal{O}_k) > 1.99 \times 10^5 \;\Rightarrow\;$ Gain Vanishes / SMO Chattering",
              ha="center", va="center", fontsize=8.0, color="#0f172a",
              bbox=dict(boxstyle="round,pad=0.38", fc="#f8fafc", ec="#64748b", lw=1.1))

    ax_d.set_title(r"(d) Analytical Observability Gradient $dV_{\mathrm{ocv}}/d\mathrm{SOC}$ & Condition Number Surge",
                   fontsize=11.2, fontweight="bold", color="#0f172a", pad=8)
    ax_d.set_xlabel("State of Charge, SOC (%)", fontsize=10.2, fontweight="bold")
    ax_d.set_ylabel(r"OCV Derivative, $dV_{\mathrm{ocv}}/d\mathrm{SOC}$ (V) [Log Scale]", fontsize=10.0, fontweight="bold")
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0.003, 8.0)
    ax_d.grid(True, which="both", linestyle="--", alpha=0.35)
    ax_d.legend(loc="upper right", fontsize=8.3, framealpha=0.92, edgecolor="#cbd5e1")

    for sp in save_paths:
        fig.savefig(sp, dpi=300)
        print(f"Saved {sp}")
    plt.close(fig)


def generate_framework_figure(save_paths):
    fig = plt.figure(figsize=(15.5, 9.8))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.93])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Proposed Physics-Informed Dual Kalman Filter (PIDKF) Hybrid Architecture and 10-Phase Benchmarking Framework",
        fontsize=13.8, fontweight="bold", color="#0f172a", y=0.975
    )

    # =========================================================================
    # TOP SECTION (y = 0.42 to 0.99): Proposed Hybrid PIDKF Architecture
    # =========================================================================
    draw_box(ax, 0.01, 0.42, 0.98, 0.56, fc="#ffffff", ec="#94a3b8", lw=1.4, radius=0.018, zorder=1)
    ax.text(0.025, 0.952, "(a) Closed-Loop Hybrid Physics-Informed Dual Kalman Filter (PIDKF) with Butler–Volmer Passivity & Exogenous Decoupling",
            fontsize=11.2, fontweight="bold", color="#0f172a", zorder=6)

    # 1. Left Column: Physical Battery Plant & BMS Telemetry (x = 0.025 to 0.215)
    draw_box(ax, 0.025, 0.45, 0.19, 0.47, fc="#f0f9ff", ec="#0284c7", lw=1.5, radius=0.016,
             header_bg="#bae6fd", title="1. Battery Plant & Sensors", title_color="#0c4a6e", title_fs=9.6)

    draw_box(ax, 0.038, 0.72, 0.164, 0.13, fc="#ffffff", ec="#38bdf8", lw=1.2, radius=0.012,
             title="NMC811 & LFP Cells",
             subtitle="INR18650-20R (2.0 Ah)\nANR26650M1-B (2.3 Ah)\n" + r"$\mathrm{SOH} \in [80\%, 100\%]$" + "\n" + r"$T \in [0^\circ\mathrm{C}, 45^\circ\mathrm{C}]$",
             title_fs=8.5, sub_fs=7.6, title_color="#0369a1")

    draw_box(ax, 0.038, 0.56, 0.164, 0.13, fc="#ffffff", ec="#38bdf8", lw=1.2, radius=0.012,
             title="Causal BMS Telemetry",
             subtitle=r"Current $I_k = I_{\mathrm{true}} + I_{\mathrm{bias}} + n_i$" + "\n" +
                      r"Voltage $V_{\mathrm{meas},k} = V_{t,k} + v_k$" + "\n" +
                      r"Surface Temp $T_k$ ($1\;\mathrm{Hz}$)",
             title_fs=8.5, sub_fs=7.6, title_color="#0369a1")

    draw_box(ax, 0.038, 0.465, 0.164, 0.075, fc="#fefce8", ec="#eab308", lw=1.1, radius=0.010,
             title="Rest-Period Gate",
             subtitle=r"If $|I_k| < 10\;\mathrm{mA}$: $\mathbf{K}_k=\mathbf{0},\;\mathbf{K}_{\theta,k}=\mathbf{0}$",
             title_fs=8.0, sub_fs=7.3, title_color="#854d0e")

    draw_arrow(ax, 0.12, 0.72, 0.12, 0.69, color="#0284c7", lw=1.6)

    # 2. Center-Left Column: Dual Extended Kalman Filter (DEKF) Backbone (x = 0.245 to 0.555)
    draw_box(ax, 0.245, 0.45, 0.31, 0.47, fc="#f0fdf4", ec="#16a34a", lw=1.5, radius=0.016,
             header_bg="#bbf7d0", title="2. Dual EKF State & Parameter Backbone (980 FLOPs)",
             title_color="#14532d", title_fs=9.6)

    # Fast State Filter Box
    draw_box(ax, 0.26, 0.69, 0.28, 0.165, fc="#ffffff", ec="#22c55e", lw=1.3, radius=0.012,
             title="Fast State Filter (Timescale $\\tau_{\\mathrm{state}} \\sim 10^1\\;\\mathrm{to}\\;10^2\\;\\mathrm{s}$)",
             subtitle=r"Prior: $\hat{\mathbf{x}}_k^- = \mathbf{A}_{k-1}\hat{\mathbf{x}}_{k-1}^+ + \mathbf{B}_{k-1}I_{k-1} + \mathbf{g}_{k-1}$" + "\n" +
                      r"State: $\hat{\mathbf{x}}_k = [\hat{z}_k,\, \hat{U}_{1,k},\, \hat{U}_{2,k},\, (\hat{V}_{h,k})]^T$" + "\n" +
                      r"ECM Voltage: $\hat{V}_{t,\mathrm{ECM},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - (\hat{V}_{h,k}^-) - I_k\hat{R}_{0,k}^-$",
             title_fs=8.4, sub_fs=7.4, title_color="#166534")

    # Slow Parameter Filter Box
    draw_box(ax, 0.26, 0.47, 0.28, 0.185, fc="#ffffff", ec="#15803d", lw=1.3, radius=0.012,
             title="Slow Parameter Filter (Timescale $\\tau_{\\mathrm{SOH}} \\sim 10^5\\;\\mathrm{to}\\;10^6\\;\\mathrm{s}$)",
             subtitle=r"Vector: $\hat{\theta}_k = [\hat{R}_{0,k},\; 1/\hat{Q}_{\mathrm{act},k}]^T$,  $\hat{\theta}_k^+ = \hat{\theta}_k^- + \mathbf{K}_{\theta,k} e_{k,\mathrm{PIDKF}}$" + "\n" +
                      r"Fréchet Sensitivity: $\mathbf{C}_{\theta,k} = [-I_k, \; -\frac{\eta\Delta t I_{k-1}}{3600}V_{\mathrm{ocv}}'(\hat{z}_k^+)]$" + "\n" +
                      r"Thermal-Inverted SOH: $\mathrm{SOH}_{\mathrm{ref},k} = \frac{1}{Q_{\mathrm{init}}\hat{\theta}_{2,k}f_Q(T_k)}$",
             title_fs=8.4, sub_fs=7.4, title_color="#14532d")

    # Mutual coupling arrows between State and Parameter filters
    draw_arrow(ax, 0.36, 0.69, 0.36, 0.655, color="#15803d", lw=1.5, mutation=11)
    draw_arrow(ax, 0.44, 0.655, 0.44, 0.69, color="#15803d", lw=1.5, mutation=11)
    ax.text(0.40, 0.672, r"$\hat{z}_k^+ \leftrightarrow \hat{\theta}_k^+$", fontsize=7.6, fontweight="bold", color="#166534", ha="center", va="center")

    # 3. Center-Right Column: Physics-Constrained Neural Compensator (x = 0.585 to 0.975)
    draw_box(ax, 0.585, 0.45, 0.39, 0.47, fc="#fef2f2", ec="#dc2626", lw=1.5, radius=0.016,
             header_bg="#fecaca", title="3. Physics-Constrained Neural Overpotential Compensator & Innovation Update",
             title_color="#7f1d1d", title_fs=9.5)

    # Neural MLP box
    draw_box(ax, 0.60, 0.70, 0.21, 0.155, fc="#ffffff", ec="#ef4444", lw=1.3, radius=0.012,
             title="Compact MLP (560 FLOPs)",
             subtitle=r"Input $\mathbf{u}_k \in \mathbb{R}^6$: $[I_k, T_k, \hat{z}_k^-, \hat{U}_{1,k}^-, \hat{U}_{2,k}^-, \mathrm{SOH}_{\mathrm{ref},k}]^T$" + "\n" +
                      r"Architecture: $6 \rightarrow 16 \rightarrow 8 \rightarrow 1$ (GELU)" + "\n" +
                      r"257 Params ($1.03\;\mathrm{KB}$ Flash, Zero Target Leakage)",
             title_fs=8.3, sub_fs=7.2, title_color="#991b1b")

    # Butler-Volmer & Passivity Projection Box
    draw_box(ax, 0.825, 0.70, 0.138, 0.155, fc="#fffbeb", ec="#d97706", lw=1.3, radius=0.012,
             title="Passivity & Bounds",
             subtitle=r"$|\delta\hat{V}_{t,\mathrm{comp},k}| \leq V_{\mathrm{pol}}^{\max}(I_k)$" + "\n" +
                      r"($\leq 20\;\mathrm{mV}$ Envelope)" + "\n" +
                      r"Dissipative Constraint:" + "\n" +
                      r"$I_k \cdot \delta\hat{V}_{t,\mathrm{comp},k} \leq 0$",
             title_fs=8.2, sub_fs=7.2, title_color="#92400e")

    draw_arrow(ax, 0.81, 0.775, 0.825, 0.775, color="#dc2626", lw=1.6, mutation=11)

    # Innovation & Exogenous Decoupling Box
    draw_box(ax, 0.60, 0.47, 0.363, 0.195, fc="#ffffff", ec="#b91c1c", lw=1.3, radius=0.012,
             title="Compensated Innovation & Exogenous Sensitivity Decoupling",
             subtitle=r"Compensated Voltage: $\hat{V}_{t,\mathrm{PIDKF},k}^- = \hat{V}_{t,\mathrm{ECM},k}^- + \delta\hat{V}_{t,\mathrm{comp},k} \;\;\Rightarrow\;\; e_{k,\mathrm{PIDKF}} = V_{\mathrm{meas},k} - \hat{V}_{t,\mathrm{PIDKF},k}^-$" + "\n" +
                      r"Expanded Variance: $R_{\mathrm{eff},k} = R_v + \sigma_{\mathrm{NN}}^2 + \sigma_{\mathrm{approx}}^2 = 1.04\times 10^{-3}\;\mathrm{V}^2$  (Joseph Covariance Update)" + "\n" +
                      r"Exogenous Jacobian: $\mathbf{H}_{\mathrm{PIDKF},k} \approx \mathbf{H}_k,\;\; \mathbf{C}_{\theta,\mathrm{PIDKF},k} \approx \mathbf{C}_{\theta,k}$  (Omits Online Autograd: $-37.3\%$ MCU Cycles!)" + "\n" +
                      r"Guaranteed First-Moment & Mean-Square UUB Stability (Eqs. 24–25) under Intermittent PE ($N=30\;\mathrm{s}$)",
             title_fs=8.4, sub_fs=7.25, title_color="#7f1d1d")

    # Connect columns in Top Section
    draw_arrow(ax, 0.215, 0.77, 0.26, 0.77, color="#0284c7", lw=1.8, mutation=13)
    draw_arrow(ax, 0.215, 0.60, 0.26, 0.60, color="#0284c7", lw=1.8, mutation=13)
    draw_arrow(ax, 0.54, 0.775, 0.60, 0.775, color="#16a34a", lw=1.8, mutation=13)
    draw_arrow(ax, 0.894, 0.70, 0.894, 0.665, color="#d97706", lw=1.8, mutation=13)
    draw_arrow(ax, 0.60, 0.565, 0.54, 0.565, color="#dc2626", lw=1.8, mutation=13)
    ax.text(0.57, 0.592, r"$e_{k,\mathrm{PIDKF}}$", fontsize=7.8, fontweight="bold", color="#b91c1c", ha="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # =========================================================================
    # BOTTOM SECTION (y = 0.015 to 0.395): Three-Tier Data & 10-Phase Benchmark
    # =========================================================================
    draw_box(ax, 0.01, 0.015, 0.98, 0.38, fc="#ffffff", ec="#94a3b8", lw=1.4, radius=0.018, zorder=1)
    ax.text(0.025, 0.368, "(b) Strict Batch-Split Validation Protocol, 7-Algorithm Taxonomy, and 10-Phase Multi-Dimensional Evaluation",
            fontsize=11.2, fontweight="bold", color="#0f172a", zorder=6)

    # Row 1 inside bottom section: Data Splits & 7 Paradigms (y = 0.225 to 0.345)
    draw_box(ax, 0.025, 0.225, 0.30, 0.125, fc="#eff6ff", ec="#2563eb", lw=1.3, radius=0.012,
             header_bg="#dbeafe", title="Empirical Datasets & Zero-Leakage Split", title_color="#1e3a8a", title_fs=8.8,
             subtitle="CALCE (INR20R, A123), Oxford EIS, NASA PCoE\nBatch A (Train/Calib) $\\perp$ Batch B (Hold-Out Test)\nArbin Reference Uncertainty $\\sigma_{\\mathrm{SOC,ref}} \\leq \\pm 0.036\\%$",
             sub_fs=7.4)

    draw_box(ax, 0.345, 0.225, 0.32, 0.125, fc="#f5f3ff", ec="#7c3aed", lw=1.3, radius=0.012,
             header_bg="#ede9fe", title="7 Evaluated Estimation Architectures", title_color="#4c1d95", title_fs=8.8,
             subtitle="1. Model-Based: EKF (3.87 $\\mu$s), UKF (12.42 $\\mu$s), SMO (1.67 $\\mu$s)\n"
                      "2. Adaptive / Joint: Sage–Husa AEKF (5.26 $\\mu$s), DEKF (8.69 $\\mu$s)\n"
                      "3. Deep / Hybrid: BiLSTM-Attn (2.45 ms), PINN-SOC (130 $\\mu$s), PIDKF (13.65 $\\mu$s)",
             sub_fs=7.3)

    draw_box(ax, 0.685, 0.225, 0.29, 0.125, fc="#ecfdf5", ec="#059669", lw=1.3, radius=0.012,
             header_bg="#d1fae5", title="Bare-Metal ARM Cortex-M4F Deployment", title_color="#065f46", title_fs=8.8,
             subtitle="STM32F407VG (168 MHz, IEEE-754 Single-Precision FPU)\n"
                      "Zero Dynamic Heap Allocation (MISRA-C:2012 Compliant)\n"
                      "PIDKF: 1,540 FLOPs/Step, 8.2 KB Flash, 1.1 KB SRAM, 13.65 $\\mu$s",
             sub_fs=7.4)

    draw_arrow(ax, 0.325, 0.287, 0.345, 0.287, color="#2563eb", lw=1.6, mutation=11)
    draw_arrow(ax, 0.665, 0.287, 0.685, 0.287, color="#7c3aed", lw=1.6, mutation=11)

    # Row 2 inside bottom section: 10 Experimental Phases (2 rows of 5 cards, y = 0.03 to 0.205)
    phases = [
        ("Phase 1: Dynamic Cycles", "UDDS, US06, FUDS, DST\nFresh Cell (100% SOH, 25°C)\nEKF: 0.063% | PIDKF: 0.185%", "#e0f2fe", "#0284c7"),
        ("Phase 2: Multi-Stage Aging", "SOH 100% → 95% → 90% → 80%\nCapacity Fade & Resistance ↑\nPIDKF: 0.422% @ 85% SOH", "#dcfce7", "#16a34a"),
        ("Phase 3: Thermal Stress", "Cold (0°C, 10°C) & Hot (45°C)\nArrhenius Viscosity Surge\nDEKF: 0.283% @ 0°C", "#fef3c7", "#d97706"),
        ("Phase 4: Convergence", "±20% & −40% Initial Offset\nRiccati Gain Adaptation\nKalman <35 s vs. ML Failure", "#f3e8ff", "#9333ea"),
        ("Phase 5: Radar Synthesis", "5-Axis Holistic Trade-Off:\nAccuracy, Aging, Thermal,\nConvergence & MCU Cost", "#f1f5f9", "#475569"),
        ("Phase 6: LFP Chemistry", "20%–80% Flat Plateau + Vh\nObservability Rank Analysis\nPIDKF: 0.151% (Aged LFP)", "#fae8ff", "#c026d3"),
        ("Phase 7: Sensor Faults", "+250 mA Hall Offset & EMI\n+15 mV Voltage Bias Drift\nSMO: 0.629% (Manifold Clamp)", "#ffe4e6", "#e11d48"),
        ("Phase 8: EIS Diagnostics", "0.01 Hz–10 kHz Nyquist Spectra\nCharge-Transfer Rct vs. SOH\nLinear R² = 0.994 Anchor", "#e0e7ff", "#4f46e5"),
        ("Phase 9: 8S Pack Scaling", "Cell-to-Cell Dispersion (±4% Q)\nThermal Core Gradient (+4.8°C)\nWeakest Cell #4 RMSE: 0.337%", "#ccfbf1", "#0d9488"),
        ("Phase 10: Hybrid PIDKF", "Ablation & Cortex-M4 Pareto\n−8.7% (NMC) & −51.0% (LFP)\nRMSE vs. DEKF at 13.65 µs", "#fee2e2", "#dc2626"),
    ]

    card_w = 0.184
    card_h = 0.078
    gap_x = 0.011
    start_x = 0.025

    for idx, (p_title, p_sub, p_fc, p_ec) in enumerate(phases):
        row = idx // 5
        col = idx % 5
        cx = start_x + col * (card_w + gap_x)
        cy = 0.120 if row == 0 else 0.030
        draw_box(ax, cx, cy, card_w, card_h, fc="#ffffff", ec=p_ec, lw=1.2, radius=0.008,
                 header_bg=p_fc, title=p_title, subtitle=p_sub,
                 title_fs=7.9, sub_fs=6.85, title_color="#0f172a", sub_color="#334155")

    for sp in save_paths:
        fig.savefig(sp, dpi=300)
        print(f"Saved {sp}")
    plt.close(fig)


if __name__ == "__main__":
    folders = ["results", "manuscript", "submission_journal_of_energy_storage"]
    mech_paths = [os.path.join(f, "fig0_electrochemical_mechanism.png") for f in folders]
    fw_paths = [os.path.join(f, "fig0_pidkf_framework.png") for f in folders]
    generate_mechanism_figure(mech_paths)
    generate_framework_figure(fw_paths)
