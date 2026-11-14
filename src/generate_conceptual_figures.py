r"""
Generate publication-grade vector/600-DPI conceptual figures using TikZ + pdflatex:
1. fig0_electrochemical_mechanism.png (600 DPI)
2. fig0_pidkf_framework.png (600 DPI)
All text strictly black, zero overlapping nodes, crisp Computer Modern typography.
"""

import os
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import fitz  # PyMuPDF
from PIL import Image, ImageChops

TEXLIVE_BIN = r"d:\texlive\2026\bin\windows"
PDFLATEX = os.path.join(TEXLIVE_BIN, "pdflatex.exe")

# =============================================================================
# 1. TikZ LaTeX Source for Mechanism Top Half (Panels a & b)
#    Canvas: 30.8 cm x 9.8 cm (1 unit = 1 cm)
# =============================================================================
TIKZ_MECH_TOP = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=32.0cm,paperheight=10.6cm,margin=0.4cm]{geometry}
\pagestyle{empty}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,backgrounds,fit,decorations.pathmorphing}

\begin{document}\noindent
\begin{tikzpicture}[
    x=1cm, y=1cm,
    >=Latex,
    font=\small,
    text=black,
    draw=black,
    panelbox/.style={draw=black, line width=1.0pt, rounded corners=3pt, fill=white},
    zonebox/.style={draw=black, line width=0.75pt, rounded corners=2pt, fill=gray!4},
    cardbox/.style={draw=black, line width=0.65pt, rounded corners=2pt, fill=white, align=center},
    headerbox/.style={draw=black, line width=0.75pt, fill=gray!15, font=\footnotesize\bfseries, align=center},
    arr/.style={-{Latex[length=2.2mm,width=1.6mm]}, line width=0.85pt, draw=black},
    darr/.style={-{Latex[length=2.0mm,width=1.5mm]}, line width=0.75pt, draw=black, densely dashed},
    resistor/.style={decorate, decoration={zigzag, segment length=2.4mm, amplitude=1.1mm}, line width=0.95pt}
]

% =========================================================================
% PANEL (a): Left Box (x = 0.0 to 15.1, y = 0.0 to 9.7)
% =========================================================================
\draw[panelbox] (0,0) rectangle (15.1,9.7);
\node[anchor=west, font=\normalsize\bfseries] at (0.30, 9.25) {(a) Microscopic Electrode Degradation and Electro-Thermal Coupling};

% --- Anode Zone (x = 0.25 to 4.75, y = 3.25 to 8.80) ---
\draw[zonebox] (0.25, 3.25) rectangle (4.75, 8.80);
\draw[headerbox] (0.25, 8.20) rectangle (4.75, 8.80);
\node[font=\footnotesize\bfseries] at (2.50, 8.50) {Graphite Anode ($\mathrm{Li_xC_6}$)};

\node[font=\scriptsize\bfseries] at (2.50, 7.86) {Porous SEI Film ($\delta_{\mathrm{SEI}}$)};
\filldraw[fill=gray!18, draw=black, line width=0.8pt, densely dashed] (2.50, 6.26) circle (1.32cm);
\filldraw[fill=white, draw=black, line width=0.9pt] (2.50, 6.26) circle (1.00cm);
\node[align=center, font=\scriptsize\bfseries] at (2.50, 6.26) {$\mathrm{Li_xC_6}$\\Solid Diffusion\\$D_s(T)$};

\node[align=left, font=\scriptsize] at (2.50, 4.02) {
    $\bullet$ Traps cyclable $\mathrm{Li^+}$ (LLI)\\[2pt]
    $\bullet$ Film impedance $\uparrow \Delta R_{\mathrm{SEI}}$\\[2pt]
    $\bullet$ Cold $\mathrm{Li}$ plating ($0^\circ\mathrm{C}$)
};

% --- Electrolyte & Interface Zone (x = 4.95 to 9.85, y = 3.25 to 8.80) ---
\draw[zonebox] (4.95, 3.25) rectangle (9.85, 8.80);
\draw[headerbox] (4.95, 8.20) rectangle (9.85, 8.80);
\node[font=\footnotesize\bfseries] at (7.40, 8.50) {Electrolyte \& Interface};

\draw[arr] (5.30, 7.30) -- (9.50, 7.30);
\node[font=\scriptsize\bfseries] at (7.40, 7.62) {Discharge: $\mathrm{Li^+}$ Migration ($I > 0$)};

\draw[darr] (9.50, 6.40) -- (5.30, 6.40);
\node[font=\scriptsize] at (7.40, 6.06) {Charge: $\mathrm{Li^+}$ Intercalation ($I < 0$)};

\draw[cardbox] (5.18, 3.45) rectangle (9.62, 5.55);
\node[align=center, font=\scriptsize] at (7.40, 4.50) {
    \textbf{Butler--Volmer Kinetics}\\[4pt]
    $\eta_{\mathrm{act}} = \dfrac{2RT}{F}\mathrm{arcsinh}\!\left(\dfrac{I}{2 A i_0}\right)$\\[5pt]
    Desolvation: $E_{a,1}^{\mathrm{cold}} = 28.4\;\mathrm{kJ/mol}$
};

% --- Cathode Zone (x = 10.05 to 14.85, y = 3.25 to 8.80) ---
\draw[zonebox] (10.05, 3.25) rectangle (14.85, 8.80);
\draw[headerbox] (10.05, 8.20) rectangle (14.85, 8.80);
\node[font=\footnotesize\bfseries] at (12.45, 8.50) {Cathode Host (NMC / LFP)};

\draw[cardbox] (10.25, 5.85) rectangle (14.65, 7.95);
\node[align=center, font=\scriptsize] at (12.45, 6.90) {
    \textbf{NMC811 (INR18650-20R)}\\[2.5pt]
    Single-phase solid solution\\[1.5pt]
    $V_{\mathrm{ocv}}^\prime(z) \in [0.481, 4.674]\;\mathrm{V}$\\[1.5pt]
    Full rank ($\mathrm{rank}\,\mathcal{O}_k = 3$)
};

\draw[cardbox] (10.25, 3.45) rectangle (14.65, 5.60);
\node[align=center, font=\scriptsize] at (12.45, 4.525) {
    \textbf{$\mathrm{LiFePO_4}$ (ANR26650M1-B)}\\[2.5pt]
    Biphasic $\mathrm{LiFePO_4 \leftrightarrow FePO_4}$\\[1.5pt]
    $V_{\mathrm{ocv}}^\prime \approx 0.08\;\mathrm{V}$ + Hysteresis $V_h$\\[1.5pt]
    $\kappa(\mathcal{O}_k) > 1.99\times 10^5$ ($F'=0$)
};

% --- Bottom Coupled Loop in Panel (a) (x = 0.25 to 14.85, y = 0.22 to 2.98) ---
\draw[zonebox] (0.25, 0.22) rectangle (14.85, 2.98);
\draw[headerbox] (0.25, 2.40) rectangle (14.85, 2.98);
\node[font=\footnotesize\bfseries] at (7.55, 2.69) {Coupled Electro-Thermal-Aging Loop (Arrhenius Kinetics \& Energy Balance)};

\draw[cardbox] (0.45, 0.38) rectangle (4.85, 2.22);
\node[align=center, font=\scriptsize] at (2.65, 1.30) {
    \textbf{Thermal Balance (Eq.\ 5)}\\[3pt]
    $C_{\mathrm{th}}\dfrac{dT}{dt} = I^2 R_0 + I(U_1+U_2)$\\[3pt]
    $- IT\dfrac{\partial V_{\mathrm{ocv}}}{\partial T} - \dfrac{T-T_{\mathrm{amb}}}{R_{\mathrm{th}}}$
};

\draw[cardbox] (5.35, 0.38) rectangle (9.75, 2.22);
\node[align=center, font=\scriptsize] at (7.55, 1.30) {
    \textbf{Arrhenius \& Aging (Eqs.\ 8--10)}\\[3pt]
    $R_i(T,\mathrm{SOH}) = R_{i,\mathrm{nom}}\,f_{T,i}(T)$\\[3pt]
    $\times\,[1 + \kappa_{R_i}(1-\mathrm{SOH})]$
};

\draw[cardbox] (10.25, 0.38) rectangle (14.65, 2.22);
\node[align=center, font=\scriptsize] at (12.45, 1.30) {
    \textbf{Usable Capacity (Eq.\ 10)}\\[3pt]
    $Q_{\mathrm{act}}(T,\mathrm{SOH}) = \mathrm{SOH}\cdot Q_{\mathrm{init}}\,f_Q(T)$\\[3pt]
    $0.890\times\;(0^\circ\mathrm{C}),\;\;1.030\times\;(45^\circ\mathrm{C})$
};

\draw[arr] (4.85, 1.30) -- (5.35, 1.30);
\draw[arr] (9.75, 1.30) -- (10.25, 1.30);

% =========================================================================
% PANEL (b): Right Box (x = 15.5 to 30.8, y = 0.0 to 9.7)
% =========================================================================
\draw[panelbox] (15.5,0) rectangle (30.8,9.7);
\node[anchor=west, font=\normalsize\bfseries] at (15.80, 9.25) {(b) Augmented 2nd-Order RC Equivalent Circuit and Parameter Allocation};

% --- Circuit Diagram Box (x = 15.75 to 30.55, y = 3.25 to 8.80) ---
\draw[zonebox, fill=white] (15.75, 3.25) rectangle (30.55, 8.80);

\coordinate (bot_l) at (16.65, 4.25);
\coordinate (top_l) at (16.65, 7.05);

% Left vertical branch: Vocv + Vh
\draw[line width=0.95pt] (bot_l) -- (16.65, 4.82);
\filldraw[fill=gray!10, draw=black, line width=0.95pt] (16.65, 5.34) circle (0.50cm);
\node[font=\scriptsize\bfseries] at (16.65, 5.34) {$V_{\mathrm{ocv}}(z)$};
\node[font=\scriptsize\bfseries] at (15.98, 5.72) {$+$};
\node[font=\scriptsize\bfseries] at (15.98, 4.96) {$-$};
\draw[line width=0.95pt] (16.65, 5.84) -- (16.65, 6.10);

% Hysteresis block V_{h,k}
\filldraw[fill=gray!14, draw=black, line width=0.85pt, rounded corners=1.5pt] (16.10, 6.10) rectangle (17.20, 6.68);
\node[font=\scriptsize\bfseries] at (16.65, 6.39) {$\pm V_{h,k}$};
\draw[line width=0.95pt] (16.65, 6.68) -- (top_l);

% R0 resistor
\draw[line width=0.95pt] (top_l) -- (17.45, 7.05);
\draw[resistor] (17.45, 7.05) -- (18.85, 7.05);
\node[align=center, font=\scriptsize\bfseries] at (18.15, 7.72) {$R_0(T,\mathrm{SOH})$\\Ohmic};
\draw[line width=0.95pt] (18.85, 7.05) -- (19.50, 7.05);

% RC1 parallel loop (x = 19.50 to 22.25, y = 6.38 to 7.72)
\draw[line width=0.9pt] (19.50, 6.38) -- (19.50, 7.72);
\draw[line width=0.9pt] (22.25, 6.38) -- (22.25, 7.72);
\draw[line width=0.9pt] (19.50, 7.72) -- (20.00, 7.72);
\draw[resistor] (20.00, 7.72) -- (21.75, 7.72);
\draw[line width=0.9pt] (21.75, 7.72) -- (22.25, 7.72);
\node[font=\scriptsize\bfseries] at (20.875, 8.20) {$R_1$ (Charge Transfer)};
\draw[line width=0.9pt] (19.50, 6.38) -- (20.74, 6.38);
\draw[line width=1.3pt] (20.74, 6.10) -- (20.74, 6.66);
\draw[line width=1.3pt] (21.01, 6.10) -- (21.01, 6.66);
\draw[line width=0.9pt] (21.01, 6.38) -- (22.25, 6.38);
\node[align=center, font=\scriptsize] at (20.875, 5.52) {$C_1$ ($\tau_1 = R_1 C_1$)\\$+ \quad U_{1,k} \quad -$};

\draw[line width=0.95pt] (22.25, 7.05) -- (23.05, 7.05);

% RC2 parallel loop (x = 23.05 to 25.80, y = 6.38 to 7.72)
\draw[line width=0.9pt] (23.05, 6.38) -- (23.05, 7.72);
\draw[line width=0.9pt] (25.80, 6.38) -- (25.80, 7.72);
\draw[line width=0.9pt] (23.05, 7.72) -- (23.55, 7.72);
\draw[resistor] (23.55, 7.72) -- (25.30, 7.72);
\draw[line width=0.9pt] (25.30, 7.72) -- (25.80, 7.72);
\node[font=\scriptsize\bfseries] at (24.425, 8.20) {$R_2$ (Warburg Diffusion)};
\draw[line width=0.9pt] (23.05, 6.38) -- (24.29, 6.38);
\draw[line width=1.3pt] (24.29, 6.10) -- (24.29, 6.66);
\draw[line width=1.3pt] (24.56, 6.10) -- (24.56, 6.66);
\draw[line width=0.9pt] (24.56, 6.38) -- (25.80, 6.38);
\node[align=center, font=\scriptsize] at (24.425, 5.52) {$C_2$ ($\tau_2 = R_2 C_2$)\\$+ \quad U_{2,k} \quad -$};

% Neural Compensator element in series (x = 26.50 to 28.85)
\draw[line width=0.95pt] (25.80, 7.05) -- (26.50, 7.05);
\filldraw[fill=gray!14, draw=black, line width=0.95pt, rounded corners=2pt] (26.50, 6.48) rectangle (28.85, 7.62);
\node[align=center, font=\scriptsize\bfseries] at (27.675, 7.05) {$\delta\hat{V}_{t,\mathrm{comp},k}$\\(PIDKF MLP)};
\draw[line width=0.95pt] (28.85, 7.05) -- (29.85, 7.05);

% Bottom return wire & Terminals
\draw[line width=0.95pt] (bot_l) -- (29.85, 4.25);
\filldraw[fill=white, draw=black, line width=0.95pt] (29.85, 7.05) circle (0.10cm);
\filldraw[fill=white, draw=black, line width=0.95pt] (29.85, 4.25) circle (0.10cm);
\node[font=\small\bfseries] at (29.85, 7.42) {$+$};
\node[font=\small\bfseries] at (29.85, 3.88) {$-$};
\node[font=\small\bfseries] at (29.85, 5.65) {$V_{t,k}$};
\draw[arr] (29.05, 7.90) -- (29.95, 7.90);
\node[font=\scriptsize\bfseries] at (29.50, 8.20) {$I_k > 0$};

% KVL summary equation inside circuit box
\filldraw[fill=gray!8, draw=black, line width=0.6pt, rounded corners=1.5pt] (16.05, 3.40) rectangle (30.25, 4.00);
\node[font=\scriptsize] at (23.15, 3.70) {
    \textbf{Terminal Voltage}: $\hat{V}_{t,\mathrm{PIDKF},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - (\hat{V}_{h,k}^-) - I_k\hat{R}_{0,k}^- + \delta\hat{V}_{t,\mathrm{comp},k}$
};

% --- Bottom Parameter Allocation Table in Panel (b) (x = 15.75 to 30.55, y = 0.22 to 2.98) ---
\draw[zonebox] (15.75, 0.22) rectangle (30.55, 2.98);
\draw[headerbox] (15.75, 2.40) rectangle (30.55, 2.98);
\node[font=\footnotesize\bfseries] at (23.15, 2.69) {Observer Parameter Allocation \& Structural Identifiability Principle};

\draw[cardbox] (15.95, 0.38) rectangle (20.45, 2.22);
\node[align=center, font=\scriptsize] at (18.20, 1.30) {
    \textbf{Fixed-Parameter Filters}\\[3pt]
    EKF, AEKF, SMO:\\[1.5pt]
    Freeze nominal fresh priors\\[1.5pt]
    $(R_0, R_1, R_2, C_1, C_2, Q_{\mathrm{init}})$
};

\draw[cardbox] (20.80, 0.38) rectangle (25.50, 2.22);
\node[align=center, font=\scriptsize] at (23.15, 1.30) {
    \textbf{Dual EKF (DEKF)}\\[3pt]
    Adapts $\hat{\boldsymbol{\theta}}_k = [\hat{R}_{0,k},\, 1/\hat{Q}_{\mathrm{act},k}]^T$\\[1.5pt]
    Freezes $(R_1, R_2, C_1, C_2)$ to prevent\\[1.5pt]
    structural collinearity
};

\draw[cardbox, fill=gray!12, line width=0.85pt] (25.85, 0.38) rectangle (30.35, 2.22);
\node[align=center, font=\scriptsize] at (28.10, 1.30) {
    \textbf{Proposed Hybrid PIDKF}\\[3pt]
    DEKF tracks $(\hat{R}_{0,k}, \hat{Q}_{\mathrm{act},k})$ + MLP\\[1.5pt]
    compensates $\delta V_{\mathrm{pol},k}^*$ under\\[1.5pt]
    passivity $I_k\cdot\delta\hat{V}_{t,\mathrm{comp},k} \le 0$
};

\end{tikzpicture}
\end{document}
"""


# =============================================================================
# 2. TikZ LaTeX Source for Framework Figure (Complete Standalone Figure)
#    Canvas: 27.2 cm x 16.4 cm (1 unit = 1 cm), zero unjustified line wrapping
# =============================================================================
TIKZ_FRAMEWORK = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=28.4cm,paperheight=17.4cm,margin=0.4cm]{geometry}
\pagestyle{empty}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,backgrounds,fit}

\begin{document}\noindent
\begin{tikzpicture}[
    x=1cm, y=1cm,
    >=Latex,
    font=\small,
    text=black,
    draw=black,
    outerbox/.style={draw=black, line width=1.0pt, rounded corners=3pt, fill=white},
    colbox/.style={draw=black, line width=0.75pt, rounded corners=2pt, fill=gray!4},
    colhdr/.style={draw=black, line width=0.75pt, fill=gray!16, font=\footnotesize\bfseries, align=center},
    subcard/.style={draw=black, line width=0.65pt, rounded corners=2pt, fill=white, align=center},
    phasecard/.style={draw=black, line width=0.65pt, rounded corners=1.5pt, fill=white, align=center},
    phasehdr/.style={draw=black, line width=0.65pt, fill=gray!15, font=\scriptsize\bfseries, align=center},
    arr/.style={-{Latex[length=2.2mm,width=1.6mm]}, line width=0.9pt, draw=black}
]

% =========================================================================
% TOP SECTION (y = 6.60 to 16.00): Proposed Hybrid PIDKF Architecture
% =========================================================================
\draw[outerbox] (0, 6.60) rectangle (26.8, 16.00);
\node[anchor=west, font=\normalsize\bfseries] at (0.30, 15.55) {(a) Closed-Loop Hybrid Physics-Informed Dual Kalman Filter (PIDKF) with Butler--Volmer Passivity and Exogenous Decoupling};

% --- Column 1: Battery Plant & Telemetry (x = 0.30 to 5.40, y = 6.85 to 15.15) ---
\draw[colbox] (0.30, 6.85) rectangle (5.40, 15.15);
\draw[colhdr] (0.30, 14.55) rectangle (5.40, 15.15);
\node[font=\footnotesize\bfseries] at (2.85, 14.85) {1.\ Battery Plant \& Telemetry};

\draw[subcard] (0.50, 12.15) rectangle (5.20, 14.30);
\node[align=center, font=\scriptsize] at (2.85, 13.225) {
    \textbf{NMC811 \& LFP Cells}\\[3pt]
    Samsung INR18650-20R ($2.00\;\mathrm{Ah}$)\\[1.5pt]
    A123 ANR26650M1-B ($2.30\;\mathrm{Ah}$)\\[1.5pt]
    $\mathrm{SOH} \in [80\%, 100\%]$, $T \in [0^\circ\mathrm{C}, 45^\circ\mathrm{C}]$
};

\draw[subcard] (0.50, 9.45) rectangle (5.20, 11.65);
\node[align=center, font=\scriptsize] at (2.85, 10.55) {
    \textbf{Causal BMS Sensing ($1\;\mathrm{Hz}$)}\\[3pt]
    Current: $I_k = I_{\mathrm{true},k} + I_{\mathrm{bias}} + n_{i,k}$\\[1.5pt]
    Voltage: $V_{\mathrm{meas},k} = V_{t,k} + v_k$\\[1.5pt]
    Surface Temperature: $T_k$
};

\draw[subcard, fill=gray!10] (0.50, 7.10) rectangle (5.20, 8.95);
\node[align=center, font=\scriptsize] at (2.85, 8.025) {
    \textbf{Hard Rest-Period Innovation Gate}\\[2pt]
    If $|I_k| < 10\;\mathrm{mA}$: $\mathbf{K}_k = \mathbf{0},\;\mathbf{K}_{\theta,k} = \mathbf{0}$\\[1.5pt]
    ($\|\mathbf{F}_k\|_2 = \|\mathbf{A}_k\|_2 \le 1$, Frozen)
};

\draw[arr] (2.85, 12.15) -- (2.85, 11.65);
\draw[arr] (2.85, 9.45) -- (2.85, 8.95);

% --- Column 2: Dual EKF Backbone (x = 5.85 to 14.90, y = 6.85 to 15.15) ---
\draw[colbox] (5.85, 6.85) rectangle (14.90, 15.15);
\draw[colhdr] (5.85, 14.55) rectangle (14.90, 15.15);
\node[font=\footnotesize\bfseries] at (10.375, 14.85) {2.\ Dual Extended Kalman Filter Backbone ($980\;\mathrm{FLOPs/Step}$)};

\draw[subcard] (6.05, 11.35) rectangle (14.70, 14.30);
\node[align=center, font=\scriptsize] at (10.375, 12.825) {
    \textbf{Fast State Filter ($\tau_{\mathrm{state}} \sim 10^1\text{--}10^2\;\mathrm{s}$, Joseph Form)}\\[3.5pt]
    $\hat{\mathbf{x}}_k^- = \mathbf{A}_{k-1}\hat{\mathbf{x}}_{k-1}^+ + \mathbf{B}_{k-1}I_{k-1} + \mathbf{g}_{k-1}$\\[3pt]
    $\mathbf{P}_k^- = \mathbf{A}_{k-1}\mathbf{P}_{k-1}^+\mathbf{A}_{k-1}^T + \mathbf{Q}, \quad \hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k e_{k,\mathrm{PIDKF}}$\\[3pt]
    $\hat{V}_{t,\mathrm{ECM},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - (\hat{V}_{h,k}^-) - I_k\hat{R}_{0,k}^-$
};

\draw[subcard] (6.05, 7.10) rectangle (14.70, 10.45);
\node[align=center, font=\scriptsize] at (10.375, 8.775) {
    \textbf{Slow Parameter Filter ($\tau_{\mathrm{SOH}} \sim 10^5\text{--}10^6\;\mathrm{s}$, Random Walk)}\\[3.5pt]
    $\hat{\boldsymbol{\theta}}_k = [\hat{R}_{0,k},\; 1/\hat{Q}_{\mathrm{act},k}]^T, \quad \hat{\boldsymbol{\theta}}_k^+ = \hat{\boldsymbol{\theta}}_k^- + \mathbf{K}_{\theta,k} e_{k,\mathrm{PIDKF}}$\\[3pt]
    Fr\'echet Sensitivity (Eq.\ 17): $\mathbf{C}_{\theta,k} = \left[-I_k, \;\; -\dfrac{\eta\Delta t I_{k-1}}{3600}V_{\mathrm{ocv}}^\prime(\hat{z}_k^+)\right]$\\[3.5pt]
    Thermal-Inverted Health: $\mathrm{SOH}_{\mathrm{ref},k} \equiv \dfrac{\hat{Q}_{\mathrm{act},k}(T_k)}{Q_{\mathrm{init}}\,f_Q(T_k)} = \dfrac{1}{Q_{\mathrm{init}}\,\hat{\theta}_{2,k}\,f_Q(T_k)}$
};

\draw[arr] (9.35, 11.35) -- (9.35, 10.45);
\draw[arr] (11.40, 10.45) -- (11.40, 11.35);
\node[font=\scriptsize\bfseries] at (10.375, 10.90) {$\hat{z}_k^+ \;\leftrightarrow\; \hat{\boldsymbol{\theta}}_k^+$};

% --- Column 3: Physics-Constrained Neural Compensator (x = 16.00 to 26.55, y = 6.85 to 15.15) ---
\draw[colbox] (16.00, 6.85) rectangle (26.55, 15.15);
\draw[colhdr] (16.00, 14.55) rectangle (26.55, 15.15);
\node[font=\footnotesize\bfseries] at (21.275, 14.85) {3.\ Physics-Constrained Neural Compensator \& Innovation Update};

\draw[subcard] (16.20, 11.35) rectangle (21.95, 14.30);
\node[align=center, font=\scriptsize] at (19.075, 12.825) {
    \textbf{Compact Residual MLP ($560\;\mathrm{FLOPs/Step}$)}\\[3pt]
    Causal Input $\mathbf{u}_k \in \mathbb{R}^6$:\\[1.5pt]
    $[I_k,\, T_k,\, \hat{z}_k^-,\, \hat{U}_{1,k}^-,\, \hat{U}_{2,k}^-,\, \mathrm{SOH}_{\mathrm{ref},k}]^T$\\[2.5pt]
    Architecture: $6 \to 16 \to 8 \to 1$ (GELU)\\[1.5pt]
    $257\;\text{Params} = 1.03\;\mathrm{KB}$ Flash (No Leakage)
};

\draw[subcard, fill=gray!10] (22.40, 11.35) rectangle (26.35, 14.30);
\node[align=center, font=\scriptsize] at (24.375, 12.825) {
    \textbf{Butler--Volmer Passivity}\\[3pt]
    Hard Envelope Clipping:\\[1.5pt]
    $|\delta\hat{V}_{t,\mathrm{comp},k}| \le V_{\mathrm{pol}}^{\max}(I_k)$\\[1.5pt]
    ($\le 20.0\;\mathrm{mV}$ Bound)\\[2.5pt]
    Dissipative Constraint:\\[1.5pt]
    $I_k \cdot \delta\hat{V}_{t,\mathrm{comp},k} \le 0$
};

\draw[arr] (21.95, 12.825) -- (22.40, 12.825);

\draw[subcard] (16.20, 7.10) rectangle (26.35, 10.75);
\node[align=center, font=\scriptsize] at (21.275, 8.925) {
    \textbf{Compensated Innovation Update \& Exogenous Sensitivity Decoupling}\\[4pt]
    $\hat{V}_{t,\mathrm{PIDKF},k}^- = \hat{V}_{t,\mathrm{ECM},k}^- + \delta\hat{V}_{t,\mathrm{comp},k}, \quad e_{k,\mathrm{PIDKF}} = V_{\mathrm{meas},k} - \hat{V}_{t,\mathrm{PIDKF},k}^-$\\[3.5pt]
    $R_{\mathrm{eff},k} = R_v + \sigma_{\mathrm{NN}}^2 + \sigma_{\mathrm{approx}}^2 = 1.04\times 10^{-3}\;\mathrm{V}^2 \;\;(\sigma_{\mathrm{NN}}=5.8,\;\sigma_{\mathrm{approx}}=1.9\;\mathrm{mV})$\\[3.5pt]
    \textbf{Exogenous}: $\mathbf{H}_{\mathrm{PIDKF},k} \approx \mathbf{H}_k,\; \mathbf{C}_{\theta,\mathrm{PIDKF},k} \approx \mathbf{C}_{\theta,k}$ \textbf{($-37.3\%$ MCU Cycles)}\\[3.5pt]
    \textbf{Mean-Square UUB}: $\limsup_{k\to\infty}\mathbb{E}[\|\mathbf{e}_k^-\|^2] \le n\bar{p} + \left(\frac{c\bar{A}\bar{K}\bar{\varepsilon}}{1-\rho}\right)^2 < \infty$ \textbf{(Eqs.\ 24--25)}
};

% Inter-column arrows in Top Section
\draw[arr] (5.20, 13.225) -- (6.05, 13.225);
\draw[arr] (5.20, 10.10) -- (6.05, 10.10);
\draw[arr] (14.70, 12.825) -- (16.20, 12.825);
\draw[arr] (24.375, 11.35) -- (24.375, 10.75);
\draw[arr] (16.20, 8.775) -- (14.70, 8.775);
\node[font=\scriptsize\bfseries, fill=white, inner sep=1pt] at (15.45, 9.15) {$e_{k,\mathrm{PIDKF}}$};

% =========================================================================
% BOTTOM SECTION (y = 0.00 to 6.25): Validation Protocol & 10-Phase Matrix
% =========================================================================
\draw[outerbox] (0, 0.00) rectangle (26.8, 6.25);
\node[anchor=west, font=\normalsize\bfseries] at (0.30, 5.82) {(b) Strict Cell-Batch Validation Protocol, 7-Algorithm Taxonomy, and 10-Phase Multi-Dimensional Evaluation};

% --- Top Row of Panel (b): 3 Summary Pillars (y = 3.85 to 5.45) ---
\draw[subcard, fill=gray!6] (0.30, 3.85) rectangle (8.70, 5.45);
\node[align=center, font=\scriptsize] at (4.50, 4.65) {
    \textbf{Empirical Datasets \& Zero-Leakage Split}\\[2.5pt]
    CALCE (INR18650-20R, ANR26650), Oxford EIS, NASA PCoE\\[1.5pt]
    Batch A (Train / Calibrate) $\perp$ Batch B (Unseen Hold-Out Test)\\[1.5pt]
    Arbin BT-2000 Reference Uncertainty: $\sigma_{\mathrm{SOC,ref}} \le \pm 0.036\%$
};

\draw[subcard, fill=gray!6] (9.20, 3.85) rectangle (17.60, 5.45);
\node[align=center, font=\scriptsize] at (13.40, 4.65) {
    \textbf{7 Evaluated State-of-Charge Estimation Paradigms}\\[2.5pt]
    1.\ Model-Based: EKF ($3.87\,\mu\mathrm{s}$), UKF ($12.42\,\mu\mathrm{s}$), SMO ($1.67\,\mu\mathrm{s}$)\\[1.5pt]
    2.\ Adaptive/Dual: AEKF ($5.26\,\mu\mathrm{s}$), DEKF ($8.69\,\mu\mathrm{s}$)\\[1.5pt]
    3.\ ML/Hybrid: BiLSTM ($2.45\,\mathrm{ms}$), PINN ($130\,\mu\mathrm{s}$), PIDKF ($13.65\,\mu\mathrm{s}$)
};

\draw[subcard, fill=gray!6] (18.10, 3.85) rectangle (26.50, 5.45);
\node[align=center, font=\scriptsize] at (22.30, 4.65) {
    \textbf{Bare-Metal ARM Cortex-M4F Hardware Profiling}\\[2.5pt]
    STM32F407VG ($168\;\mathrm{MHz}$, IEEE-754 FPU, \texttt{DWT->CYCCNT})\\[1.5pt]
    Zero Heap Allocation (MISRA-C:2012 Compliant C99)\\[1.5pt]
    PIDKF: $1,540\;\mathrm{FLOPs/Step}$, $8.2\;\mathrm{KB}$ Flash, $1.1\;\mathrm{KB}$ SRAM
};

\draw[arr] (8.70, 4.65) -- (9.20, 4.65);
\draw[arr] (17.60, 4.65) -- (18.10, 4.65);

% --- Bottom Grid of Panel (b): 10 Experimental Phases (2 rows x 5 columns) ---
% Phase 1
\draw[phasecard] (0.30, 2.05) rectangle (5.30, 3.55);
\draw[phasehdr] (0.30, 3.12) rectangle (5.30, 3.55);
\node[font=\scriptsize\bfseries] at (2.80, 3.335) {Phase 1: Dynamic Cycles};
\node[align=center, font=\scriptsize] at (2.80, 2.58) {
    UDDS, US06, FUDS, DST\\
    Fresh Cell ($100\%$ SOH, $25^\circ\mathrm{C}$)\\
    EKF: $0.063\%$ $\mid$ PIDKF: $0.185\%$
};

% Phase 2
\draw[phasecard] (5.60, 2.05) rectangle (10.60, 3.55);
\draw[phasehdr] (5.60, 3.12) rectangle (10.60, 3.55);
\node[font=\scriptsize\bfseries] at (8.10, 3.335) {Phase 2: Multi-Stage Aging};
\node[align=center, font=\scriptsize] at (8.10, 2.58) {
    $\mathrm{SOH}\;100\% \to 95\% \to 90\% \to 80\%$\\
    EKF Surge: $45.7\times$ ($2.973\%$)\\
    DEKF: $0.585\%$ $\mid$ PIDKF: $0.422\%$
};

% Phase 3
\draw[phasecard] (10.90, 2.05) rectangle (15.90, 3.55);
\draw[phasehdr] (10.90, 3.12) rectangle (15.90, 3.55);
\node[font=\scriptsize\bfseries] at (13.40, 3.335) {Phase 3: Thermal Regimes};
\node[align=center, font=\scriptsize] at (13.40, 2.58) {
    Cold ($0^\circ\mathrm{C}, 10^\circ\mathrm{C}$) \& Hot ($45^\circ\mathrm{C}$)\\
    Arrhenius Surge ($2.85\times$ at $0^\circ\mathrm{C}$)\\
    DEKF Maintains $0.283\%$ at $0^\circ\mathrm{C}$
};

% Phase 4
\draw[phasecard] (16.20, 2.05) rectangle (21.20, 3.55);
\draw[phasehdr] (16.20, 3.12) rectangle (21.20, 3.55);
\node[font=\scriptsize\bfseries] at (18.70, 3.335) {Phase 4: Convergence};
\node[align=center, font=\scriptsize] at (18.70, 2.58) {
    $\pm 20\%$ \& $-40\%$ Initial Offset\\
    Closed-Loop Riccati Contraction\\
    Kalman $<35\;\mathrm{s}$ vs.\ ML Failure
};

% Phase 5
\draw[phasecard] (21.50, 2.05) rectangle (26.50, 3.55);
\draw[phasehdr] (21.50, 3.12) rectangle (26.50, 3.55);
\node[font=\scriptsize\bfseries] at (24.00, 3.335) {Phase 5: Radar Trade-Off};
\node[align=center, font=\scriptsize] at (24.00, 2.58) {
    5-Axis Holistic Benchmark:\\
    Accuracy, Aging, Thermal,\\
    Convergence \& MCU Efficiency
};

% Phase 6
\draw[phasecard] (0.30, 0.25) rectangle (5.30, 1.75);
\draw[phasehdr] (0.30, 1.32) rectangle (5.30, 1.75);
\node[font=\scriptsize\bfseries] at (2.80, 1.535) {Phase 6: LFP Chemistry};
\node[align=center, font=\scriptsize] at (2.80, 0.78) {
    $20\%\text{--}80\%$ Flat Plateau + $V_h$\\
    Gramian $\lambda_{\min}(\mathcal{G}_o) \ge 1.21\times 10^{-3}$\\
    PIDKF Aged LFP RMSE: $0.151\%$
};

% Phase 7
\draw[phasecard] (5.60, 0.25) rectangle (10.60, 1.75);
\draw[phasehdr] (5.60, 1.32) rectangle (10.60, 1.75);
\node[font=\scriptsize\bfseries] at (8.10, 1.535) {Phase 7: Sensor Faults};
\node[align=center, font=\scriptsize] at (8.10, 0.78) {
    $+250\;\mathrm{mA}$ Hall Bias \& $25\;\mathrm{mV}$ EMI\\
    $+15\;\mathrm{mV}$ Voltage Offset Drift\\
    SMO Manifold Clamp: $0.629\%$
};

% Phase 8
\draw[phasecard] (10.90, 0.25) rectangle (15.90, 1.75);
\draw[phasehdr] (10.90, 1.32) rectangle (15.90, 1.75);
\node[font=\scriptsize\bfseries] at (13.40, 1.535) {Phase 8: EIS Diagnostics};
\node[align=center, font=\scriptsize] at (13.40, 0.78) {
    $0.01\;\mathrm{Hz}\text{--}10\;\mathrm{kHz}$ Nyquist Spectra\\
    Charge-Transfer $R_{\mathrm{ct}}$ vs.\ SOH\\
    Linear Anchor $R^2 = 0.994$
};

% Phase 9
\draw[phasecard] (16.20, 0.25) rectangle (21.20, 1.75);
\draw[phasehdr] (16.20, 1.32) rectangle (21.20, 1.75);
\node[font=\scriptsize\bfseries] at (18.70, 1.535) {Phase 9: 8S Pack Scaling};
\node[align=center, font=\scriptsize] at (18.70, 0.78) {
    Dispersion ($\pm 4\%\;Q$, $\pm 12\%\;R_0$)\\
    Core Thermal Gradient ($+4.8^\circ\mathrm{C}$)\\
    Weakest Cell \#4 RMSE: $0.337\%$
};

% Phase 10
\draw[phasecard, fill=gray!10, line width=0.85pt] (21.50, 0.25) rectangle (26.50, 1.75);
\draw[phasehdr, fill=gray!22, line width=0.85pt] (21.50, 1.32) rectangle (26.50, 1.75);
\node[font=\scriptsize\bfseries] at (24.00, 1.535) {Phase 10: Hybrid PIDKF};
\node[align=center, font=\scriptsize] at (24.00, 0.78) {
    Ablation \& Cortex-M4 Pareto\\
    $-8.7\%$ (NMC) \& $-51.0\%$ (LFP)\\
    Error Reduction at $13.65\;\mu\mathrm{s/step}$
};

\end{tikzpicture}
\end{document}
"""


def compile_tikz_to_png(tex_str, basename, work_dir="results", dpi=600):
    os.makedirs(work_dir, exist_ok=True)
    tex_path = os.path.join(work_dir, f"{basename}.tex")
    pdf_path = os.path.join(work_dir, f"{basename}.pdf")
    png_path = os.path.join(work_dir, f"{basename}.png")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_str)

    cmd = [PDFLATEX, "-interaction=nonstopmode", f"{basename}.tex"]
    res = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(pdf_path):
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise RuntimeError(f"Failed to compile {tex_path}")

    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(png_path)
    doc.close()

    with Image.open(png_path) as src_im:
        im = src_im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        pad = 24
        x0 = max(0, bbox[0] - pad)
        y0 = max(0, bbox[1] - pad)
        x1 = min(im.size[0], bbox[2] + pad)
        y1 = min(im.size[1], bbox[3] + pad)
        cropped = im.crop((x0, y0, x1, y1))
        cropped.save(png_path, dpi=(dpi, dpi))
        cropped.close()
    im.close()

    for ext in [".tex", ".aux", ".log"]:
        p = os.path.join(work_dir, f"{basename}{ext}")
        if os.path.exists(p):
            os.remove(p)

    return pdf_path, png_path


def build_combined_mechanism_figure(top_png_path, save_paths):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    top_img = plt.imread(top_png_path)

    fig = plt.figure(figsize=(15.5, 9.5), facecolor="white")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.18, 1.0], width_ratios=[1.0, 1.0],
                           hspace=0.16, wspace=0.18, left=0.055, right=0.985, top=0.985, bottom=0.065)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(top_img, interpolation="lanczos")
    ax_top.axis("off")
    pos_top = ax_top.get_position()
    ax_top.set_position([0.012, pos_top.y0, 0.976, pos_top.height])

    # PANEL (c): OCV-SOC Curve & Hysteresis Loop
    ax_c = fig.add_subplot(gs[1, 0])
    z = np.linspace(0.0, 1.0, 500)
    vocv_nmc = 9.204 * z**5 - 28.841 * z**4 + 33.376 * z**3 - 17.419 * z**2 + 4.674 * z + 3.212
    z_lfp = np.linspace(0.12, 0.95, 400)
    vocv_lfp = 3.415 - 0.286 * z_lfp + 1.831 * z_lfp**2 - 4.120 * z_lfp**3 + 4.185 * z_lfp**4 - 1.545 * z_lfp**5
    M_hyst = 0.030

    ax_c.plot(z * 100, vocv_nmc, color="#1d4ed8", lw=2.2,
              label=r"NMC811 $V_{\mathrm{ocv}}(z)$ (Eq. 11, Monotonic 5th-Order)")
    ax_c.plot(z_lfp * 100, vocv_lfp, color="#15803d", lw=2.2, ls="--",
              label=r"$\mathrm{LiFePO_4}$ Equilibrium $V_{\mathrm{ocv}}^{\mathrm{LFP}}(z)$")
    ax_c.fill_between(z_lfp * 100, vocv_lfp - M_hyst, vocv_lfp + M_hyst,
                      color="#86efac", alpha=0.40, edgecolor="#15803d", lw=0.6,
                      label=r"LFP Hysteresis Envelope $\pm M_0$ ($\pm 30\;\mathrm{mV}$)")

    ax_c.axvspan(20, 80, color="#f1f5f9", alpha=0.75, zorder=0)
    ax_c.axvline(20, color="#64748b", ls=":", lw=0.9)
    ax_c.axvline(80, color="#64748b", ls=":", lw=0.9)

    ax_c.annotate("Biphasic Plateau (20%–80% SOC)\n" +
                  r"$\mathrm{LiFePO_4 \leftrightarrow FePO_4}$ ($F' = C - P = 0$)" + "\n" +
                  r"Flat Slope $\approx 0.08\;\mathrm{V}$ & $\pm 30\;\mathrm{mV}$ Hysteresis",
                  xy=(50, 3.315), xytext=(45, 3.63),
                  fontsize=8.6, fontweight="bold", color="black",
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))

    ax_c.set_title("(c) Calibrated Open-Circuit Voltage Curves and LFP Hysteresis Loop",
                   fontsize=10.8, fontweight="bold", color="black", pad=7)
    ax_c.set_xlabel("State of Charge, SOC (%)", fontsize=10.0, fontweight="bold", color="black")
    ax_c.set_ylabel(r"Open-Circuit Potential, $V_{\mathrm{ocv}}$ (V)", fontsize=10.0, fontweight="bold", color="black")
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(3.05, 4.30)
    ax_c.grid(True, linestyle=":", color="#94a3b8", alpha=0.6)
    leg_c = ax_c.legend(loc="upper left", fontsize=8.3, frameon=True, facecolor="white", edgecolor="black")
    for text in leg_c.get_texts():
        text.set_color("black")

    # PANEL (d): OCV Slope dVocv/dSOC & Observability Condition Number
    ax_d = fig.add_subplot(gs[1, 1])
    dvocv_nmc = 46.020 * z**4 - 115.364 * z**3 + 100.128 * z**2 - 34.838 * z + 4.674
    dvocv_lfp = -0.286 + 3.662 * z_lfp - 12.36 * z_lfp**2 + 16.74 * z_lfp**3 - 7.725 * z_lfp**4

    ax_d.semilogy(z * 100, dvocv_nmc, color="#1d4ed8", lw=2.2,
                  label=r"NMC Slope $V_{\mathrm{ocv}}'(z) \in [0.481, 4.674]\;\mathrm{V}$")
    ax_d.semilogy(z_lfp * 100, dvocv_lfp, color="#b91c1c", lw=2.2, ls="-.",
                  label=r"LFP Slope $V_{\mathrm{ocv}}'(z)$ ($0.065\;\mathrm{to}\;0.08\;\mathrm{V}$ on Plateau)")

    ax_d.axvspan(20, 80, color="#f1f5f9", alpha=0.75, zorder=0)
    ax_d.axvline(20, color="#64748b", ls=":", lw=0.9)
    ax_d.axvline(80, color="#64748b", ls=":", lw=0.9)
    ax_d.axhline(0.08, color="black", ls=":", lw=1.0)

    ax_d.text(50, 0.22, r"LFP Plateau Observability Bottleneck ($V_{\mathrm{ocv}}' \approx 0.08\;\mathrm{V}$)",
              fontsize=8.4, fontweight="bold", color="black", ha="center",
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="black", lw=0.8))

    ax_d.text(50, 0.0105,
              r"Observability Determinant: $|\det(\mathcal{O}_k)| \propto V_{\mathrm{ocv}}'(z_k)$ (Eqs. 12, 16)" + "\n" +
              r"$\bullet$ NMC: $\kappa(\mathcal{O}_k) \approx 3.2 \times 10^4 \;\Rightarrow\;$ Strong Innovation Gain" + "\n" +
              r"$\bullet$ LFP Plateau: $\kappa(\mathcal{O}_k) > 1.99 \times 10^5 \;\Rightarrow\;$ Gain Vanishes / SMO Chattering",
              ha="center", va="center", fontsize=8.0, color="black",
              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8))

    ax_d.set_title(r"(d) Analytical Observability Gradient $dV_{\mathrm{ocv}}/d\mathrm{SOC}$ and Condition Surge",
                   fontsize=10.8, fontweight="bold", color="black", pad=7)
    ax_d.set_xlabel("State of Charge, SOC (%)", fontsize=10.0, fontweight="bold", color="black")
    ax_d.set_ylabel(r"OCV Derivative, $dV_{\mathrm{ocv}}/d\mathrm{SOC}$ (V) [Log Scale]", fontsize=9.8, fontweight="bold", color="black")
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0.003, 8.0)
    ax_d.grid(True, which="both", linestyle=":", color="#94a3b8", alpha=0.6)
    leg_d = ax_d.legend(loc="upper right", fontsize=8.3, frameon=True, facecolor="white", edgecolor="black")
    for text in leg_d.get_texts():
        text.set_color("black")

    for sp in save_paths:
        fig.savefig(sp, dpi=600)
        print(f"Saved 600-DPI {sp}")
    plt.close(fig)


if __name__ == "__main__":
    folders = ["results", "manuscript", "submission_journal_of_energy_storage"]

    print("1. Compiling TikZ top panels for Mechanism figure...")
    top_pdf, top_png = compile_tikz_to_png(TIKZ_MECH_TOP, "temp_mech_top", work_dir="results", dpi=600)
    mech_paths = [os.path.join(f, "fig0_electrochemical_mechanism.png") for f in folders]
    build_combined_mechanism_figure(top_png, mech_paths)
    for p in [top_pdf, top_png]:
        if os.path.exists(p):
            os.remove(p)

    print("2. Compiling TikZ Framework figure...")
    fw_pdf, fw_png = compile_tikz_to_png(TIKZ_FRAMEWORK, "fig0_pidkf_framework", work_dir="results", dpi=600)
    for f in ["manuscript", "submission_journal_of_energy_storage"]:
        shutil.copy2(fw_png, os.path.join(f, "fig0_pidkf_framework.png"))
        print(f"Copied 600-DPI fig0_pidkf_framework.png to {f}/")
    if os.path.exists(fw_pdf):
        os.remove(fw_pdf)
