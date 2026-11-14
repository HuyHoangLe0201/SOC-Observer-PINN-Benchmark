r"""
Generate publication-grade pure-vector (PDF) and 600-DPI (PNG) conceptual figures
using unified LaTeX TikZ + PGFPlots (100% Computer Modern typography, strictly
black text everywhere, harmonious scientific pastel fills, zero overlapping nodes):

1. fig0_electrochemical_mechanism (.pdf & .png)
2. fig0_pidkf_framework (.pdf & .png)
"""

import os
import shutil
import subprocess
import fitz  # PyMuPDF
from PIL import Image, ImageChops

TEXLIVE_BIN = r"d:\texlive\2026\bin\windows"
PDFLATEX = os.path.join(TEXLIVE_BIN, "pdflatex.exe")

# =============================================================================
# FIGURE 1: ELECTROCHEMICAL DEGRADATION MECHANISM, ECM & OBSERVABILITY
# Unified 32.0 cm x 20.6 cm TikZ + PGFPlots vector figure
# =============================================================================
TIKZ_MECHANISM = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=34.2cm,paperheight=22.8cm,margin=0.8cm]{geometry}
\pagestyle{empty}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,backgrounds,fit,decorations.pathmorphing,shadows}

% Harmonious Q1 Journal Pastel Palette (All typography remains strictly pure black)
\definecolor{cufoil}{RGB}{244,205,165}
\definecolor{alfoil}{RGB}{216,223,232}
\definecolor{anodebg}{RGB}{244,247,251}
\definecolor{graphcore}{RGB}{210,220,232}
\definecolor{seifill}{RGB}{254,238,195}
\definecolor{dendritefill}{RGB}{185,196,210}
\definecolor{elytebg}{RGB}{240,249,255}
\definecolor{sepfill}{RGB}{226,238,248}
\definecolor{nmcbg}{RGB}{238,244,255}
\definecolor{nmcpart}{RGB}{199,218,252}
\definecolor{lfpbg}{RGB}{236,253,245}
\definecolor{lfpshell}{RGB}{176,240,210}
\definecolor{fepcore}{RGB}{254,230,224}
\definecolor{thermbg}{RGB}{254,242,242}
\definecolor{arrhbg}{RGB}{255,250,235}
\definecolor{capbg}{RGB}{236,253,245}
\definecolor{hdrslate}{RGB}{226,232,240}
\definecolor{hdrblue}{RGB}{219,234,254}
\definecolor{hdrgreen}{RGB}{209,250,229}
\definecolor{hdramber}{RGB}{254,243,199}
\definecolor{hdrrose}{RGB}{254,226,226}
\definecolor{nmcline}{RGB}{29,78,216}
\definecolor{lfpline}{RGB}{21,128,61}
\definecolor{lfpderiv}{RGB}{185,28,28}
\definecolor{platshade}{RGB}{241,245,249}

\begin{document}\noindent
\begin{tikzpicture}[
    x=1cm, y=1cm,
    >=Latex,
    text=black,
    draw=black,
    font=\normalsize,
    panelbox/.style={draw=black, line width=1.1pt, rounded corners=4pt, fill=white},
    card/.style={draw=black, line width=0.75pt, rounded corners=2.5pt, fill=white, drop shadow={opacity=0.08, shadow xshift=1pt, shadow yshift=-1pt}},
    arr/.style={-{Latex[length=2.4mm,width=1.7mm]}, line width=0.95pt, draw=black},
    darr/.style={-{Latex[length=2.2mm,width=1.6mm]}, line width=0.85pt, draw=black, densely dashed},
    resistor/.style={decorate, decoration={zigzag, segment length=2.5mm, amplitude=1.15mm}, line width=1.05pt}
]

% =========================================================================
% PANEL (a): Top-Left (x = 0.0 to 15.7, y = 9.9 to 20.6)
% =========================================================================
\draw[panelbox] (0.0, 9.9) rectangle (15.7, 20.6);
\node[anchor=west, font=\fontsize{11.5}{13}\selectfont\bfseries] at (0.28, 20.16)
    {(a) Multi-Scale Electrode Degradation \& Electro-Thermal-Aging Loop};

% Outer Cell Cross-Section Frame: y = 13.45 to 19.62
\draw[draw=black, line width=0.8pt, rounded corners=2.5pt, fill=white] (0.25, 13.45) rectangle (15.45, 19.62);

% Column Header Strip for Cell Zones (y = 19.06 to 19.62)
\filldraw[fill=hdrslate, draw=black, line width=0.75pt] (0.25, 19.06) rectangle (4.95, 19.62);
\node[font=\fontsize{9.2}{10.5}\selectfont\bfseries] at (2.60, 19.34) {Cu $|$ Anode ($\mathrm{Li_xC_6}$)};

\filldraw[fill=hdrblue, draw=black, line width=0.75pt] (4.95, 19.06) rectangle (9.55, 19.62);
\node[font=\fontsize{9.2}{10.5}\selectfont\bfseries] at (7.25, 19.34) {Electrolyte \& Separator};

\filldraw[fill=hdrgreen, draw=black, line width=0.75pt] (9.55, 19.06) rectangle (15.45, 19.62);
\node[font=\fontsize{9.2}{10.5}\selectfont\bfseries] at (12.50, 19.34) {Cathode (NMC / LFP) $|$ Al};

% --- 1. Cu Current Collector (x = 0.25 to 0.72) ---
\filldraw[fill=cufoil, draw=black, line width=0.7pt] (0.25, 13.45) rectangle (0.72, 19.06);
\node[rotate=90, font=\fontsize{9.2}{10.5}\selectfont\bfseries] at (0.485, 16.255) {Cu Foil ($-$)};

% --- 2. Graphite Anode Zone (x = 0.72 to 4.95) ---
\filldraw[fill=anodebg, draw=black, line width=0.7pt] (0.72, 13.45) rectangle (4.95, 19.06);

% Top Active Particle with SEI Shell & Graphene Planes
\filldraw[fill=seifill, draw=black, line width=0.8pt, densely dashed] (2.38, 17.32) circle (1.06cm);
\filldraw[fill=graphcore, draw=black, line width=0.9pt] (2.38, 17.32) circle (0.79cm);
\draw[line width=0.5pt] (1.76, 17.58) -- (3.00, 17.58);
\draw[line width=0.5pt] (1.64, 17.32) -- (3.12, 17.32);
\draw[line width=0.5pt] (1.76, 17.06) -- (3.00, 17.06);
\node[fill=white, fill opacity=0.90, text opacity=1, inner sep=1.5pt, rounded corners=1.5pt, align=center, font=\fontsize{8.2}{9.5}\selectfont\bfseries]
    at (2.38, 17.32) {$\mathrm{Li_xC_6}$\\$D_s(T)$};
\node[fill=white, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.8pt, font=\fontsize{8}{9.2}\selectfont\bfseries]
    at (2.38, 18.68) {Porous SEI ($\delta_{\mathrm{SEI}}$)};

% SEI Side Effects Pill
\node[draw=black, line width=0.55pt, fill=white, rounded corners=2pt, align=left, font=\fontsize{7.5}{9}\selectfont, inner sep=2.4pt]
    at (2.84, 15.58) {$\bullet$ Traps $\mathrm{Li^+}$ (LLI) \quad $\bullet$ $\uparrow\Delta R_{\mathrm{SEI}}$\\[-1pt]$\bullet$ Capacity \& impedance drift};

% Bottom Active Particle with Cold Li-Plating Dendrites
\filldraw[fill=seifill, draw=black, line width=0.75pt, densely dashed] (1.74, 14.24) circle (0.60cm);
\filldraw[fill=graphcore, draw=black, line width=0.85pt] (1.74, 14.24) circle (0.45cm);
\node[font=\fontsize{7.8}{8.8}\selectfont\bfseries] at (1.74, 14.24) {$\mathrm{Li_xC_6}$};
\filldraw[fill=dendritefill, draw=black, line width=0.65pt]
    (2.28, 14.48) -- (2.68, 14.66) -- (2.54, 14.46) -- (2.88, 14.52) -- (2.56, 14.30) -- (2.78, 14.14) -- (2.34, 14.18) -- cycle;
\node[draw=black, line width=0.5pt, fill=hdrrose, rounded corners=1.5pt, inner sep=2pt, align=center, font=\fontsize{7.6}{9}\selectfont\bfseries]
    at (3.86, 14.32) {Cold $\mathrm{Li}^0$\\Plating ($0^\circ\mathrm{C}$)};

% --- 3. Electrolyte & Separator Zone (x = 4.95 to 9.55) ---
\filldraw[fill=elytebg, draw=black, line width=0.7pt] (4.95, 13.45) rectangle (9.55, 19.06);

% Porous Polymeric Separator Column (x = 7.06 to 7.44)
\filldraw[fill=sepfill, draw=black, line width=0.65pt, densely dashed] (7.06, 15.62) rectangle (7.44, 18.92);
\foreach \yy in {15.80, 16.25, 16.70, 17.15, 17.60, 18.05, 18.50} {
    \draw[line width=0.6pt, fill=white] (7.14, \yy) rectangle (7.36, \yy+0.20);
}
\node[fill=white, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.6pt, font=\fontsize{7.8}{9}\selectfont\bfseries]
    at (7.25, 18.68) {Separator};

% Solvated Li+ Cluster Icon on left -> Desolvated Li+ crossing separator
\filldraw[fill=nmcpart, draw=black, line width=0.7pt] (5.58, 17.56) circle (0.18cm);
\node[font=\fontsize{6.8}{7.8}\selectfont\bfseries] at (5.58, 17.56) {$\mathrm{Li^+}$};
\foreach \ang in {35, 145, 215, 325} {
    \filldraw[fill=hdramber, draw=black, line width=0.5pt] ($(5.58,17.56)+(\ang:0.29cm)$) circle (0.08cm);
}
\node[font=\fontsize{7.5}{8.5}\selectfont\bfseries] at (5.75, 18.14) {$\mathrm{Li^+(EC)_4}$};

% Discharge & Charge arrows
\draw[arr, line width=1.05pt] (5.96, 17.56) -- (9.18, 17.56);
\filldraw[fill=hdrgreen, draw=black, line width=0.7pt] (8.55, 17.56) circle (0.18cm);
\node[font=\fontsize{6.8}{7.8}\selectfont\bfseries] at (8.55, 17.56) {$\mathrm{Li^+}$};
\node[fill=white, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.5pt, font=\fontsize{7.4}{8.5}\selectfont\bfseries]
    at (7.80, 18.06) {Discharge ($I > 0$)};

\draw[darr, line width=0.95pt] (9.18, 16.58) -- (5.32, 16.58);
\node[fill=white, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.5pt, font=\fontsize{7.5}{8.5}\selectfont]
    at (7.25, 16.10) {Charge ($I < 0$)};

% Desolvation + Butler-Volmer Card inside Electrolyte
\draw[card] (5.10, 13.60) rectangle (9.40, 15.52);
\node[align=center, font=\fontsize{7.2}{8.8}\selectfont] at (7.25, 14.56) {
    \textbf{Butler--Volmer Kinetics}\\[2pt]
    $\eta_{\mathrm{act}} = \dfrac{2RT}{F}\mathrm{arcsinh}\!\left(\dfrac{I}{2 A i_0}\right)$\\[3pt]
    Desolvation:\\[-1pt]$E_{a,1}^{\mathrm{cold}} = 28.4\;\mathrm{kJ/mol}$
};

% --- 4. Dual Cathode Comparison (x = 9.55 to 14.98) + Al Collector ---
\filldraw[fill=nmcbg, draw=black, line width=0.7pt] (9.55, 16.25) rectangle (14.98, 19.06);
\filldraw[fill=nmcpart, draw=black, line width=0.85pt] (10.22, 17.65) circle (0.50cm);
\foreach \dy in {-0.26, -0.13, 0.0, 0.13, 0.26} {
    \draw[line width=0.5pt, draw=nmcline] (9.84, 17.65+\dy) -- (10.60, 17.65+\dy);
}
\node[fill=white, fill opacity=0.90, text opacity=1, inner sep=1pt, rounded corners=1pt, font=\fontsize{6.6}{7.5}\selectfont\bfseries]
    at (10.22, 17.65) {1-Phase};
\node[anchor=west, align=left, font=\fontsize{7.6}{9.2}\selectfont] at (10.86, 17.65) {
    \textbf{NMC811} (\texttt{INR18650-20R})\\[1.5pt]
    Solid solution host\\[1pt]
    $V_{\mathrm{ocv}}^\prime(z)\!\in\![0.481, 4.674]\;\mathrm{V}$\\[1pt]
    Full rank ($\mathrm{rank}\,\mathcal{O}_k = 3$)
};

\filldraw[fill=lfpbg, draw=black, line width=0.7pt] (9.55, 13.45) rectangle (14.98, 16.25);
\filldraw[fill=lfpshell, draw=black, line width=0.85pt] (10.28, 15.05) circle (0.56cm);
\filldraw[fill=fepcore, draw=black, line width=0.75pt, densely dashed] (10.28, 15.05) circle (0.31cm);
\node[font=\fontsize{5.6}{6.6}\selectfont\bfseries] at (10.28, 15.05) {$\mathrm{FePO_4}$};
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{6.2}{7.2}\selectfont\bfseries]
    at (10.28, 14.12) {LFP Shell};
\node[anchor=west, align=left, font=\fontsize{7.4}{8.8}\selectfont] at (10.90, 14.85) {
    \textbf{$\mathrm{LiFePO_4}$} (\texttt{ANR26650M1-B})\\[1.5pt]
    Biphasic core-shell ($F'\!=\!0$)\\[1pt]
    $V_{\mathrm{ocv}}^\prime\!\approx\!0.08\;\mathrm{V} + \text{Hyst.}\;V_h$\\[1pt]
    Cond.\ $\kappa(\mathcal{O}_k) > 1.99\times 10^5$
};

% Al Current Collector
\filldraw[fill=alfoil, draw=black, line width=0.7pt] (14.98, 13.45) rectangle (15.45, 19.06);
\node[rotate=90, font=\fontsize{9.2}{10.5}\selectfont\bfseries] at (15.215, 16.255) {Al Foil ($+$)};

% --- Bottom Section of Panel (a): True Closed Electro-Thermal-Aging Feedback Loop ---
\draw[draw=black, line width=0.75pt, rounded corners=2.5pt, fill=anodebg] (0.25, 10.08) rectangle (15.45, 13.26);
\filldraw[fill=hdrslate, draw=black, line width=0.75pt] (0.25, 12.74) rectangle (15.45, 13.26);
\node[font=\fontsize{9.5}{11}\selectfont\bfseries] at (7.85, 13.00) {Closed Electro-Thermal-Aging Loop (Arrhenius Kinetics \& Energy Balance)};

\draw[card, fill=thermbg] (0.42, 10.68) rectangle (5.05, 12.56);
\node[align=center, font=\fontsize{7.8}{9.5}\selectfont] at (2.735, 11.62) {
    \textbf{Thermal Balance (Eq.\ 5)}\\[2pt]
    $C_{\mathrm{th}}\dfrac{dT}{dt} = I^2 R_0 + I(U_1+U_2)$\\[2pt]
    $- IT\dfrac{\partial V_{\mathrm{ocv}}}{\partial T} - \dfrac{T-T_{\mathrm{amb}}}{R_{\mathrm{th}}}$
};

\draw[card, fill=arrhbg] (5.55, 10.68) rectangle (10.15, 12.56);
\node[align=center, font=\fontsize{7.8}{9.5}\selectfont] at (7.85, 11.62) {
    \textbf{Arrhenius \& Aging (Eqs.\ 8--10)}\\[3pt]
    $R_i(T,\mathrm{SOH}) = R_{i,\mathrm{nom}}\,f_{T,i}(T)$\\[2.5pt]
    $\times\,[1 + \kappa_{R_i}(1-\mathrm{SOH})]$
};

\draw[card, fill=capbg] (10.65, 10.68) rectangle (15.28, 12.56);
\node[align=center, font=\fontsize{7.8}{9.5}\selectfont] at (12.965, 11.62) {
    \textbf{Usable Capacity (Eq.\ 10)}\\[3pt]
    $Q_{\mathrm{act}}(T,\mathrm{SOH}) = \mathrm{SOH}\cdot Q_{\mathrm{init}}\,f_Q(T)$\\[2.5pt]
    $0.890\times\;(0^\circ\mathrm{C}),\;\; 1.030\times\;(45^\circ\mathrm{C})$
};

\draw[arr] (5.05, 11.62) -- (5.55, 11.62);
\node[font=\fontsize{8}{9}\selectfont\bfseries] at (5.30, 11.90) {$T_k$};
\draw[arr] (10.15, 11.62) -- (10.65, 11.62);
\node[font=\fontsize{8}{9}\selectfont\bfseries] at (10.40, 11.90) {$R_i$};

\draw[arr, line width=0.9pt] (12.965, 10.68) -- (12.965, 10.28) -- (2.735, 10.28) -- (2.735, 10.68);
\draw[line width=0.9pt] (7.85, 10.68) -- (7.85, 10.28);
\node[fill=anodebg, inner sep=2pt, font=\fontsize{8}{9.5}\selectfont\bfseries] at (7.85, 10.28)
    {Joule Heat \& SEI Feedback ($I^2 R_0,\, \Delta R_{\mathrm{SEI}},\, \mathrm{SOH}$)};


% =========================================================================
% PANEL (b): Top-Right (x = 16.0 to 31.8, y = 9.9 to 20.6)
% =========================================================================
\draw[panelbox] (16.0, 9.9) rectangle (31.8, 20.6);
\node[anchor=west, font=\fontsize{11.5}{13}\selectfont\bfseries] at (16.28, 20.16)
    {(b) Augmented 2nd-Order RC Equivalent Circuit \& Parameter Allocation};

% --- Circuit Diagram Box (x = 16.25 to 31.55, y = 13.45 to 19.62) ---
\draw[draw=black, line width=0.8pt, rounded corners=2.5pt, fill=white] (16.25, 13.45) rectangle (31.55, 19.62);

\coordinate (bot_l) at (17.15, 14.68);
\coordinate (top_l) at (17.15, 17.75);

\draw[line width=1.0pt] (bot_l) -- (17.15, 15.18);
\filldraw[fill=hdrblue, draw=black, line width=1.0pt] (17.15, 15.72) circle (0.52cm);
\node[font=\fontsize{8.5}{10}\selectfont\bfseries] at (17.15, 15.72) {$V_{\mathrm{ocv}}(z)$};
\node[font=\fontsize{10}{11}\selectfont\bfseries] at (16.48, 16.10) {$+$};
\node[font=\fontsize{10}{11}\selectfont\bfseries] at (16.48, 15.32) {$-$};
\draw[line width=1.0pt] (17.15, 16.24) -- (17.15, 16.54);

\filldraw[fill=hdramber, draw=black, line width=0.9pt, rounded corners=2pt] (16.54, 16.54) rectangle (17.76, 17.28);
\node[align=center, font=\fontsize{8}{9.2}\selectfont\bfseries] at (17.15, 16.91) {$\pm V_{h,k}$\\[-2pt]\fontsize{6.8}{8}\selectfont (LFP)};
\draw[line width=1.0pt] (17.15, 17.28) -- (top_l);

% R0(T, SOH) + DEKF pill badge
\draw[line width=1.0pt] (top_l) -- (18.10, 17.75);
\draw[resistor] (18.10, 17.75) -- (19.55, 17.75);
\draw[line width=1.0pt] (19.55, 17.75) -- (20.15, 17.75);
\filldraw[fill=black] (20.15, 17.75) circle (1.8pt);
\node[font=\fontsize{8.5}{10}\selectfont\bfseries] at (18.825, 18.36) {$R_0(T,\mathrm{SOH})$};
\node[draw=black, line width=0.5pt, fill=hdrblue, rounded corners=2pt, inner sep=1.6pt, font=\fontsize{6.8}{8}\selectfont\bfseries]
    at (18.55, 18.92) {DEKF Tracks $\hat{R}_0$};

% RC1 parallel loop
\draw[line width=0.95pt] (20.15, 17.02) -- (20.15, 18.48);
\draw[line width=0.95pt] (23.00, 17.02) -- (23.00, 18.48);
\filldraw[fill=black] (23.00, 17.75) circle (1.8pt);
\draw[line width=0.95pt] (20.15, 18.48) -- (20.72, 18.48);
\draw[resistor] (20.72, 18.48) -- (22.42, 18.48);
\draw[line width=0.95pt] (22.42, 18.48) -- (23.00, 18.48);
\node[font=\fontsize{8.5}{10}\selectfont\bfseries] at (21.575, 19.04) {$R_1$ (Charge Transfer)};
\draw[line width=0.95pt] (20.15, 17.02) -- (21.43, 17.02);
\draw[line width=1.45pt] (21.43, 16.72) -- (21.43, 17.32);
\draw[line width=1.45pt] (21.72, 16.72) -- (21.72, 17.32);
\draw[line width=0.95pt] (21.72, 17.02) -- (23.00, 17.02);
\node[align=center, font=\fontsize{8.2}{9.8}\selectfont] at (21.575, 16.06) {$C_1$ ($\tau_1 = 24.0\;\mathrm{s}$)\\[1pt]$+$ \,$U_{1,k}$\, $-$};

\draw[line width=1.0pt] (23.00, 17.75) -- (23.70, 17.75);
\filldraw[fill=black] (23.70, 17.75) circle (1.8pt);

% RC2 parallel loop
\draw[line width=0.95pt] (23.70, 17.02) -- (23.70, 18.48);
\draw[line width=0.95pt] (26.55, 17.02) -- (26.55, 18.48);
\filldraw[fill=black] (26.55, 17.75) circle (1.8pt);
\draw[line width=0.95pt] (23.70, 18.48) -- (24.27, 18.48);
\draw[resistor] (24.27, 18.48) -- (25.97, 18.48);
\draw[line width=0.95pt] (25.97, 18.48) -- (26.55, 18.48);
\node[font=\fontsize{8.5}{10}\selectfont\bfseries] at (25.125, 19.04) {$R_2$ (Warburg Diffusion)};
\draw[line width=0.95pt] (23.70, 17.02) -- (24.98, 17.02);
\draw[line width=1.45pt] (24.98, 16.72) -- (24.98, 17.32);
\draw[line width=1.45pt] (25.27, 16.72) -- (25.27, 17.32);
\draw[line width=0.95pt] (25.27, 17.02) -- (26.55, 17.02);
\node[align=center, font=\fontsize{8.2}{9.8}\selectfont] at (25.125, 16.06) {$C_2$ ($\tau_2 = 112.5\;\mathrm{s}$)\\[1pt]$+$ \,$U_{2,k}$\, $-$};

% Active Neural Residual Compensator Element
\draw[line width=1.0pt] (26.55, 17.75) -- (27.15, 17.75);
\filldraw[fill=hdrgreen, draw=black, line width=1.0pt, rounded corners=2.5pt, drop shadow={opacity=0.08, shadow xshift=0.8pt, shadow yshift=-0.8pt}]
    (27.15, 17.12) rectangle (29.75, 18.38);
\node[align=center, font=\fontsize{8.2}{9.8}\selectfont\bfseries] at (28.45, 17.75) {$\delta\hat{V}_{t,\mathrm{comp},k}$\\(PIDKF MLP)};
\node[draw=black, line width=0.5pt, fill=white, rounded corners=2pt, inner sep=1.8pt, font=\fontsize{7}{8}\selectfont\bfseries]
    at (28.45, 18.92) {Passivity: $I_k\cdot\delta\hat{V} \le 0$};
\draw[line width=1.0pt] (29.75, 17.75) -- (30.80, 17.75);

% Output Terminals + Load Current
\draw[line width=1.0pt] (bot_l) -- (30.80, 14.68);
\filldraw[fill=white, draw=black, line width=1.0pt] (30.80, 17.75) circle (0.11cm);
\filldraw[fill=white, draw=black, line width=1.0pt] (30.80, 14.68) circle (0.11cm);
\node[font=\fontsize{10}{11}\selectfont\bfseries] at (30.80, 18.14) {$+$};
\node[font=\fontsize{10}{11}\selectfont\bfseries] at (30.80, 14.28) {$-$};
\node[font=\fontsize{10.5}{12}\selectfont\bfseries] at (30.80, 16.22) {$V_{t,k}$};
\draw[arr, line width=1.0pt] (30.05, 18.54) -- (31.05, 18.54);
\node[font=\fontsize{8}{9}\selectfont\bfseries] at (30.55, 18.94) {$I_k > 0$};

% KVL Bar at Bottom of Circuit Card
\filldraw[fill=anodebg, draw=black, line width=0.65pt, rounded corners=2pt] (16.45, 13.60) rectangle (31.35, 14.20);
\node[font=\fontsize{8.2}{9.5}\selectfont] at (23.90, 13.90) {
    \textbf{KVL:} $\hat{V}_{t,\mathrm{PIDKF},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - \hat{V}_{h,k}^- - I_k\hat{R}_{0,k}^- + \delta\hat{V}_{t,\mathrm{comp},k}$ \;\;($\hat{V}_{h,k}^-\!=\!0$ on NMC)
};

% --- Structural Identifiability Matrix (Bottom of Panel b: y = 10.08 to 13.26) ---
\draw[draw=black, line width=0.75pt, rounded corners=2.5pt, fill=white] (16.25, 10.08) rectangle (31.55, 13.26);
\filldraw[fill=hdrslate, draw=black, line width=0.75pt] (16.25, 12.74) rectangle (31.55, 13.26);
\node[font=\fontsize{9.5}{11}\selectfont\bfseries] at (23.90, 13.00) {Observer Parameter Allocation \& Structural Identifiability Matrix};

% Columns: [16.25, 19.75] | [19.75, 26.55] | [26.55, 31.55]
\filldraw[fill=anodebg, draw=black, line width=0.6pt] (16.25, 12.26) rectangle (19.75, 12.74);
\filldraw[fill=anodebg, draw=black, line width=0.6pt] (19.75, 12.26) rectangle (26.55, 12.74);
\filldraw[fill=anodebg, draw=black, line width=0.6pt] (26.55, 12.26) rectangle (31.55, 12.74);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (18.00, 12.50) {Estimator Class};
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (23.15, 12.50) {Online Parameter \& Residual Ownership};
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (29.05, 12.50) {Identifiability / Drift Impact};

% Row 1
\draw[draw=black, line width=0.5pt] (16.25, 11.54) rectangle (19.75, 12.26);
\draw[draw=black, line width=0.5pt] (19.75, 11.54) rectangle (26.55, 12.26);
\draw[draw=black, line width=0.5pt] (26.55, 11.54) rectangle (31.55, 12.26);
\node[align=center, font=\fontsize{7.8}{9.2}\selectfont\bfseries] at (18.00, 11.90) {Fixed-Prior Filters\\(EKF, AEKF, SMO)};
\node[align=center, font=\fontsize{7.8}{9.2}\selectfont] at (23.15, 11.90) {Freeze fresh $(R_0, R_1, R_2, C_1, C_2, Q_{\mathrm{init}})$};
\node[align=center, font=\fontsize{7.8}{9.2}\selectfont] at (29.05, 11.90) {$45.7\times$ EKF RMSE surge ($80\%$ SOH)};

% Row 2
\filldraw[fill=nmcbg, draw=black, line width=0.5pt] (16.25, 10.81) rectangle (19.75, 11.54);
\filldraw[fill=nmcbg, draw=black, line width=0.5pt] (19.75, 10.81) rectangle (26.55, 11.54);
\filldraw[fill=nmcbg, draw=black, line width=0.5pt] (26.55, 10.81) rectangle (31.55, 11.54);
\node[align=center, font=\fontsize{7.8}{9.2}\selectfont\bfseries] at (18.00, 11.175) {Dual EKF\\(DEKF)};
\node[align=center, font=\fontsize{7.8}{9.2}\selectfont] at (23.15, 11.175) {Tracks $\hat{\boldsymbol{\theta}}_k=[\hat{R}_{0,k}, 1/\hat{Q}_{\mathrm{act},k}]^T$; freezes $R_{1,2}, C_{1,2}$};
\node[align=center, font=\fontsize{7.8}{9.2}\selectfont] at (29.05, 11.175) {Avoids collinearity; leaves $V_{\mathrm{pol}}$ bias};

% Row 3
\filldraw[fill=lfpbg, draw=black, line width=0.65pt] (16.25, 10.08) rectangle (19.75, 10.81);
\filldraw[fill=lfpbg, draw=black, line width=0.65pt] (19.75, 10.08) rectangle (26.55, 10.81);
\filldraw[fill=lfpbg, draw=black, line width=0.65pt] (26.55, 10.08) rectangle (31.55, 10.81);
\node[align=center, font=\fontsize{7.6}{9}\selectfont\bfseries] at (18.00, 10.445) {Hybrid PIDKF\\(Proposed)};
\node[align=center, font=\fontsize{7.4}{8.8}\selectfont] at (23.15, 10.445) {DEKF tracks $(\hat{R}_{0,k}, \hat{Q}_{\mathrm{act},k})$\\$+\;[16,8]$ MLP $\delta\hat{V}_{t,\mathrm{comp},k}$};
\node[align=center, font=\fontsize{7.6}{9}\selectfont\bfseries] at (29.05, 10.445) {Unbiased $\delta V_{\mathrm{pol}}^*$ under BV passivity};


% =========================================================================
% PANEL (c): Bottom-Left PGFPlots (x = 0.0 to 15.7, y = 0.0 to 9.5)
% =========================================================================
\draw[panelbox] (0.0, 0.0) rectangle (15.7, 9.5);
\node[anchor=west, font=\fontsize{11.5}{13}\selectfont\bfseries] at (0.28, 9.06)
    {(c) Calibrated Open-Circuit Voltage \& LFP Hysteresis};

\begin{axis}[
    at={(1.70cm, 1.15cm)},
    anchor=south west,
    width=13.60cm,
    height=7.30cm,
    xmin=0, xmax=100,
    ymin=3.05, ymax=4.30,
    xlabel={State of Charge, SOC (\%)},
    ylabel={Open-Circuit Potential, $V_{\mathrm{ocv}}$ (V)},
    xlabel style={font=\fontsize{10}{12}\selectfont\bfseries, text=black},
    ylabel style={font=\fontsize{10}{12}\selectfont\bfseries, text=black},
    tick label style={font=\fontsize{9.5}{11}\selectfont, text=black},
    grid=both,
    grid style={dotted, black!35},
    axis line style={line width=0.85pt, draw=black},
    legend pos=north west,
    legend style={
        font=\fontsize{8.2}{9.8}\selectfont,
        text=black,
        draw=black,
        fill=white,
        fill opacity=0.95,
        cells={anchor=west},
        row sep=1.5pt
    },
    samples=160
]
    \fill[platshade] (axis cs:20, 3.05) rectangle (axis cs:80, 4.30);
    \draw[densely dotted, black!60, line width=0.7pt] (axis cs:20, 3.05) -- (axis cs:20, 4.30);
    \draw[densely dotted, black!60, line width=0.7pt] (axis cs:80, 3.05) -- (axis cs:80, 4.30);

    \fill[green!22, draw=lfpline!70, line width=0.5pt]
        plot[domain=12:95, samples=85] (\x, {3.415 - 0.286*(\x/100) + 1.831*(\x/100)^2 - 4.120*(\x/100)^3 + 4.185*(\x/100)^4 - 1.545*(\x/100)^5 + 0.030})
        -- plot[domain=95:12, samples=85] (\x, {3.415 - 0.286*(\x/100) + 1.831*(\x/100)^2 - 4.120*(\x/100)^3 + 4.185*(\x/100)^4 - 1.545*(\x/100)^5 - 0.030})
        -- cycle;
    \addlegendimage{area legend, fill=green!22, draw=lfpline!70}
    \addlegendentry{LFP Hysteresis Envelope $\pm M_0$ ($\pm 30\;\mathrm{mV}$)}

    \addplot[domain=0:100, color=nmcline, line width=1.7pt]
        {9.204*(x/100)^5 - 28.841*(x/100)^4 + 33.376*(x/100)^3 - 17.419*(x/100)^2 + 4.674*(x/100) + 3.212};
    \addlegendentry{NMC811 $V_{\mathrm{ocv}}(z)$ (Eq.\ 11, Monotonic)}

    \addplot[domain=12:95, color=lfpline, line width=1.6pt, dashed]
        {3.415 - 0.286*(x/100) + 1.831*(x/100)^2 - 4.120*(x/100)^3 + 4.185*(x/100)^4 - 1.545*(x/100)^5};
    \addlegendentry{$\mathrm{LiFePO_4}$ Equilibrium $V_{\mathrm{ocv}}^{\mathrm{LFP}}(z)$}

    \node[draw=black, line width=0.65pt, fill=white, rounded corners=2pt, align=center, font=\fontsize{8.2}{9.8}\selectfont, inner sep=3.2pt]
        (platbox) at (axis cs:39, 3.63) {
            \textbf{Biphasic Plateau (20\%--80\% SOC)}\\
            $\mathrm{LiFePO_4 \leftrightarrow FePO_4}$ ($F' = C - P = 0$)\\
            Flat Slope $\approx 0.08\;\mathrm{V}$ \& $\pm 30\;\mathrm{mV}$ Hysteresis
        };
    \draw[-{Latex[length=2mm]}, line width=0.95pt, draw=black] (platbox.south) -- (axis cs:39, 3.458);

    % Spacious Dynamic Major Hysteresis Loop Inset in Lower-Right
    \draw[draw=black, line width=0.65pt, fill=white, rounded corners=2pt]
        (axis cs:50, 3.075) rectangle (axis cs:97.5, 3.375);
    \node[font=\fontsize{7.8}{9}\selectfont\bfseries] at (axis cs:73.75, 3.346) {Dynamic Hysteresis Envelope ($\beta = 45$)};
    \fill[green!18] (axis cs:54, 3.155) rectangle (axis cs:93.5, 3.240);
    \draw[line width=1.0pt, draw=nmcline, -{Latex[length=1.8mm]}] (axis cs:54, 3.240) -- (axis cs:93.5, 3.240);
    \draw[line width=1.0pt, draw=lfpderiv, -{Latex[length=1.8mm]}] (axis cs:93.5, 3.155) -- (axis cs:54, 3.155);
    \draw[line width=0.75pt, dashed, draw=lfpline] (axis cs:54, 3.1975) -- (axis cs:93.5, 3.1975);
    \draw[line width=0.75pt, draw=black, densely dotted] (axis cs:54, 3.155) .. controls (axis cs:56, 3.222) and (axis cs:58.5, 3.240) .. (axis cs:62, 3.240);
    \draw[line width=0.75pt, draw=black, densely dotted] (axis cs:93.5, 3.240) .. controls (axis cs:91.5, 3.173) and (axis cs:89, 3.155) .. (axis cs:85.5, 3.155);
    \node[font=\fontsize{7}{8}\selectfont\bfseries] at (axis cs:73.75, 3.285) {Charge Branch ($V_{\mathrm{ocv}} + M_0$)};
    \node[font=\fontsize{7}{8}\selectfont\bfseries] at (axis cs:73.75, 3.112) {Discharge Branch ($V_{\mathrm{ocv}} - M_0$)};
\end{axis}


% =========================================================================
% PANEL (d): Bottom-Right PGFPlots (x = 16.0 to 31.8, y = 0.0 to 9.5)
% =========================================================================
\draw[panelbox] (16.0, 0.0) rectangle (31.8, 9.5);
\node[anchor=west, font=\fontsize{11.5}{13}\selectfont\bfseries] at (16.28, 9.06)
    {(d) Analytical Observability Slope $dV_{\mathrm{ocv}}/dz$ \& Condition Surge};

\begin{axis}[
    at={(17.85cm, 1.15cm)},
    anchor=south west,
    width=13.55cm,
    height=7.30cm,
    ymode=log,
    xmin=0, xmax=100,
    ymin=0.0018, ymax=9.0,
    xlabel={State of Charge, SOC (\%)},
    ylabel={OCV Slope, $dV_{\mathrm{ocv}}/dz$ (V)},
    xlabel style={font=\fontsize{10}{12}\selectfont\bfseries, text=black},
    ylabel style={font=\fontsize{10}{12}\selectfont\bfseries, text=black},
    tick label style={font=\fontsize{9.5}{11}\selectfont, text=black},
    grid=major,
    grid style={dotted, black!35},
    axis line style={line width=0.85pt, draw=black},
    legend pos=north east,
    legend style={
        font=\fontsize{8.2}{9.8}\selectfont,
        text=black,
        draw=black,
        fill=white,
        fill opacity=0.95,
        cells={anchor=west},
        row sep=1.5pt
    },
    samples=180
]
    \fill[platshade] (axis cs:20, 0.0018) rectangle (axis cs:80, 9.0);
    \draw[densely dotted, black!60, line width=0.7pt] (axis cs:20, 0.0018) -- (axis cs:20, 9.0);
    \draw[densely dotted, black!60, line width=0.7pt] (axis cs:80, 0.0018) -- (axis cs:80, 9.0);

    \addplot[domain=0:100, color=nmcline, line width=1.7pt]
        {46.020*(x/100)^4 - 115.364*(x/100)^3 + 100.128*(x/100)^2 - 34.838*(x/100) + 4.674};
    \addlegendentry{NMC Slope $V_{\mathrm{ocv}}'(z) \in [0.481, 4.674]\;\mathrm{V}$};

    \addplot[domain=12:95, color=lfpderiv, line width=1.7pt, dash pattern=on 6pt off 2.5pt on 1.5pt off 2.5pt]
        {-0.286 + 3.662*(x/100) - 12.36*(x/100)^2 + 16.74*(x/100)^3 - 7.725*(x/100)^4};
    \addlegendentry{LFP Slope $V_{\mathrm{ocv}}'(z)$ ($\approx 0.08\;\mathrm{V}$ on Plateau)};

    % BMS cutoff marker cleanly pointed out without overlapping curve
    \filldraw[fill=lfpderiv, draw=black, line width=0.6pt] (axis cs:12, 0.00278) circle (2.2pt);
    \node[draw=black, line width=0.55pt, fill=white, rounded corners=1.5pt, inner sep=2.2pt, align=left, font=\fontsize{7.5}{9}\selectfont\bfseries, anchor=west]
        (cutbox) at (axis cs:21.5, 0.0033) {BMS Cutoff $z=0.12$\\[-1pt]$V_{\mathrm{ocv}}^\prime = 0.00278\;\mathrm{V}$};
    \draw[-{Latex[length=1.7mm]}, line width=0.8pt, draw=black] (cutbox.west) -- (axis cs:12.6, 0.0028);

    \draw[densely dotted, black, line width=0.85pt] (axis cs:0, 0.08) -- (axis cs:100, 0.08);

    \draw[{Latex[length=1.8mm]}-{Latex[length=1.8mm]}, line width=0.95pt, draw=black]
        (axis cs:50, 0.132) -- (axis cs:50, 0.56);
    \node[draw=black, line width=0.6pt, fill=white, rounded corners=2pt, inner sep=2.5pt, align=center, font=\fontsize{8}{9.5}\selectfont\bfseries]
        at (axis cs:50, 0.275) {LFP Plateau Bottleneck\\[-1pt]($V_{\mathrm{ocv}}'\approx 0.08\;\mathrm{V},\;\sim 7\times\text{ Drop}$)};

    \node[draw=black, line width=0.65pt, fill=white, rounded corners=2pt, align=left, font=\fontsize{7.8}{9.4}\selectfont, inner sep=3.2pt]
        at (axis cs:64, 0.0155) {
            \textbf{Observability:} $|\det(\mathcal{O}_k)| \propto V_{\mathrm{ocv}}'(z_k)$ (Eqs.\ 12, 16)\\[1.5pt]
            $\bullet$ NMC811: $\kappa(\mathcal{O}_k) \approx 3.2\times 10^4 \;\Rightarrow\;$ Strong Innovation Gain\\[1pt]
            $\bullet$ LFP Plateau: $\kappa(\mathcal{O}_k) > 1.99\times 10^5 \;\Rightarrow\;$ Needs $V_h + \delta\hat{V}_{\mathrm{comp}}$
        };
\end{axis}

\end{tikzpicture}
\end{document}
"""


# =============================================================================
# FIGURE 2: HYBRID PIDKF CLOSED-LOOP ARCHITECTURE & 10-PHASE BENCHMARK
# Unified 32.0 cm x 20.8 cm TikZ vector figure
# =============================================================================
TIKZ_FRAMEWORK = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=34.2cm,paperheight=22.8cm,margin=0.8cm]{geometry}
\pagestyle{empty}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,backgrounds,fit,shadows}

\definecolor{cplant}{RGB}{241,245,249}
\definecolor{cdekf}{RGB}{239,246,255}
\definecolor{cmlp}{RGB}{236,253,245}
\definecolor{cgate}{RGB}{254,243,199}
\definecolor{cforbid}{RGB}{254,226,226}
\definecolor{cadmit}{RGB}{187,247,208}
\definecolor{hdrslate}{RGB}{226,232,240}
\definecolor{hdrblue}{RGB}{219,234,254}
\definecolor{hdrgreen}{RGB}{209,250,229}
\definecolor{hdramber}{RGB}{254,243,199}
\definecolor{hdrteal}{RGB}{204,251,241}

\begin{document}\noindent
\begin{tikzpicture}[
    x=1cm, y=1cm,
    >=Latex,
    text=black,
    draw=black,
    font=\normalsize,
    outerbox/.style={draw=black, line width=1.1pt, rounded corners=4pt, fill=white},
    colbox/.style={draw=black, line width=0.8pt, rounded corners=3pt, fill=cplant},
    colhdr/.style={draw=black, line width=0.8pt, font=\fontsize{9.5}{11}\selectfont\bfseries, align=center},
    subcard/.style={draw=black, line width=0.7pt, rounded corners=2.5pt, fill=white, align=center, drop shadow={opacity=0.08, shadow xshift=1pt, shadow yshift=-1pt}},
    phasecard/.style={draw=black, line width=0.65pt, rounded corners=2pt, fill=white, align=center, drop shadow={opacity=0.06, shadow xshift=0.8pt, shadow yshift=-0.8pt}},
    arr/.style={-{Latex[length=2.4mm,width=1.7mm]}, line width=0.95pt, draw=black},
    darr/.style={-{Latex[length=2.2mm,width=1.6mm]}, line width=0.85pt, draw=black, densely dashed}
]

% =========================================================================
% PANEL (a): Closed-Loop PIDKF Architecture (y = 8.45 to 20.80)
% =========================================================================
\draw[outerbox] (0.0, 8.45) rectangle (31.8, 20.80);
\node[anchor=west, font=\fontsize{12.5}{14}\selectfont\bfseries] at (0.28, 20.36)
    {(a) Closed-Loop Hybrid PIDKF Architecture with Butler--Volmer Passivity and Exogenous Decoupling};

% Draw Module background containers first so top feedforward bus renders cleanly on top!
\filldraw[colbox] (0.28, 8.70) rectangle (6.85, 19.90);
\filldraw[colbox, fill=cdekf] (7.45, 8.70) rectangle (17.55, 19.90);
\filldraw[colbox, fill=cmlp] (18.15, 8.70) rectangle (31.56, 19.90);

% --- MODULE 1: Plant, Telemetry & Hard Rest Gate (x = 0.28 to 6.85) ---
\filldraw[colhdr, fill=hdrslate] (0.28, 19.30) rectangle (6.85, 19.90);
\node[font=\fontsize{9.5}{11}\selectfont\bfseries] at (3.565, 19.60) {1.\ Plant, Sensing \& Rest Gate};

% Card 1A: Physical Cells & Battery Icon
\draw[subcard] (0.46, 16.58) rectangle (6.67, 19.08);
\filldraw[fill=hdrgreen, draw=black, line width=0.75pt, rounded corners=1.5pt] (0.64, 17.08) rectangle (1.18, 18.42);
\filldraw[fill=hdrslate, draw=black, line width=0.75pt] (0.80, 18.42) rectangle (1.02, 18.56);
\draw[line width=0.6pt] (0.72, 17.46) -- (1.10, 17.46);
\draw[line width=0.6pt] (0.72, 17.90) -- (1.10, 17.90);
\node[font=\fontsize{7}{8}\selectfont\bfseries] at (0.91, 18.18) {$+$};
\node[font=\fontsize{7}{8}\selectfont\bfseries] at (0.91, 17.24) {$-$};
\node[anchor=west, align=left, font=\fontsize{8}{9.6}\selectfont] at (1.30, 17.83) {
    \textbf{Multi-Chemistry Cell Plant}\\[2pt]
    NMC811: INR18650-20R ($2.00\;\mathrm{Ah}$)\\[1pt]
    LFP: ANR26650M1-B ($2.30\;\mathrm{Ah}$)\\[1pt]
    $\mathrm{SOH}\in[80\%,100\%],\; T\in[0^\circ, 45^\circ\mathrm{C}]$
};

% Card 1B: Causal BMS Sensing (1 Hz) + Mini Drive Waveform
\draw[subcard] (0.46, 12.60) rectangle (6.67, 16.08);
\node[font=\fontsize{8.5}{10}\selectfont\bfseries] at (3.565, 15.76) {Causal BMS Telemetry ($1\;\mathrm{Hz}$)};
\draw[line width=0.6pt, draw=black!50] (0.76, 15.02) -- (6.37, 15.02);
\draw[line width=0.95pt, draw=black]
    (0.76, 15.02) -- (1.35, 15.02) -- (1.35, 15.38) -- (2.22, 15.38) -- (2.22, 14.70) -- (2.88, 14.70)
    -- (2.88, 15.22) -- (3.76, 15.22) -- (3.76, 15.02) -- (4.58, 15.02) -- (4.58, 15.40) -- (5.60, 15.40) -- (5.60, 15.02) -- (6.37, 15.02);
\node[align=center, font=\fontsize{8}{9.6}\selectfont] at (3.565, 13.56) {
    Current: $I_k = I_{\mathrm{true},k} + I_{\mathrm{bias}} + n_{i,k}$\\[1.5pt]
    Voltage: $V_{\mathrm{meas},k} = V_{t,k} + v_k$\\[1.5pt]
    Surface Temperature: $T_k$
};

\draw[arr] (3.565, 16.58) -- (3.565, 16.08);

% Card 1C: Hard Rest-Period Gate
\draw[subcard, fill=cgate] (0.46, 8.98) rectangle (6.67, 12.06);
\node[font=\fontsize{8.5}{10}\selectfont\bfseries] at (3.565, 11.74) {Hard Rest-Period Innovation Gate};
\draw[line width=0.8pt] (1.10, 11.00) -- (1.76, 11.00);
\draw[line width=1.1pt] (1.76, 11.00) -- (2.38, 11.28);
\filldraw[fill=white, draw=black, line width=0.7pt] (1.76, 11.00) circle (1.5pt);
\filldraw[fill=white, draw=black, line width=0.7pt] (2.42, 11.00) circle (1.5pt);
\draw[line width=0.8pt] (2.42, 11.00) -- (2.88, 11.00);
\node[anchor=west, align=left, font=\fontsize{8}{9.4}\selectfont] at (3.08, 11.04) {
    $|I_k| < 10\;\mathrm{mA}$: Freeze Gains
};
\node[align=center, font=\fontsize{7.8}{9.4}\selectfont] at (3.565, 9.75) {
    Set $\mathbf{K}_k = \mathbf{0},\;\; \mathbf{K}_{\theta,k} = \mathbf{0}$ during rest\\[1.5pt]
    Enforces $\|\mathbf{F}_k\|_2 = \|\mathbf{A}_k\|_2 \le 1$ (Passivity)
};

\draw[arr] (3.565, 12.60) -- (3.565, 12.06);

% --- MODULE 2: Dual EKF Backbone (x = 7.45 to 17.55) ---
\filldraw[colhdr, fill=hdrblue] (7.45, 19.30) rectangle (17.55, 19.90);
\node[font=\fontsize{9.5}{11}\selectfont\bfseries] at (12.50, 19.60) {2.\ Dual EKF Backbone ($980\;\mathrm{FLOPs/Step}$)};

% Fast State Filter (y = 14.32 to 18.42)
\draw[subcard] (7.70, 14.32) rectangle (17.30, 18.42);
\filldraw[fill=hdrblue, draw=black, line width=0.55pt, rounded corners=1.5pt] (7.90, 17.82) rectangle (17.10, 18.28);
\node[font=\fontsize{8.4}{9.8}\selectfont\bfseries] at (12.50, 18.05) {Fast State Filter ($\tau_{\mathrm{state}}\sim 10^1\text{--}10^2\;\mathrm{s}$, Joseph Form)};
\node[align=center, font=\fontsize{8}{9.6}\selectfont] at (12.50, 16.06) {
    \textbf{Predict:} $\hat{\mathbf{x}}_k^- = \mathbf{A}_{k-1}\hat{\mathbf{x}}_{k-1}^+ + \mathbf{B}_{k-1}I_{k-1} + \mathbf{g}_{k-1}$\\[3.5pt]
    $\mathbf{P}_k^- = \mathbf{A}_{k-1}\mathbf{P}_{k-1}^+\mathbf{A}_{k-1}^T + \mathbf{Q}$\\[4pt]
    \textbf{Update:} $\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k e_{k,\mathrm{PIDKF}}$,\;\; $\mathbf{P}_k^+$ (Joseph Form)\\[4pt]
    \textbf{ECM Output:} $\hat{V}_{t,\mathrm{ECM},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - \hat{V}_{h,k}^- - I_k\hat{R}_{0,k}^-$
};

\draw[arr] (11.00, 14.32) -- (11.00, 13.34);
\node[fill=cdekf, inner sep=1.5pt, font=\fontsize{8}{9}\selectfont\bfseries] at (10.42, 13.83) {$\hat{z}_k^+$};
\draw[arr] (14.00, 13.34) -- (14.00, 14.32);
\node[fill=cdekf, inner sep=1.5pt, font=\fontsize{8}{9}\selectfont\bfseries] at (14.58, 13.83) {$\hat{\boldsymbol{\theta}}_k^+$};
\node[draw=black, line width=0.45pt, fill=white, rounded corners=1.5pt, inner sep=1.8pt, font=\fontsize{7.2}{8.5}\selectfont\bfseries]
    at (12.50, 13.83) {$\tau_{\mathrm{state}} / \tau_{\mathrm{SOH}} \ll 10^{-4}$};

% Slow Parameter Filter (y = 8.98 to 13.34)
\draw[subcard] (7.70, 8.98) rectangle (17.30, 13.34);
\filldraw[fill=hdrblue, draw=black, line width=0.55pt, rounded corners=1.5pt] (7.90, 12.74) rectangle (17.10, 13.20);
\node[font=\fontsize{8.4}{9.8}\selectfont\bfseries] at (12.50, 12.97) {Slow Parameter Filter ($\tau_{\mathrm{SOH}}\sim 10^5\text{--}10^6\;\mathrm{s}$, Random Walk)};
\node[align=center, font=\fontsize{8}{9.6}\selectfont] at (12.50, 10.86) {
    $\hat{\boldsymbol{\theta}}_k = \left[\hat{R}_{0,k},\; 1/\hat{Q}_{\mathrm{act},k}\right]^T$,\quad $\hat{\boldsymbol{\theta}}_k^+ = \hat{\boldsymbol{\theta}}_k^- + \mathbf{K}_{\theta,k} e_{k,\mathrm{PIDKF}}$\\[4pt]
    Fr\'echet Jacobian (Eq.\ 17):\; $\mathbf{C}_{\theta,k} = \left[-I_k,\; -\dfrac{\eta\Delta t I_{k-1}}{3600}V_{\mathrm{ocv}}^\prime(\hat{z}_k^+)\right]$\\[4.5pt]
    Thermal-Inverted Health:\; $\mathrm{SOH}_{\mathrm{ref},k} \equiv \dfrac{\hat{Q}_{\mathrm{act},k}(T_k)}{Q_{\mathrm{init}}\,f_Q(T_k)} = \dfrac{1}{Q_{\mathrm{init}}\,\hat{\theta}_{2,k}\,f_Q(T_k)}$
};

% Routing Wires Module 1 -> Module 2
\draw[arr] (6.67, 15.05) -- (7.70, 15.05);
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{7}{8}\selectfont\bfseries]
    at (7.15, 15.48) {$I_k, T_k$};
\draw[darr] (6.67, 10.52) -- (7.70, 10.52);
\draw[darr] (7.15, 10.52) |- (7.70, 14.62);
\node[fill=cgate, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{7}{8}\selectfont\bfseries]
    at (7.15, 12.60) {Gate};

% Clean Top Measurement Feedforward Corridor (y = 18.88, cleanly above ALL subcards!)
\draw[arr] (6.67, 15.80) -- (7.15, 15.80) -- (7.15, 18.88) -- (31.36, 18.88) -- (31.36, 13.45) -- (30.22, 13.45);
\node[fill=white, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.8pt, font=\fontsize{7.6}{8.8}\selectfont\bfseries]
    at (14.40, 18.88) {Direct BMS Bus: $(I_k, T_k)$ to MLP $\mathbf{u}_k$ and $V_{\mathrm{meas},k}$ to Summing Junction $\Sigma_2$};
\draw[arr] (21.65, 18.88) -- (21.65, 18.42);


% --- MODULE 3: Neural Compensator Topology, Passivity Plot & Innovation (x = 18.15 to 31.56) ---
\filldraw[colhdr, fill=hdrgreen] (18.15, 19.30) rectangle (31.56, 19.90);
\node[font=\fontsize{9.5}{11}\selectfont\bfseries] at (24.855, 19.60) {3.\ Residual MLP Compensator \& Innovation Synthesis};

% Subcard 3A: REAL 6 -> 16 -> 8 -> 1 Neural Network Topology Graphic (x = 18.35 to 25.00, y = 14.00 to 18.42)
\draw[subcard] (18.35, 14.00) rectangle (25.00, 18.42);
\filldraw[fill=hdrgreen, draw=black, line width=0.55pt, rounded corners=1.5pt] (18.50, 17.84) rectangle (24.85, 18.28);
\node[font=\fontsize{7.4}{8.5}\selectfont\bfseries] at (21.675, 18.06) {MLP ($6\to 16\to 8\to 1$ GELU, $560\;\mathrm{FLOPs}$)};

% Nodes shifted comfortably right (x=20.00) so SOH_{ref,k} fits cleanly inside left border!
\foreach \i/\lbl/\yy in {1/{$I_k$}/17.32, 2/{$T_k$}/16.80, 3/{$\hat{z}_k^-$}/16.28, 4/{$\hat{U}_{1,k}^-$}/15.76, 5/{$\hat{U}_{2,k}^-$}/15.24, 6/{$\mathrm{SOH}_{\mathrm{ref},k}$}/14.72} {
    \filldraw[fill=hdrblue, draw=black, line width=0.6pt] (20.00, \yy) circle (0.13cm);
    \node[anchor=east, font=\fontsize{7.2}{8.2}\selectfont\bfseries] at (19.82, \yy) {\lbl};
}
\foreach \j/\yj in {1/17.16, 2/16.56, 3/15.96, 4/15.36, 5/14.76} {
    \filldraw[fill=hdramber, draw=black, line width=0.6pt] (21.35, \yj) circle (0.13cm);
    \foreach \yy in {17.32, 16.80, 16.28, 15.76, 15.24, 14.72} {
        \draw[line width=0.28pt, black!45] (20.13, \yy) -- (21.22, \yj);
    }
}
\foreach \k/\yk in {1/16.90, 2/16.26, 3/15.62, 4/14.98} {
    \filldraw[fill=hdrteal, draw=black, line width=0.6pt] (22.95, \yk) circle (0.13cm);
    \foreach \yj in {17.16, 16.56, 15.96, 15.36, 14.76} {
        \draw[line width=0.28pt, black!45] (21.48, \yj) -- (22.82, \yk);
    }
}
\filldraw[fill=hdrgreen, draw=black, line width=0.75pt] (24.18, 15.96) circle (0.18cm);
\foreach \yk in {16.90, 16.26, 15.62, 14.98} {
    \draw[line width=0.35pt, black!60] (23.08, \yk) -- (24.00, 15.96);
}
\node[font=\fontsize{6.8}{7.8}\selectfont\bfseries] at (21.35, 17.56) {$16$ (GELU)};
\node[font=\fontsize{6.8}{7.8}\selectfont\bfseries] at (22.95, 17.56) {$8$ (GELU)};
\node[font=\fontsize{7}{8}\selectfont\bfseries] at (24.22, 16.48) {$\mathcal{NN}(\mathbf{u}_k)$};
\node[font=\fontsize{7.4}{8.5}\selectfont\bfseries] at (21.675, 14.22) {$257\;\text{Params} = 1.03\;\mathrm{KB}$ Flash (Zero Leakage)};

% Subcard 3B: VISUAL BUTLER-VOLMER PASSIVITY QUADRANT PLOT (x = 25.35 to 31.12, y = 14.00 to 18.42)
\draw[subcard] (25.35, 14.00) rectangle (31.12, 18.42);
\filldraw[fill=hdrgreen, draw=black, line width=0.55pt, rounded corners=1.5pt] (25.50, 17.84) rectangle (30.97, 18.28);
\node[font=\fontsize{8}{9.2}\selectfont\bfseries] at (28.235, 18.06) {Butler--Volmer Passivity Sector};

% 2D Coordinate Plane centered at (28.235, 15.92)
\fill[cforbid] (28.235, 15.92) rectangle (30.52, 17.16);
\fill[cforbid] (25.95, 14.68) rectangle (28.235, 15.92);
\fill[cadmit] (25.95, 15.92) -- (28.235, 15.92) -- (28.235, 16.36) -- (25.95, 16.92) -- cycle;
\fill[cadmit] (28.235, 15.92) -- (30.52, 15.92) -- (30.52, 14.92) -- (28.235, 15.48) -- cycle;
\draw[line width=1.05pt, draw=black] (25.95, 16.92) -- (28.235, 16.36) -- (28.235, 15.48) -- (30.52, 14.92);
\draw[line width=0.7pt, densely dashed, draw=black!75] (25.95, 17.04) -- (30.52, 17.04);
\draw[line width=0.7pt, densely dashed, draw=black!75] (25.95, 14.80) -- (30.52, 14.80);
\draw[arr, line width=0.8pt] (25.82, 15.92) -- (30.72, 15.92);
\draw[arr, line width=0.8pt] (28.235, 14.56) -- (28.235, 17.34);
\node[font=\fontsize{7.5}{8.5}\selectfont\bfseries] at (30.65, 16.14) {$I_k$};
\node[font=\fontsize{7.5}{8.5}\selectfont\bfseries] at (28.86, 17.36) {$\delta\hat{V}_{t,\mathrm{comp},k}$};
\node[font=\fontsize{7}{8}\selectfont\bfseries] at (27.10, 16.30) {Charge OK};
\node[font=\fontsize{7}{8}\selectfont\bfseries] at (29.38, 15.52) {Disch.\ OK};
\node[font=\fontsize{6.8}{8}\selectfont] at (29.40, 16.62) {Clamped ($I_k\delta\hat{V}\!>\!0$)};
\node[font=\fontsize{7.2}{8.4}\selectfont\bfseries] at (28.235, 14.22)
    {$|\delta\hat{V}|\le V_{\mathrm{pol}}^{\max}(I_k)\le 20.0\;\mathrm{mV}$;\; $I_k\cdot\delta\hat{V}\le 0$};

\draw[arr] (25.00, 15.98) -- (25.35, 15.98);

% Cascaded Exact Summing Junctions: Sigma1 = V_ECM + deltaV -> Sigma2 = +V_meas - V_PIDKF
\draw[arr] (17.30, 15.10) -- (17.85, 15.10) -- (17.85, 13.45) -- (25.76, 13.45);
\node[fill=cmlp, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.4pt, font=\fontsize{7.2}{8.2}\selectfont\bfseries]
    at (21.50, 13.45) {$\hat{V}_{t,\mathrm{ECM},k}^-$};
\draw[arr] (26.00, 14.00) -- (26.00, 13.69);
\filldraw[fill=white, draw=black, line width=0.85pt] (26.00, 13.45) circle (0.24cm);
\node[font=\fontsize{7.8}{8.8}\selectfont\bfseries] at (26.00, 13.45) {$\Sigma_1$};
\node[font=\fontsize{6.8}{7.5}\selectfont\bfseries] at (25.60, 13.64) {$+$};
\node[font=\fontsize{6.8}{7.5}\selectfont\bfseries] at (26.20, 13.84) {$+$};

% Output of Sigma1 feeds Sigma2 with minus sign!
\draw[arr] (26.24, 13.45) -- (29.74, 13.45);
\node[fill=cmlp, inner sep=1.2pt, font=\fontsize{7}{8}\selectfont\bfseries]
    at (28.00, 13.45) {$\hat{V}_{t,\mathrm{PIDKF},k}^-$};
\filldraw[fill=white, draw=black, line width=0.85pt] (29.98, 13.45) circle (0.24cm);
\node[font=\fontsize{7.8}{8.8}\selectfont\bfseries] at (29.98, 13.45) {$\Sigma_2$};
\node[font=\fontsize{7}{7.8}\selectfont\bfseries] at (29.56, 13.64) {$-$};
\node[font=\fontsize{7}{7.8}\selectfont\bfseries] at (30.38, 13.64) {$+$};
\node[font=\fontsize{7}{8}\selectfont\bfseries] at (30.82, 13.68) {$V_{\mathrm{meas},k}$};

\draw[arr] (29.98, 13.21) -- (29.98, 12.88);
\node[anchor=east, font=\fontsize{7.2}{8.2}\selectfont\bfseries] at (29.82, 13.06) {$e_{k,\mathrm{PIDKF}}$};

% Subcard 3C: Innovation Update, Exogenous Decoupling & Mean-Square UUB (y = 8.98 to 12.88)
\draw[subcard] (18.35, 8.98) rectangle (31.32, 12.88);
\filldraw[fill=hdrgreen, draw=black, line width=0.55pt, rounded corners=1.5pt] (18.50, 12.30) rectangle (31.17, 12.74);
\node[font=\fontsize{8.2}{9.4}\selectfont\bfseries] at (24.835, 12.52)
    {Compensated Innovation Update, Exogenous Decoupling \& Stability Bound};
\node[align=center, font=\fontsize{7.8}{9.4}\selectfont] at (24.835, 10.64) {
    $\hat{V}_{t,\mathrm{PIDKF},k}^- = \hat{V}_{t,\mathrm{ECM},k}^- + \delta\hat{V}_{t,\mathrm{comp},k}$,\quad $e_{k,\mathrm{PIDKF}} = V_{\mathrm{meas},k} - \hat{V}_{t,\mathrm{PIDKF},k}^-$\\[3.5pt]
    $R_{\mathrm{eff},k} = R_v + \sigma_{\mathrm{NN}}^2 + \sigma_{\mathrm{approx}}^2 = 1.04\times 10^{-3}\;\mathrm{V}^2$ \;\;($R_v\!=\!10^{-3}\;\mathrm{V}^2,\; \sigma_{\mathrm{NN}}\!=\!5.8,\; \sigma_{\mathrm{approx}}\!=\!1.9\;\mathrm{mV}$)\\[3.5pt]
    \textbf{Exogenous Decoupling:}\; $\mathbf{H}_{\mathrm{PIDKF},k}\approx\mathbf{H}_k,\;\mathbf{C}_{\theta,\mathrm{PIDKF},k}\approx\mathbf{C}_{\theta,k}$ \;\textbf{($-37.3\%$ MCU Cycles)}\\[3.5pt]
    \textbf{Mean-Square UUB (Eqs.\ 24--25):}\; $\limsup_{k\to\infty}\mathbb{E}[\|\mathbf{e}_k^-\|^2] \le n\bar{p} + \left(\dfrac{c\bar{A}\bar{K}\bar{\varepsilon}}{1-\rho}\right)^2 < \infty$
};

% Clean routing from DEKF -> MLP inputs
\draw[arr] (17.30, 16.45) -- (18.35, 16.45);
\node[fill=white, draw=black, line width=0.4pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{7}{8}\selectfont\bfseries]
    at (17.85, 16.86) {$\hat{\mathbf{x}}_k^-$};
\draw[arr] (17.30, 11.40) -- (17.85, 11.40) |- (18.35, 14.74);
\node[fill=white, draw=black, line width=0.4pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{7}{8}\selectfont\bfseries]
    at (17.85, 12.90) {$\mathrm{SOH}_{\mathrm{ref}}$};

% Innovation feedback back to Fast State & Slow Parameter Kalman updates
\draw[arr] (18.35, 9.75) -- (17.30, 9.75);
\draw[arr] (17.65, 9.75) |- (17.30, 16.02);
\node[fill=cgate, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.5pt, font=\fontsize{7.2}{8.2}\selectfont\bfseries]
    at (17.85, 9.30) {$e_{k,\mathrm{PIDKF}}$};


% =========================================================================
% PANEL (b): Empirical Protocol, 8-Estimator Taxonomy, Hardware & 10 Phases
% =========================================================================
\draw[outerbox] (0.0, 0.0) rectangle (31.8, 8.15);
\node[anchor=west, font=\fontsize{12.5}{14}\selectfont\bfseries] at (0.28, 7.72)
    {(b) Dual-Tier Cell-Batch Validation Protocol, 8-Estimator Benchmark Taxonomy, Hardware Profiling \& 10-Phase Matrix};

% Top Pillar 1: Empirical Datasets & Zero-Leakage Split (x = 0.28 to 10.30)
\draw[subcard, fill=cplant] (0.28, 5.18) rectangle (10.30, 7.34);
\filldraw[fill=hdrslate, draw=black, line width=0.65pt] (0.28, 6.82) rectangle (10.30, 7.34);
\node[font=\fontsize{8.8}{10}\selectfont\bfseries] at (5.29, 7.08) {Zero-Leakage Datasets \& Experimental Split};
\node[align=center, font=\fontsize{7.8}{9.4}\selectfont] at (5.29, 5.98) {
    CALCE (\texttt{INR18650-20R}, \texttt{ANR26650M1-B}), Oxford EIS, NASA PCoE\\[2pt]
    Cell Batch~A (Calibrate/Train) $\perp$ Cell Batch~B (Hold-Out Test)\\[2pt]
    Arbin BT-2000 Reference Uncertainty: $\sigma_{\mathrm{SOC,ref}} \le \pm 0.036\%$
};

% Top Pillar 2: 8 Benchmarked Estimators Across 3 Paradigms (x = 10.85 to 21.05)
\draw[subcard, fill=cdekf] (10.85, 5.18) rectangle (21.05, 7.34);
\filldraw[fill=hdrblue, draw=black, line width=0.65pt] (10.85, 6.82) rectangle (21.05, 7.34);
\node[font=\fontsize{8.8}{10}\selectfont\bfseries] at (15.95, 7.08) {8 Benchmarked Estimators Across 3 Paradigms (M4F)};
\node[align=center, font=\fontsize{7.8}{9.4}\selectfont] at (15.95, 5.98) {
    1.\ \textbf{Model-Based:} EKF ($3.87\;\mu\mathrm{s}$), UKF ($12.42\;\mu\mathrm{s}$), SMO ($1.67\;\mu\mathrm{s}$)\\[2pt]
    2.\ \textbf{Adaptive/Dual:} AEKF ($5.24\;\mu\mathrm{s}$), DEKF ($8.63\;\mu\mathrm{s}$)\\[2pt]
    3.\ \textbf{ML/Hybrid:} BiLSTM ($2.44\;\mathrm{ms}$), PINN ($127.98\;\mu\mathrm{s}$), \textbf{PIDKF ($14.20\;\mu\mathrm{s}$)}
};

% Top Pillar 3: Bare-Metal MCU Profiling + Chip Icon (x = 21.60 to 31.52)
\draw[subcard, fill=cmlp] (21.60, 5.18) rectangle (31.52, 7.34);
\filldraw[fill=hdrgreen, draw=black, line width=0.65pt] (21.60, 6.82) rectangle (31.52, 7.34);
\node[font=\fontsize{8.8}{10}\selectfont\bfseries] at (26.56, 7.08) {Bare-Metal ARM Cortex-M4F Hardware Bench};
% Chip icon placed cleanly at left without touching text!
\filldraw[fill=hdrslate, draw=black, line width=0.75pt] (21.96, 5.56) rectangle (22.64, 6.42);
\foreach \p in {5.70, 5.90, 6.10, 6.28} {
    \draw[line width=0.65pt] (21.78, \p) -- (21.96, \p);
    \draw[line width=0.65pt] (22.64, \p) -- (22.82, \p);
}
\node[font=\fontsize{6.8}{8}\selectfont\bfseries] at (22.30, 5.99) {M4F};
\node[anchor=west, align=left, font=\fontsize{7.5}{9.2}\selectfont] at (22.96, 5.98) {
    STM32F407VG ($168\;\mathrm{MHz}$, IEEE-754 FPU, \texttt{DWT->CYCCNT})\\[2pt]
    Zero Heap (MISRA-C:2012 C99 / ARM GCC 12.3 \texttt{-O3})\\[2pt]
    $1,540\;\mathrm{FLOPs}$, $2,386\;\text{cyc}$, $8.2\;\mathrm{KB}$ Flash, $0.85\;\mathrm{KB}$ RAM ($4.97\;\mu\mathrm{s}$ M7)
};

\draw[arr] (10.30, 6.26) -- (10.85, 6.26);
\draw[arr] (21.05, 6.26) -- (21.60, 6.26);

\draw[line width=0.75pt, draw=black] (5.29, 5.18) -- (5.29, 4.92) -- (26.56, 4.92) -- (26.56, 5.18);
\draw[arr] (15.95, 5.18) -- (15.95, 4.65);

% --- 10-Phase Matrix Grid (2 rows x 5 columns, y = 0.25 to 4.65) ---
% Row 1: Phases 1-5 (y = 2.55 to 4.65)
\draw[phasecard] (0.28, 2.55) rectangle (6.30, 4.65);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (0.28, 4.18) rectangle (6.30, 4.65);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (3.29, 4.415) {Phase 1: Dynamic Cycles};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (3.29, 3.36) {
    UDDS, US06, FUDS, DST\\
    Fresh Cell ($100\%$ SOH, $25^\circ\mathrm{C}$)\\
    \textbf{EKF: $0.065\%$ $\mid$ DEKF: $0.191\%$}
};

\draw[phasecard] (6.65, 2.55) rectangle (12.67, 4.65);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (6.65, 4.18) rectangle (12.67, 4.65);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (9.66, 4.415) {Phase 2: Multi-Stage Aging};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (9.66, 3.36) {
    $\mathrm{SOH}\;100\% \to 95\% \to 90\% \to 80\%$\\
    EKF Surge: $45.7\times$ ($2.973\%$)\\
    \textbf{DEKF: $0.585\%$ $\mid$ PIDKF: $0.482\%$}
};

\draw[phasecard] (13.02, 2.55) rectangle (19.04, 4.65);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (13.02, 4.18) rectangle (19.04, 4.65);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (16.03, 4.415) {Phase 3: Thermal Regimes};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (16.03, 3.36) {
    Cold ($0^\circ\mathrm{C}$) \& Hot ($45^\circ\mathrm{C}$)\\
    DEKF $0^\circ\mathrm{C}$ Bias ($2.254\%$) vs.\ SMO ($0.929\%$)\\
    \textbf{Aged PIDKF ($0^\circ\mathrm{C}$): $0.842\%$ (NMC)}
};

\draw[phasecard] (19.39, 2.55) rectangle (25.41, 4.65);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (19.39, 4.18) rectangle (25.41, 4.65);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (22.40, 4.415) {Phase 4: Initial Convergence};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (22.40, 3.36) {
    Initial Offset $\Delta\mathrm{SOC}_0 = -40\%$\\
    Closed-Loop Riccati Contraction\\
    \textbf{EKF/AEKF: $7/6\;\mathrm{s}$ $\mid$ PIDKF: $18.0\;\mathrm{s}$}
};

\draw[phasecard] (25.76, 2.55) rectangle (31.52, 4.65);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (25.76, 4.18) rectangle (31.52, 4.65);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (28.64, 4.415) {Phase 5: Radar Trade-Off};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (28.64, 3.36) {
    5-Axis Engineering Trade-Off:\\
    Accuracy, Aging, Thermal,\\
    Convergence \& MCU Efficiency
};

% Row 2: Phases 6-10 (y = 0.25 to 2.35)
\draw[phasecard] (0.28, 0.25) rectangle (6.30, 2.35);
\filldraw[fill=hdrteal, draw=black, line width=0.6pt] (0.28, 1.88) rectangle (6.30, 2.35);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (3.29, 2.115) {Phase 6: LFP Chemistry};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (3.29, 1.06) {
    $20\%\text{--}80\%$ Plateau $+V_h$ ($\pm 30\;\mathrm{mV}$)\\
    Gramian $\lambda_{\min}(\mathcal{G}_o) \ge 1.21\times 10^{-3}$\\
    \textbf{Aged PIDKF: $0.149\%$ ($0.158\%$ CALCE)}
};

\draw[phasecard] (6.65, 0.25) rectangle (12.67, 2.35);
\filldraw[fill=hdramber, draw=black, line width=0.6pt] (6.65, 1.88) rectangle (12.67, 2.35);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (9.66, 2.115) {Phase 7: Sensor Faults};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (9.66, 1.06) {
    $+0.25\;\mathrm{A}$ Hall Bias \& $25\;\mathrm{mV}$ EMI\\
    SMO Drift Clamping: $1.024\%$\\
    \textbf{EKF: $1.817\%$ $\mid$ PIDKF: $1.428\%$ RMSE}
};

\draw[phasecard] (13.02, 0.25) rectangle (19.04, 2.35);
\filldraw[fill=hdrteal, draw=black, line width=0.6pt] (13.02, 1.88) rectangle (19.04, 2.35);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (16.03, 2.115) {Phase 8: EIS Diagnostics};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (16.03, 1.06) {
    $100\;\mathrm{kHz}\text{--}1\;\mathrm{mHz}$ Randles--CPE\\
    $R_0$ ($+50\%$), $R_{\mathrm{ct}}$ ($+150\%$), $\alpha=0.88$\\
    \textbf{RMS Fit Error: $1.1\%\text{--}1.8\%$}
};

\draw[phasecard] (19.39, 0.25) rectangle (25.41, 2.35);
\filldraw[fill=hdramber, draw=black, line width=0.6pt] (19.39, 1.88) rectangle (25.41, 2.35);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (22.40, 2.115) {Phase 9: 8S Pack Scaling};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (22.40, 1.06) {
    Dispersion ($\sigma_Q = 3.0\%, \sigma_R = 10.0\%$)\\
    Core Thermal $\Delta T = +3.2^\circ\mathrm{C}$ ($1.24\times$ Fade)\\
    \textbf{Weakest Cell 4: $0.337\%$ (Dist.\ DEKF)}
};

\draw[phasecard, fill=cmlp, line width=0.9pt] (25.76, 0.25) rectangle (31.52, 2.35);
\filldraw[fill=hdrgreen, draw=black, line width=0.9pt] (25.76, 1.88) rectangle (31.52, 2.35);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (28.64, 2.115) {Phase 10: Hybrid PIDKF};
\node[align=center, font=\fontsize{7.6}{9.2}\selectfont] at (28.64, 1.06) {
    Ablation \& Cortex-M4 Pareto\\
    $0.414\%$ (NMC) \& $0.149\%$ (LFP)\\
    \textbf{$14.20\;\mu\mathrm{s}$ M4 ($-37.3\%$ Exog.\ Cyc)}
};

\end{tikzpicture}
\end{document}
"""


def compile_vector_and_png(tex_str, basename, folders, dpi=600):
    work_dir = folders[0]
    os.makedirs(work_dir, exist_ok=True)
    tex_path = os.path.join(work_dir, f"{basename}.tex")
    pdf_path = os.path.join(work_dir, f"{basename}.pdf")
    png_path = os.path.join(work_dir, f"{basename}.png")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_str)

    cmd = [PDFLATEX, "-interaction=nonstopmode", f"{basename}.tex"]
    res = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(pdf_path):
        print(res.stdout[-2500:])
        print(res.stderr[-2500:])
        raise RuntimeError(f"Failed to compile {tex_path}")

    doc = fitz.open(pdf_path)
    if len(doc) != 1:
        print(f"WARNING: {pdf_path} has {len(doc)} pages (expected 1)")
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(png_path)

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

        pt_rect = fitz.Rect(x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom) & page.mediabox
        page.set_cropbox(pt_rect)
        tight_pdf = os.path.join(work_dir, f"{basename}_tight.pdf")
        doc.save(tight_pdf, garbage=4, deflate=True)
        doc.close()
        shutil.move(tight_pdf, pdf_path)
    else:
        doc.close()
    im.close()

    for ext in [".tex", ".aux", ".log"]:
        p = os.path.join(work_dir, f"{basename}{ext}")
        if os.path.exists(p):
            os.remove(p)

    for folder in folders[1:]:
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(pdf_path, os.path.join(folder, f"{basename}.pdf"))
        shutil.copy2(png_path, os.path.join(folder, f"{basename}.png"))
        print(f"Copied vector PDF & 600-DPI PNG for {basename} to {folder}/")

    print(f"Built vector {pdf_path} & 600-DPI {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    target_folders = ["results", "manuscript", "submission_journal_of_energy_storage"]
    compile_vector_and_png(TIKZ_MECHANISM, "fig0_electrochemical_mechanism", target_folders, dpi=600)
    compile_vector_and_png(TIKZ_FRAMEWORK, "fig0_pidkf_framework", target_folders, dpi=600)
