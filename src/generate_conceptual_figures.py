r"""
Generate flagship Q1 journal pure-vector (PDF) and 600-DPI (PNG) conceptual figures
using unified LaTeX TikZ + PGFPlots (100% Computer Modern typography, strictly
pure black text #000000 everywhere, generous typography clearance, zero collisions):

1. fig0_electrochemical_mechanism (.pdf & .png)
2. fig0_pidkf_framework (.pdf & .png)
"""

import os
import shutil
import subprocess
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageChops

TEXLIVE_BIN = r"d:\texlive\2026\bin\windows"
PDFLATEX = os.path.join(TEXLIVE_BIN, "pdflatex.exe")


def build_pgfplots_data():
    """Precompute exact analytical OCV curves, derivatives, and Eq. (15) hysteresis loop."""
    z_nmc = np.linspace(0.0, 1.0, 101)
    v_nmc = (
        9.204 * z_nmc**5
        - 28.841 * z_nmc**4
        + 33.376 * z_nmc**3
        - 17.419 * z_nmc**2
        + 4.674 * z_nmc
        + 3.212
    )
    dv_nmc = (
        46.020 * z_nmc**4
        - 115.364 * z_nmc**3
        + 100.128 * z_nmc**2
        - 34.838 * z_nmc
        + 4.674
    )

    z_lfp = np.linspace(0.12, 0.95, 84)
    v_lfp = (
        3.415
        - 0.286 * z_lfp
        + 1.831 * z_lfp**2
        - 4.120 * z_lfp**3
        + 4.185 * z_lfp**4
        - 1.545 * z_lfp**5
    )
    dv_lfp = (
        -0.286
        + 3.662 * z_lfp
        - 12.360 * z_lfp**2
        + 16.740 * z_lfp**3
        - 7.725 * z_lfp**4
    )

    # Exact Eq. (15) discrete dynamic hysteresis trajectory (beta=45, M0=30 mV)
    beta, m0 = 45.0, 30.0
    z_chg = np.linspace(0.36, 0.64, 57)
    dz = z_chg[1] - z_chg[0]
    vh_chg = np.zeros_like(z_chg)
    vh_chg[0] = -m0
    for k in range(len(z_chg) - 1):
        a_h = np.exp(-beta * dz)
        vh_chg[k + 1] = a_h * vh_chg[k] + (1.0 - a_h) * m0

    z_dis = np.linspace(0.64, 0.36, 57)
    vh_dis = np.zeros_like(z_dis)
    vh_dis[0] = vh_chg[-1]
    for k in range(len(z_dis) - 1):
        a_h = np.exp(-beta * dz)
        vh_dis[k + 1] = a_h * vh_dis[k] + (1.0 - a_h) * (-m0)

    def fmt_coords(xs, ys):
        return " ".join(f"({x:.4f},{y:.5f})" for x, y in zip(xs, ys))

    return {
        "nmc_ocv": fmt_coords(z_nmc * 100.0, v_nmc),
        "lfp_ocv": fmt_coords(z_lfp * 100.0, v_lfp),
        "lfp_env_hi": fmt_coords(z_lfp * 100.0, v_lfp + 0.030),
        "lfp_env_lo": fmt_coords(z_lfp[::-1] * 100.0, (v_lfp - 0.030)[::-1]),
        "nmc_dv": fmt_coords(z_nmc * 100.0, dv_nmc),
        "lfp_dv": fmt_coords(z_lfp * 100.0, dv_lfp),
        "vh_chg": fmt_coords(z_chg * 100.0, vh_chg),
        "vh_dis": fmt_coords(z_dis * 100.0, vh_dis),
    }


def get_mechanism_tex(data):
    """Build Figure 1 TikZ+PGFPlots source at spacious 29.2 cm x 21.8 cm canvas with zero collisions."""
    return r"""\documentclass[10pt]{article}
\usepackage[paperwidth=31.0cm,paperheight=23.6cm,margin=0.6cm]{geometry}
\pagestyle{empty}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,backgrounds,fit,decorations.pathmorphing,shadows}

% Harmonious Q1 Journal Pastel Palette (All typography remains strictly pure black #000000)
\definecolor{cufoilhi}{RGB}{246,214,179}
\definecolor{cufoillo}{RGB}{224,173,124}
\definecolor{alfoilhi}{RGB}{230,236,243}
\definecolor{alfoillo}{RGB}{198,208,220}
\definecolor{anodebg}{RGB}{246,248,251}
\definecolor{graphcore}{RGB}{198,210,224}
\definecolor{graphrim}{RGB}{226,233,242}
\definecolor{seifill}{RGB}{254,238,195}
\definecolor{dendritefill}{RGB}{176,189,205}
\definecolor{elytebg}{RGB}{240,249,255}
\definecolor{sepfill}{RGB}{219,233,246}
\definecolor{nmcbg}{RGB}{238,244,255}
\definecolor{nmcpart}{RGB}{179,206,250}
\definecolor{lfpbg}{RGB}{236,253,245}
\definecolor{lfpshell}{RGB}{167,238,205}
\definecolor{fepcore}{RGB}{254,218,208}
\definecolor{thermbg}{RGB}{254,242,242}
\definecolor{arrhbg}{RGB}{255,250,235}
\definecolor{capbg}{RGB}{236,253,245}
\definecolor{hdrslate}{RGB}{229,235,242}
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
    panelbox/.style={draw=black, line width=1.0pt, rounded corners=4pt, fill=white},
    card/.style={draw=black, line width=0.7pt, rounded corners=2.5pt, fill=white, drop shadow={opacity=0.07, shadow xshift=0.9pt, shadow yshift=-0.9pt}},
    arr/.style={-{Latex[length=2.2mm,width=1.55mm]}, line width=0.9pt, draw=black},
    darr/.style={-{Latex[length=2.0mm,width=1.45mm]}, line width=0.8pt, draw=black, densely dashed},
    resistor/.style={decorate, decoration={zigzag, segment length=2.2mm, amplitude=1.0mm}, line width=0.98pt}
]

% =========================================================================
% PANEL (a): Top-Left (x = 0.0 to 14.40, y = 10.60 to 21.80)
% =========================================================================
\draw[panelbox] (0.0, 10.60) rectangle (14.40, 21.80);
\node[anchor=west, font=\fontsize{10.4}{12}\selectfont\bfseries] at (0.24, 21.40)
    {(a) Multi-Scale Electrode Degradation \& Electro-Thermal-Aging Loop};

% External Load Electron Circuit Arch (y = 20.62 to 21.14)
\draw[line width=0.85pt] (0.44, 20.52) -- (0.44, 20.92) -- (13.96, 20.92) -- (13.96, 20.52);
\node[draw=black, line width=0.65pt, fill=hdrslate, rounded corners=2pt, inner sep=2.6pt, font=\fontsize{7.4}{8.5}\selectfont\bfseries]
    at (7.20, 20.92) {External Circuit ($I > 0$: $e^- \longrightarrow$ Cathode)};
\draw[arr, line width=0.85pt] (2.10, 20.92) -- (3.30, 20.92);
\draw[arr, line width=0.85pt] (11.00, 20.92) -- (12.20, 20.92);

% Main Cell Cutaway Outer Frame (x = 0.22 to 14.18, y = 13.95 to 20.52)
\draw[draw=black, line width=0.75pt, rounded corners=2.5pt, fill=white] (0.22, 13.95) rectangle (14.18, 20.52);

% Header Strips (y = 20.02 to 20.52) — comfortably wide!
\filldraw[fill=hdrslate, draw=black, line width=0.7pt] (0.22, 20.02) rectangle (4.86, 20.52);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (2.54, 20.27) {Cu $|$ Anode ($\mathrm{Li_xC_6}$)};

\filldraw[fill=hdrblue, draw=black, line width=0.7pt] (4.86, 20.02) rectangle (9.24, 20.52);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (7.05, 20.27) {Electrolyte \& Separator};

\filldraw[fill=hdrgreen, draw=black, line width=0.7pt] (9.24, 20.02) rectangle (14.18, 20.52);
\node[font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (11.71, 20.27) {Cathode (NMC / LFP) $|$ Al};

% 1. Metallic Shaded Cu Collector (x = 0.22 to 0.66)
\shade[top color=cufoilhi, bottom color=cufoillo] (0.22, 13.95) rectangle (0.66, 20.02);
\draw[line width=0.65pt] (0.22, 13.95) rectangle (0.66, 20.02);
\node[rotate=90, font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (0.44, 16.985) {Cu Foil ($-$)};

% 2. Porous Composite Graphite Anode Zone (x = 0.66 to 4.86)
\filldraw[fill=anodebg, draw=black, line width=0.65pt] (0.66, 13.95) rectangle (4.86, 20.02);

\filldraw[fill=seifill!65, draw=black!50, line width=0.45pt, densely dashed] (1.16, 19.34) circle (0.38cm);
\shade[outer color=graphcore, inner color=graphrim] (1.16, 19.34) circle (0.28cm);
\draw[line width=0.45pt, draw=black!75] (1.16, 19.34) circle (0.28cm);

\filldraw[fill=seifill!65, draw=black!50, line width=0.45pt, densely dashed] (1.16, 16.44) circle (0.38cm);
\shade[outer color=graphcore, inner color=graphrim] (1.16, 16.44) circle (0.28cm);
\draw[line width=0.45pt, draw=black!75] (1.16, 16.44) circle (0.28cm);

% Hero Top Anode Particle with Bilayer SEI & Graphene Intercalation Galleries
\filldraw[fill=seifill, draw=black, line width=0.75pt, densely dashed] (2.78, 18.20) circle (0.94cm);
\shade[outer color=graphcore, inner color=graphrim] (2.78, 18.20) circle (0.70cm);
\draw[line width=0.8pt] (2.78, 18.20) circle (0.70cm);
\foreach \dy in {-0.28, 0.0, 0.28} {
    \draw[line width=0.45pt, draw=black!75] (2.22, 18.20+\dy) -- (3.34, 18.20+\dy);
}
\node[fill=white, fill opacity=0.92, text opacity=1, inner sep=1.2pt, rounded corners=1.2pt, align=center, font=\fontsize{7.4}{8.4}\selectfont\bfseries]
    at (2.78, 18.20) {$\mathrm{Li_xC_6}$\\[-1pt]$D_s(T)$};
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1.2pt, inner sep=1.6pt, font=\fontsize{7.0}{8.0}\selectfont\bfseries]
    at (2.94, 19.52) {Porous SEI ($\delta_{\mathrm{SEI}}$)};

\node[draw=black, line width=0.55pt, fill=white, rounded corners=2pt, align=left, font=\fontsize{6.8}{8.2}\selectfont, inner sep=2.2pt]
    (seibadge) at (3.24, 16.36) {
        $\bullet$ Traps $\mathrm{Li^+}$ (LLI)\\
        $\bullet$ $\uparrow \Delta R_{\mathrm{SEI}}$ \& fade
    };
\draw[->, line width=0.55pt] (seibadge.north) -- (3.24, 17.28);

% Bottom Anode Particle + Realistic Multi-Branch Cold Li0 Plating Dendrites
\filldraw[fill=seifill, draw=black, line width=0.65pt, densely dashed] (1.44, 14.84) circle (0.50cm);
\shade[outer color=graphcore, inner color=graphrim] (1.44, 14.84) circle (0.37cm);
\draw[line width=0.75pt] (1.44, 14.84) circle (0.37cm);
\node[font=\fontsize{6.8}{7.8}\selectfont\bfseries] at (1.44, 14.84) {$\mathrm{Li_xC_6}$};

\filldraw[fill=dendritefill, draw=black, line width=0.55pt]
    (1.90, 15.06) -- (2.24, 15.26) -- (2.14, 15.06) -- (2.50, 15.18) -- (2.26, 14.92)
    -- (2.48, 14.72) -- (2.12, 14.78) -- (2.20, 14.56) -- (1.90, 14.68) -- cycle;
\node[draw=black, line width=0.5pt, fill=hdrrose, rounded corners=1.5pt, inner sep=2.0pt, align=center, font=\fontsize{6.8}{8.0}\selectfont\bfseries]
    (libadge) at (3.72, 14.84) {Cold $\mathrm{Li}^0$\\Plating ($0^\circ\mathrm{C}$)};
\draw[->, line width=0.55pt] (libadge.west) -- (2.50, 14.84);

% 3. Electrolyte, Desolvation & Microporous Separator (x = 4.86 to 9.24, width 4.38 cm)
\shade[left color=elytebg, right color=elytebg!65!white] (4.86, 13.95) rectangle (9.24, 20.02);
\draw[line width=0.65pt] (4.86, 13.95) rectangle (9.24, 20.02);

\node[fill=white, draw=black, line width=0.45pt, rounded corners=1.2pt, inner sep=1.4pt, font=\fontsize{7.0}{8.0}\selectfont\bfseries]
    at (7.05, 19.52) {Separator};
\filldraw[fill=sepfill, draw=black, line width=0.55pt, densely dashed] (6.87, 16.40) rectangle (7.23, 19.16);
\foreach \yy in {16.54, 16.96, 17.38, 17.80, 18.22, 18.64} {
    \draw[line width=0.5pt, fill=white] (6.94, \yy) rectangle (7.16, \yy+0.18);
}

\filldraw[fill=nmcpart, draw=black, line width=0.65pt] (5.50, 18.30) circle (0.16cm);
\node[font=\fontsize{6.3}{7.3}\selectfont\bfseries] at (5.50, 18.30) {$\mathrm{Li^+}$};
\foreach \ang in {40, 140, 220, 320} {
    \filldraw[fill=hdramber, draw=black, line width=0.45pt] ($(5.50,18.30)+(\ang:0.26cm)$) circle (0.07cm);
}
\node[font=\fontsize{6.8}{7.8}\selectfont\bfseries] at (5.60, 18.84) {$\mathrm{Li^+(EC)_4}$};

\draw[arr, line width=0.95pt] (5.84, 18.30) -- (8.94, 18.30);
\filldraw[fill=hdrgreen, draw=black, line width=0.65pt] (8.18, 18.30) circle (0.16cm);
\node[font=\fontsize{6.3}{7.3}\selectfont\bfseries] at (8.18, 18.30) {$\mathrm{Li^+}$};
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1.2pt, inner sep=1.3pt, font=\fontsize{6.5}{7.5}\selectfont\bfseries]
    at (8.22, 18.82) {Discharge ($I\!>\!0$)};

\draw[darr, line width=0.85pt] (8.94, 17.24) -- (5.14, 17.24);
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1.2pt, inner sep=1.3pt, font=\fontsize{6.5}{7.5}\selectfont]
    at (7.05, 16.74) {Charge ($I < 0$)};

% Butler-Volmer + Desolvation Card inside Electrolyte (two clean lines so zero edge touch!)
\draw[card] (5.04, 14.12) rectangle (9.06, 16.16);
\node[align=center, font=\fontsize{6.7}{8.0}\selectfont] at (7.05, 15.14) {
    \textbf{Butler--Volmer Kinetics}\\[1.5pt]
    $\eta_{\mathrm{act}} = \dfrac{2RT}{F}\mathrm{arcsinh}\!\left(\dfrac{I}{2 A i_0}\right)$\\[2pt]
    Cold Desolvation Barrier:\\
    $E_{a,1}^{\mathrm{cold}} = 28.4\;\mathrm{kJ/mol}$
};

% 4. Dual Cathode Comparison (x = 9.24 to 13.74, width 4.50 cm!) + Al Foil (13.74 to 14.18)
% Upper: NMC811
\filldraw[fill=nmcbg, draw=black, line width=0.65pt] (9.24, 16.98) rectangle (13.74, 20.02);
\shade[outer color=nmcpart, inner color=white] (9.94, 18.80) circle (0.44cm);
\draw[line width=0.75pt] (9.94, 18.80) circle (0.44cm);
\foreach \dy in {-0.22, -0.11, 0.0, 0.11, 0.22} {
    \draw[line width=0.45pt, draw=nmcline] (9.60, 18.80+\dy) -- (10.28, 18.80+\dy);
}
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.1pt, align=center, font=\fontsize{5.8}{6.7}\selectfont\bfseries]
    (onelbl) at (9.94, 17.64) {1-Phase\\[-1pt]Host};
\draw[->, line width=0.45pt] (onelbl.north) -- (9.94, 18.34);
\node[anchor=west, align=left, font=\fontsize{6.6}{8.0}\selectfont] at (10.58, 18.52) {
    \textbf{NMC811} (\texttt{INR20R})\\[1pt]
    Solid-solution host\\[0.5pt]
    $V_{\mathrm{ocv}}'\!\in\![0.481, 4.674]\;\mathrm{V}$\\[0.5pt]
    $\mathrm{rank}(\mathcal{O}_k) = 3$
};

% Lower: LFP Core-Shell
\filldraw[fill=lfpbg, draw=black, line width=0.65pt] (9.24, 13.95) rectangle (13.74, 16.98);
\shade[outer color=lfpshell, inner color=lfpshell!55] (9.92, 15.72) circle (0.46cm);
\draw[line width=0.75pt] (9.92, 15.72) circle (0.46cm);
\filldraw[fill=fepcore, draw=black, line width=0.65pt, densely dashed] (9.92, 15.72) circle (0.23cm);
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.1pt, align=center, font=\fontsize{5.8}{6.8}\selectfont\bfseries]
    (corelbl) at (9.92, 14.56) {Core: $\mathrm{FePO_4}$\\[-1pt]Shell: $\mathrm{LiFePO_4}$};
\draw[->, line width=0.45pt] (corelbl.north) -- (9.92, 15.48);
\node[anchor=west, align=left, font=\fontsize{6.6}{8.0}\selectfont] at (10.56, 15.46) {
    \textbf{$\mathrm{LiFePO_4}$} (\texttt{ANR26650})\\[1pt]
    Biphasic ($F'\!=\!C\!-\!P\!=\!0$)\\[0.5pt]
    $V_{\mathrm{ocv}}'\!\approx\!0.08\;\mathrm{V} + V_h$\\[0.5pt]
    $\kappa(\mathcal{O}_k) > 1.99\times 10^5$
};

% Brushed Aluminum Collector
\shade[top color=alfoilhi, bottom color=alfoillo] (13.74, 13.95) rectangle (14.18, 20.02);
\draw[line width=0.65pt] (13.74, 13.95) rectangle (14.18, 20.02);
\node[rotate=90, font=\fontsize{8.2}{9.5}\selectfont\bfseries] at (13.96, 16.985) {Al Foil ($+$)};

% --- Bottom Section of Panel (a): Spacious Closed Electro-Thermal-Aging Loop (y = 10.76 to 13.78) ---
% Loop width 13.96 cm -> 3 cards each 4.24 cm wide, clean margins!
\draw[draw=black, line width=0.7pt, rounded corners=2.5pt, fill=anodebg] (0.22, 10.76) rectangle (14.18, 13.78);
\filldraw[fill=hdrslate, draw=black, line width=0.65pt] (0.22, 13.30) rectangle (14.18, 13.78);
\node[font=\fontsize{8.4}{9.6}\selectfont\bfseries] at (7.20, 13.54)
    {Closed Electro-Thermal-Aging Loop (Arrhenius Kinetics \& Energy Balance)};

\draw[card, fill=thermbg] (0.38, 11.38) rectangle (4.60, 13.16);
\node[align=center, font=\fontsize{6.4}{7.8}\selectfont] at (2.49, 12.27) {
    \textbf{Thermal Balance (Eq.\ 5)}\\[1.5pt]
    $C_{\mathrm{th}}\dfrac{dT}{dt} = I^2 R_0 + I(U_1+U_2)$\\[1.5pt]
    $- IT\dfrac{\partial V_{\mathrm{ocv}}}{\partial T} - \dfrac{T-T_{\mathrm{amb}}}{R_{\mathrm{th}}}$
};

\draw[card, fill=arrhbg] (5.10, 11.38) rectangle (9.32, 13.16);
\node[align=center, font=\fontsize{6.4}{7.8}\selectfont] at (7.21, 12.27) {
    \textbf{Arrhenius \& Aging (Eqs.\ 8--10)}\\[2pt]
    $R_i(T,\mathrm{SOH}) = R_{i,\mathrm{nom}}\,f_{T,i}(T)$\\[2pt]
    $\times\,[1 + \kappa_{R_i}(1-\mathrm{SOH})]$
};

\draw[card, fill=capbg] (9.82, 11.38) rectangle (14.02, 13.16);
\node[align=center, font=\fontsize{6.4}{7.8}\selectfont] at (11.92, 12.27) {
    \textbf{Usable Capacity (Eq.\ 10)}\\[2pt]
    $Q_{\mathrm{act}} = \mathrm{SOH}\cdot Q_{\mathrm{init}}\,f_Q(T)$\\[2pt]
    $0.890\times\,(0^\circ\mathrm{C}),\;\; 1.030\times\,(45^\circ\mathrm{C})$
};

\draw[arr, line width=0.85pt] (4.60, 12.27) -- (5.10, 12.27);
\node[font=\fontsize{7.0}{8.0}\selectfont\bfseries] at (4.85, 12.52) {$T_k$};
\draw[arr, line width=0.85pt] (9.32, 12.27) -- (9.82, 12.27);
\node[font=\fontsize{7.0}{8.0}\selectfont\bfseries] at (9.57, 12.52) {$R_i$};

\draw[arr, line width=0.85pt] (11.92, 11.38) -- (11.92, 10.98) -- (2.49, 10.98) -- (2.49, 11.38);
\node[fill=anodebg, inner sep=2.0pt, font=\fontsize{7.0}{8.0}\selectfont\bfseries] at (7.20, 10.98)
    {Joule Heat \& SEI Feedback ($I^2 R_0,\, \Delta R_{\mathrm{SEI}},\, \mathrm{SOH}$)};


% =========================================================================
% PANEL (b): Top-Right (x = 14.76 to 29.20, y = 10.60 to 21.80)
% =========================================================================
\draw[panelbox] (14.76, 10.60) rectangle (29.20, 21.80);
\node[anchor=west, font=\fontsize{10.4}{12}\selectfont\bfseries] at (14.98, 21.40)
    {(b) Augmented 2nd-Order RC Circuit \& Parameter Allocation};

% Circuit Diagram Box (x = 14.98 to 28.98, width 14.00 cm!, y = 13.95 to 20.92)
\draw[draw=black, line width=0.75pt, rounded corners=2.5pt, fill=white] (14.98, 13.95) rectangle (28.98, 20.92);

\coordinate (bot_l) at (15.86, 15.18);
\coordinate (top_l) at (15.86, 18.78);

\draw[line width=0.95pt] (bot_l) -- (15.86, 15.64);
\filldraw[fill=hdrblue, draw=black, line width=0.9pt] (15.86, 16.16) circle (0.48cm);
\node[font=\fontsize{7.8}{9.0}\selectfont\bfseries] at (15.86, 16.16) {$V_{\mathrm{ocv}}(z)$};
\node[font=\fontsize{8.8}{9.8}\selectfont\bfseries] at (15.26, 16.50) {$+$};
\node[font=\fontsize{8.8}{9.8}\selectfont\bfseries] at (15.26, 15.80) {$-$};
\draw[line width=0.95pt] (15.86, 16.64) -- (15.86, 17.02);

% Diamond Dependent Hysteresis Voltage Source
\filldraw[fill=hdramber, draw=black, line width=0.85pt]
    (15.86, 18.18) -- (16.40, 17.60) -- (15.86, 17.02) -- (15.32, 17.60) -- cycle;
\node[align=center, font=\fontsize{7.2}{8.2}\selectfont\bfseries] at (15.86, 17.60) {$\pm V_{h,k}$\\[-2pt]\fontsize{6.0}{7.0}\selectfont (LFP)};
\draw[line width=0.95pt] (15.86, 18.18) -- (top_l);

% R0(T, SOH) + DEKF pill cleanly placed UNDER R0 (completely eliminates top-label overlap!)
\draw[line width=0.95pt] (top_l) -- (16.58, 18.78);
\draw[resistor] (16.58, 18.78) -- (17.80, 18.78);
\draw[line width=0.95pt] (17.80, 18.78) -- (18.42, 18.78);
\filldraw[fill=black] (18.42, 18.78) circle (1.6pt);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (17.19, 19.34) {$R_0(T,\mathrm{SOH})$};
\node[draw=black, line width=0.5pt, fill=hdrblue, rounded corners=1.8pt, inner sep=1.6pt, font=\fontsize{6.5}{7.5}\selectfont\bfseries]
    at (17.20, 17.98) {DEKF Tracks $\hat{R}_0$};

% RC1 Parallel Branch (Centered cleanly at x = 19.86)
\draw[line width=0.9pt] (18.42, 18.08) -- (18.42, 19.48);
\draw[line width=0.9pt] (21.30, 18.08) -- (21.30, 19.48);
\filldraw[fill=black] (21.30, 18.78) circle (1.6pt);
\draw[line width=0.9pt] (18.42, 19.48) -- (19.04, 19.48);
\draw[resistor] (19.04, 19.48) -- (20.68, 19.48);
\draw[line width=0.9pt] (20.68, 19.48) -- (21.30, 19.48);
\node[font=\fontsize{7.4}{8.5}\selectfont\bfseries] at (19.86, 20.02) {$R_1$ (Charge Transfer)};
\draw[line width=0.9pt] (18.42, 18.08) -- (19.72, 18.08);
\draw[line width=1.35pt] (19.72, 17.80) -- (19.72, 18.36);
\draw[line width=1.35pt] (20.00, 17.80) -- (20.00, 18.36);
\draw[line width=0.9pt] (20.00, 18.08) -- (21.30, 18.08);
\node[align=center, font=\fontsize{6.9}{8.0}\selectfont] at (19.86, 17.12)
    {$C_1$ ($\tau_1 = 24.0 / 37.5\;\mathrm{s}$)\\[1pt]$+\;\; U_{1,k}\;\; -$};

\draw[line width=0.95pt] (21.30, 18.78) -- (21.92, 18.78);
\filldraw[fill=black] (21.92, 18.78) circle (1.6pt);

% RC2 Parallel Branch (Centered cleanly at x = 23.36 — 3.50 cm spacing!)
\draw[line width=0.9pt] (21.92, 18.08) -- (21.92, 19.48);
\draw[line width=0.9pt] (24.80, 18.08) -- (24.80, 19.48);
\filldraw[fill=black] (24.80, 18.78) circle (1.6pt);
\draw[line width=0.9pt] (21.92, 19.48) -- (22.54, 19.48);
\draw[resistor] (22.54, 19.48) -- (24.18, 19.48);
\draw[line width=0.9pt] (24.18, 19.48) -- (24.80, 19.48);
\node[font=\fontsize{7.4}{8.5}\selectfont\bfseries] at (23.36, 20.02) {$R_2$ (Warburg Diffusion)};
\draw[line width=0.9pt] (21.92, 18.08) -- (23.22, 18.08);
\draw[line width=1.35pt] (23.22, 17.80) -- (23.22, 18.36);
\draw[line width=1.35pt] (23.50, 17.80) -- (23.50, 18.36);
\draw[line width=0.9pt] (23.50, 18.08) -- (24.80, 18.08);
\node[align=center, font=\fontsize{6.9}{8.0}\selectfont] at (23.36, 17.12)
    {$C_2$ ($\tau_2 = 112.5 / 144.0\;\mathrm{s}$)\\[1pt]$+\;\; U_{2,k}\;\; -$};

% Controlled Dependent Diamond Neural Source: delta V_{t,comp,k} (Centered cleanly at x = 26.40)
\draw[line width=0.95pt] (24.80, 18.78) -- (25.44, 18.78);
\filldraw[fill=hdrgreen, draw=black, line width=0.9pt, drop shadow={opacity=0.07, shadow xshift=0.8pt, shadow yshift=-0.8pt}]
    (26.40, 19.48) -- (27.36, 18.78) -- (26.40, 18.08) -- (25.44, 18.78) -- cycle;
\node[align=center, font=\fontsize{6.9}{8.0}\selectfont\bfseries] at (26.40, 18.78)
    {$\delta\hat{V}_{t,\mathrm{comp},k}$\\[-1pt]MLP};
\node[font=\fontsize{8.0}{8.8}\selectfont\bfseries] at (25.34, 18.42) {$-$};
\node[font=\fontsize{8.0}{8.8}\selectfont\bfseries] at (27.44, 18.42) {$+$};
\draw[darr, line width=0.75pt] (26.40, 19.92) -- (26.40, 19.48);
\node[draw=black, line width=0.45pt, fill=white, rounded corners=1.5pt, inner sep=1.5pt, font=\fontsize{6.5}{7.5}\selectfont\bfseries]
    at (26.40, 20.16) {Passivity: $I_k\cdot\delta\hat{V} \le 0$};
\node[font=\fontsize{6.6}{7.6}\selectfont] at (26.40, 17.78) {$\mathcal{NN}(\mathbf{u}_k)$ Control};
\draw[line width=0.95pt] (27.36, 18.78) -- (28.14, 18.78);

% Output Terminals + Vertical V_{t,k} Span Indicator + I_k > 0
\draw[line width=0.95pt] (bot_l) -- (28.14, 15.18);
\filldraw[fill=white, draw=black, line width=0.95pt] (28.14, 18.78) circle (0.10cm);
\filldraw[fill=white, draw=black, line width=0.95pt] (28.14, 15.18) circle (0.10cm);
\node[font=\fontsize{8.8}{9.8}\selectfont\bfseries] at (28.14, 19.12) {$+$};
\node[font=\fontsize{8.8}{9.8}\selectfont\bfseries] at (28.14, 14.84) {$-$};
\draw[{Latex[length=1.5mm]}-{Latex[length=1.5mm]}, line width=0.7pt, black!75] (28.14, 15.34) -- (28.14, 18.62);
\node[fill=white, inner sep=1.8pt, font=\fontsize{8.8}{10}\selectfont\bfseries] at (28.14, 16.98) {$V_{t,k}$};
\draw[arr, line width=0.9pt] (27.76, 19.50) -- (28.56, 19.50);
\node[font=\fontsize{7.0}{8.0}\selectfont\bfseries] at (28.16, 19.80) {$I_k > 0$};

% KVL Bar
\filldraw[fill=anodebg, draw=black, line width=0.6pt, rounded corners=1.8pt] (15.18, 14.12) rectangle (28.78, 14.66);
\node[font=\fontsize{7.2}{8.4}\selectfont] at (21.98, 14.39) {
    \textbf{KVL:} $\hat{V}_{t,\mathrm{PIDKF},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - \hat{V}_{h,k}^- - I_k\hat{R}_{0,k}^- + \delta\hat{V}_{t,\mathrm{comp},k}$ \;\;($\hat{V}_{h,k}^-\!=\!0$ NMC)
};

% Structural Identifiability Matrix (y = 10.76 to 13.78, width 14.00 cm -> 3.30 | 6.20 | 4.50 cm!)
\draw[draw=black, line width=0.7pt, rounded corners=2.5pt, fill=white] (14.98, 10.76) rectangle (28.98, 13.78);
\filldraw[fill=hdrslate, draw=black, line width=0.65pt] (14.98, 13.30) rectangle (28.98, 13.78);
\node[font=\fontsize{8.4}{9.6}\selectfont\bfseries] at (21.98, 13.54)
    {Observer Parameter Allocation \& Structural Identifiability Matrix};

\filldraw[fill=anodebg, draw=black, line width=0.55pt] (14.98, 12.82) rectangle (18.18, 13.30);
\filldraw[fill=anodebg, draw=black, line width=0.55pt] (18.18, 12.82) rectangle (24.12, 13.30);
\filldraw[fill=anodebg, draw=black, line width=0.55pt] (24.12, 12.82) rectangle (28.98, 13.30);
\node[font=\fontsize{7.2}{8.4}\selectfont\bfseries] at (16.58, 13.06) {Observer Class};
\node[font=\fontsize{7.2}{8.4}\selectfont\bfseries] at (21.15, 13.06) {Online Parameter \& Residual Ownership};
\node[font=\fontsize{7.2}{8.4}\selectfont\bfseries] at (26.55, 13.06) {Identifiability / Drift Impact};

\draw[draw=black, line width=0.5pt] (14.98, 12.14) rectangle (18.18, 12.82);
\draw[draw=black, line width=0.5pt] (18.18, 12.14) rectangle (24.12, 12.82);
\draw[draw=black, line width=0.5pt] (24.12, 12.14) rectangle (28.98, 12.82);
\node[align=center, font=\fontsize{6.6}{7.8}\selectfont\bfseries] at (16.58, 12.48) {Fixed-Prior Filters\\(EKF, AEKF, SMO)};
\node[align=center, font=\fontsize{6.5}{7.7}\selectfont] at (21.15, 12.48) {Freeze fresh $(R_0, R_1, R_2, C_1, C_2, Q_{\mathrm{init}})$};
\node[align=center, font=\fontsize{6.5}{7.7}\selectfont] at (26.55, 12.48) {$45.7\times$ EKF surge ($80\%$ SOH)};

\filldraw[fill=nmcbg, draw=black, line width=0.5pt] (14.98, 11.45) rectangle (18.18, 12.14);
\filldraw[fill=nmcbg, draw=black, line width=0.5pt] (18.18, 11.45) rectangle (24.12, 12.14);
\filldraw[fill=nmcbg, draw=black, line width=0.5pt] (24.12, 11.45) rectangle (28.98, 12.14);
\node[align=center, font=\fontsize{6.6}{7.8}\selectfont\bfseries] at (16.58, 11.795) {Dual EKF (DEKF)};
\node[align=center, font=\fontsize{6.4}{7.6}\selectfont] at (21.15, 11.795) {Tracks $[\hat{R}_{0,k}, 1/\hat{Q}_{\mathrm{act},k}]^T$; freezes $R_{1,2}, C_{1,2}$};
\node[align=center, font=\fontsize{6.5}{7.7}\selectfont] at (26.55, 11.795) {Avoids collinearity; leaves $V_{\mathrm{pol}}$ bias};

\filldraw[fill=lfpbg, draw=black, line width=0.6pt] (14.98, 10.76) rectangle (18.18, 11.45);
\filldraw[fill=lfpbg, draw=black, line width=0.6pt] (18.18, 10.76) rectangle (24.12, 11.45);
\filldraw[fill=lfpbg, draw=black, line width=0.6pt] (24.12, 10.76) rectangle (28.98, 11.45);
\node[align=center, font=\fontsize{6.6}{7.8}\selectfont\bfseries] at (16.58, 11.105) {Hybrid PIDKF\\(Proposed)};
\node[align=center, font=\fontsize{6.4}{7.6}\selectfont] at (21.15, 11.105) {DEKF tracks $(\hat{R}_{0,k}, \hat{Q}_{\mathrm{act},k})$ + $[16,8]$ MLP};
\node[align=center, font=\fontsize{6.3}{7.4}\selectfont\bfseries] at (26.55, 11.105) {Unbiased $\delta V_{\mathrm{pol}}^*$ under BV passivity};


% =========================================================================
% PANEL (c): Bottom-Left PGFPlots (x = 0.0 to 14.40, y = 0.0 to 10.25)
% =========================================================================
\draw[panelbox] (0.0, 0.0) rectangle (14.40, 10.25);
\node[anchor=west, font=\fontsize{10.4}{12}\selectfont\bfseries] at (0.24, 9.86)
    {(c) Calibrated Open-Circuit Voltage \& LFP Hysteresis};

\begin{axis}[
    at={(1.56cm, 1.18cm)},
    anchor=south west,
    width=12.50cm,
    height=8.12cm,
    xmin=0, xmax=100,
    ymin=3.05, ymax=4.32,
    xlabel={State of Charge, SOC (\%)},
    ylabel={Open-Circuit Potential, $V_{\mathrm{ocv}}$ (V)},
    xlabel style={font=\fontsize{9.2}{11}\selectfont\bfseries, text=black},
    ylabel style={font=\fontsize{9.2}{11}\selectfont\bfseries, text=black},
    tick label style={font=\fontsize{8.5}{10}\selectfont, text=black},
    grid=both,
    grid style={dotted, black!35},
    axis line style={line width=0.85pt, draw=black},
    legend pos=north west,
    legend style={
        font=\fontsize{7.8}{9.2}\selectfont,
        text=black,
        draw=black,
        fill=white,
        fill opacity=0.95,
        cells={anchor=west},
        row sep=1.4pt
    }
]
    \fill[platshade] (axis cs:20, 3.05) rectangle (axis cs:80, 4.32);
    \draw[densely dotted, black!65, line width=0.7pt] (axis cs:20, 3.05) -- (axis cs:20, 4.32);
    \draw[densely dotted, black!65, line width=0.7pt] (axis cs:80, 3.05) -- (axis cs:80, 4.32);

    \addplot[draw=lfpline!70, fill=green!22, line width=0.5pt] coordinates {
        """ + data["lfp_env_hi"] + " " + data["lfp_env_lo"] + r"""
    } -- cycle;
    \addlegendentry{LFP Hysteresis Envelope $\pm M_0$ ($\pm 30\;\mathrm{mV}$)}

    \addplot[color=nmcline, line width=1.65pt] coordinates {
        """ + data["nmc_ocv"] + r"""
    };
    \addlegendentry{NMC811 $V_{\mathrm{ocv}}(z)$ (Eq.\ 11, Monotonic)}

    \addplot[color=lfpline, line width=1.5pt, dashed] coordinates {
        """ + data["lfp_ocv"] + r"""
    };
    \addlegendentry{$\mathrm{LiFePO_4}$ Equilibrium $V_{\mathrm{ocv}}^{\mathrm{LFP}}(z)$}

    % Shifted comfortably up/left so ZERO intersection occurs with bottom-right inset!
    \node[draw=black, line width=0.6pt, fill=white, rounded corners=2pt, align=center, font=\fontsize{7.2}{8.5}\selectfont, inner sep=2.6pt]
        (platbox) at (axis cs:33, 3.63) {
            \textbf{Biphasic Plateau (20\%--80\% SOC)}\\
            $\mathrm{LiFePO_4 \leftrightarrow FePO_4}$ ($F' = C - P = 0$)\\
            Flat Slope $\approx 0.08\;\mathrm{V}$ ($0.8\;\mathrm{mV/\%}$)
        };
    \draw[-{Latex[length=1.9mm]}, line width=0.9pt, draw=black] (platbox.south) -- (axis cs:33, 3.455);
\end{axis}

% Authentic Eq. (15) Numerically Simulated Hysteresis Inset cleanly tucked at bottom-right (y <= 3.55 cm -> well BELOW platbox!)
\filldraw[fill=white, draw=black, line width=0.7pt, rounded corners=2.5pt, drop shadow={opacity=0.07, shadow xshift=0.8pt, shadow yshift=-0.8pt}]
    (7.65, 1.32) rectangle (13.92, 3.62);
\node[font=\fontsize{7.1}{8.1}\selectfont\bfseries] at (10.785, 3.43)
    {Eq.\ (15) Hysteresis Loop ($\beta = 45$)};

\begin{axis}[
    at={(8.56cm, 1.62cm)},
    anchor=south west,
    width=5.06cm,
    height=1.62cm,
    scale only axis,
    xmin=35, xmax=65,
    ymin=-38, ymax=38,
    xtick={36, 45, 55, 64},
    xticklabels={36\%, 45\%, 55\%, 64\%},
    ytick={-30, 0, 30},
    yticklabels={$-30$, $0$, $+30$},
    ylabel={$V_h$ (mV)},
    ylabel style={font=\fontsize{6.4}{7.2}\selectfont\bfseries, text=black, yshift=-6pt},
    tick label style={font=\fontsize{6.0}{6.8}\selectfont, text=black},
    grid=major,
    grid style={dotted, black!25},
    axis line style={line width=0.65pt, draw=black},
    axis background/.style={fill=white},
    clip=true
]
    \fill[green!14] (axis cs:35,-30) rectangle (axis cs:65,30);
    \draw[line width=0.6pt, dashed, draw=lfpline] (axis cs:35,0) -- (axis cs:65,0);
    \draw[line width=0.65pt, densely dotted, draw=black] (axis cs:35,30) -- (axis cs:65,30);
    \draw[line width=0.65pt, densely dotted, draw=black] (axis cs:35,-30) -- (axis cs:65,-30);

    \addplot[color=nmcline, line width=1.3pt] coordinates {
        """ + data["vh_chg"] + r"""
    };
    \addplot[color=lfpderiv, line width=1.3pt, dashed] coordinates {
        """ + data["vh_dis"] + r"""
    };
    \draw[-{Latex[length=1.4mm]}, line width=0.75pt, draw=nmcline] (axis cs:43, 18.5) -- (axis cs:51, 27.0);
    \draw[-{Latex[length=1.4mm]}, line width=0.75pt, draw=lfpderiv] (axis cs:57, -18.5) -- (axis cs:49, -27.0);
    \node[font=\fontsize{6.0}{6.8}\selectfont\bfseries, anchor=west] at (axis cs:49.5, 22) {Charge};
    \node[font=\fontsize{6.0}{6.8}\selectfont\bfseries, anchor=east] at (axis cs:50.5, -22) {Discharge};
\end{axis}


% =========================================================================
% PANEL (d): Bottom-Right PGFPlots (x = 14.76 to 29.20, y = 0.0 to 10.25)
% =========================================================================
\draw[panelbox] (14.76, 0.0) rectangle (29.20, 10.25);
\node[anchor=west, font=\fontsize{10.4}{12}\selectfont\bfseries] at (14.98, 9.86)
    {(d) Analytical Observability Slope $dV_{\mathrm{ocv}}/dz$ \& Condition Surge};

\begin{axis}[
    at={(16.40cm, 1.18cm)},
    anchor=south west,
    width=12.46cm,
    height=8.12cm,
    ymode=log,
    xmin=0, xmax=100,
    ymin=0.0018, ymax=9.5,
    xlabel={State of Charge, SOC (\%)},
    ylabel={OCV Slope, $dV_{\mathrm{ocv}}/dz$ (V)},
    xlabel style={font=\fontsize{9.2}{11}\selectfont\bfseries, text=black},
    ylabel style={font=\fontsize{9.2}{11}\selectfont\bfseries, text=black},
    tick label style={font=\fontsize{8.5}{10}\selectfont, text=black},
    grid=major,
    grid style={dotted, black!35},
    axis line style={line width=0.85pt, draw=black},
    legend pos=north east,
    legend style={
        font=\fontsize{7.8}{9.2}\selectfont,
        text=black,
        draw=black,
        fill=white,
        fill opacity=0.95,
        cells={anchor=west},
        row sep=1.4pt
    }
]
    \fill[platshade] (axis cs:20, 0.0018) rectangle (axis cs:80, 9.5);
    \draw[densely dotted, black!65, line width=0.7pt] (axis cs:20, 0.0018) -- (axis cs:20, 9.5);
    \draw[densely dotted, black!65, line width=0.7pt] (axis cs:80, 0.0018) -- (axis cs:80, 9.5);

    \addplot[color=nmcline, line width=1.65pt] coordinates {
        """ + data["nmc_dv"] + r"""
    };
    \addlegendentry{NMC Slope $V_{\mathrm{ocv}}'(z) \in [0.481, 4.674]\;\mathrm{V}$}

    \addplot[color=lfpderiv, line width=1.65pt, dash pattern=on 6pt off 2.2pt on 1.4pt off 2.2pt] coordinates {
        """ + data["lfp_dv"] + r"""
    };
    \addlegendentry{LFP Slope $V_{\mathrm{ocv}}'(z)$ ($\approx 0.08\;\mathrm{V}$ on Plateau)}

    \draw[densely dotted, black, line width=0.8pt] (axis cs:0, 0.08) -- (axis cs:100, 0.08);

    \filldraw[fill=nmcline, draw=black, line width=0.55pt] (axis cs:50, 0.74275) circle (2.3pt);
    \filldraw[fill=lfpderiv, draw=black, line width=0.55pt] (axis cs:50, 0.06469) circle (2.3pt);
    \draw[{Latex[length=1.8mm]}-{Latex[length=1.8mm]}, line width=0.95pt, draw=black]
        (axis cs:50, 0.072) -- (axis cs:50, 0.675);

    \node[draw=black, line width=0.6pt, fill=white, rounded corners=2pt, inner sep=2.4pt, align=center, font=\fontsize{6.9}{8.2}\selectfont\bfseries]
        (gapbox) at (axis cs:65.5, 0.25) {Plateau Collapse (@ 50\%)\\$0.743 \to 0.065\;\mathrm{V}$ ($11.5\times$)\\(Mean $V_{\mathrm{ocv}}' \approx 0.08\;\mathrm{V}$)};
    \draw[-{Latex[length=1.7mm]}, line width=0.8pt, draw=black] (gapbox.west) -- (axis cs:50.6, 0.25);

    \filldraw[fill=lfpderiv, draw=black, line width=0.55pt] (axis cs:12, 0.00278) circle (2.3pt);
    \node[draw=black, line width=0.55pt, fill=white, rounded corners=1.8pt, inner sep=2.2pt, align=left, font=\fontsize{7.0}{8.2}\selectfont\bfseries, anchor=west]
        (cutbox) at (axis cs:22.0, 0.0029) {BMS Cutoff $z\!=\!0.12$\\$V_{\mathrm{ocv}}' = 0.00278\;\mathrm{V}$};
    \draw[-{Latex[length=1.6mm]}, line width=0.8pt, draw=black] (cutbox.west) -- (axis cs:12.6, 0.00278);

    \node[draw=black, line width=0.6pt, fill=white, rounded corners=2pt, align=left, font=\fontsize{7.1}{8.5}\selectfont, inner sep=2.6pt]
        at (axis cs:50, 0.0155) {
            \textbf{Observability:} $|\det(\mathcal{O}_k)| \propto V_{\mathrm{ocv}}'(z_k)$ (Eqs.\ 12, 16)\\[1pt]
            $\bullet$ NMC811: $\kappa(\mathcal{O}_k)\approx 3.2\times 10^4 \Rightarrow$ Strong Innovation Gain\\
            $\bullet$ LFP: $\kappa(\mathcal{O}_k) > 1.99\times 10^5 \Rightarrow$ Needs $V_h + \delta\hat{V}_{\mathrm{comp}}$
        };
\end{axis}

\end{tikzpicture}
\end{document}
"""


def get_framework_tex():
    """Build Figure 2 TikZ+PGFPlots source at spacious 29.4 cm x 21.8 cm canvas with zero collisions."""
    return r"""\documentclass[10pt]{article}
\usepackage[paperwidth=31.2cm,paperheight=23.6cm,margin=0.6cm]{geometry}
\pagestyle{empty}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,bm}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,calc,backgrounds,fit,shadows}

\definecolor{cplant}{RGB}{241,245,249}
\definecolor{cdekf}{RGB}{239,246,255}
\definecolor{cmlp}{RGB}{236,253,245}
\definecolor{cgate}{RGB}{254,243,199}
\definecolor{cforbid}{RGB}{254,226,226}
\definecolor{cadmit}{RGB}{187,247,208}
\definecolor{hdrslate}{RGB}{229,235,242}
\definecolor{hdrblue}{RGB}{219,234,254}
\definecolor{hdrgreen}{RGB}{209,250,229}
\definecolor{hdramber}{RGB}{254,243,199}
\definecolor{hdrteal}{RGB}{204,251,241}
\definecolor{barblue}{RGB}{147,197,253}
\definecolor{barslate}{RGB}{148,163,184}
\definecolor{baramber}{RGB}{253,224,71}
\definecolor{bargreen}{RGB}{52,211,153}

\begin{document}\noindent
\begin{tikzpicture}[
    x=1cm, y=1cm,
    >=Latex,
    text=black,
    draw=black,
    font=\normalsize,
    outerbox/.style={draw=black, line width=1.0pt, rounded corners=4pt, fill=white},
    colbox/.style={draw=black, line width=0.8pt, rounded corners=3pt, fill=cplant},
    colhdr/.style={draw=black, line width=0.8pt, font=\fontsize{8.8}{10.2}\selectfont\bfseries, align=center},
    subcard/.style={draw=black, line width=0.7pt, rounded corners=2.5pt, fill=white, align=center, drop shadow={opacity=0.07, shadow xshift=0.9pt, shadow yshift=-0.9pt}},
    mblk/.style={draw=black, line width=0.6pt, rounded corners=2pt, fill=cdekf!65, align=center},
    phasecard/.style={draw=black, line width=0.65pt, rounded corners=2pt, fill=white, align=center, drop shadow={opacity=0.06, shadow xshift=0.8pt, shadow yshift=-0.8pt}},
    arr/.style={-{Latex[length=2.1mm,width=1.5mm]}, line width=0.9pt, draw=black},
    darr/.style={-{Latex[length=2.0mm,width=1.4mm]}, line width=0.8pt, draw=black, densely dashed}
]

% =========================================================================
% PANEL (a): Closed-Loop Hybrid PIDKF Architecture (y = 8.60 to 21.80)
% =========================================================================
\draw[outerbox] (0.0, 8.60) rectangle (29.40, 21.80);
\node[anchor=west, font=\fontsize{10.8}{12.2}\selectfont\bfseries] at (0.24, 21.40)
    {(a) Closed-Loop Hybrid PIDKF Architecture with Butler--Volmer Passivity and Exogenous Decoupling};

% Three Stage Containers (y = 9.24 to 20.98)
\filldraw[colbox] (0.24, 9.24) rectangle (6.24, 20.98);
\filldraw[colbox, fill=cdekf] (6.84, 9.24) rectangle (17.16, 20.98);
\filldraw[colbox, fill=cmlp] (17.92, 9.24) rectangle (29.16, 20.98);

% --- STAGE 1: Plant, Sensing & Rest Gate (x = 0.24 to 6.24, 6.00 cm wide) ---
\filldraw[colhdr, fill=hdrslate] (0.24, 20.46) rectangle (6.24, 20.98);
\node[font=\fontsize{8.5}{9.8}\selectfont\bfseries] at (3.24, 20.72) {1.\ Plant, Sensing \& Rest Gate};

% 1A: Cell Plant
\draw[subcard] (0.42, 17.56) rectangle (6.06, 19.74);
\filldraw[fill=hdrgreen, draw=black, line width=0.7pt, rounded corners=1.5pt] (0.58, 18.04) rectangle (1.08, 19.16);
\filldraw[fill=hdrslate, draw=black, line width=0.7pt] (0.72, 19.16) rectangle (0.94, 19.30);
\draw[line width=0.55pt] (0.66, 18.36) -- (1.00, 18.36);
\draw[line width=0.55pt] (0.66, 18.76) -- (1.00, 18.76);
\node[font=\fontsize{6.4}{7.3}\selectfont\bfseries] at (0.83, 18.96) {$+$};
\node[font=\fontsize{6.4}{7.3}\selectfont\bfseries] at (0.83, 18.18) {$-$};
\node[anchor=west, align=left, font=\fontsize{6.9}{8.3}\selectfont] at (1.20, 18.65) {
    \textbf{Multi-Chemistry Cell Plant}\\[1.5pt]
    NMC811: \texttt{INR18650-20R} ($2.00\;\mathrm{Ah}$)\\
    LFP: \texttt{ANR26650M1-B} ($2.30\;\mathrm{Ah}$)\\
    $\mathrm{SOH}\in[80\%,100\%],\;T\in[0^\circ,45^\circ\mathrm{C}]$
};

% 1B: Causal BMS Telemetry (1 Hz)
\draw[subcard] (0.42, 13.56) rectangle (6.06, 17.08);
\node[font=\fontsize{7.8}{9.0}\selectfont\bfseries] at (3.24, 16.80) {Causal BMS Telemetry ($1\;\mathrm{Hz}$)};
\draw[line width=0.5pt, draw=black!45] (0.72, 16.04) -- (5.76, 16.04);
\draw[line width=0.9pt, draw=black]
    (0.72, 16.04) -- (1.26, 16.04) -- (1.26, 16.36) -- (2.04, 16.36) -- (2.04, 15.72)
    -- (2.68, 15.72) -- (2.68, 16.22) -- (3.52, 16.22) -- (3.52, 16.04) -- (4.26, 16.04)
    -- (4.26, 16.38) -- (5.16, 16.38) -- (5.16, 16.04) -- (5.76, 16.04);
\node[align=center, font=\fontsize{7.1}{8.5}\selectfont] at (3.24, 14.54) {
    Current: $I_k = I_{\mathrm{true},k} + I_{\mathrm{bias}} + n_{i,k}$\\[1.5pt]
    Voltage: $V_{\mathrm{meas},k} = V_{t,k} + v_k$\\[1.5pt]
    Surface Temperature: $T_k$
};
\draw[arr] (3.24, 17.56) -- (3.24, 17.08);

% 1C: Hard Rest-Period Innovation Gate
\draw[subcard, fill=cgate] (0.42, 9.48) rectangle (6.06, 13.06);
\node[font=\fontsize{7.8}{9.0}\selectfont\bfseries] at (3.24, 12.74) {Hard Rest-Period Innovation Gate};
\draw[line width=0.8pt] (0.86, 11.96) -- (1.46, 11.96);
\draw[line width=1.0pt] (1.46, 11.96) -- (2.06, 12.22);
\filldraw[fill=white, draw=black, line width=0.65pt] (1.46, 11.96) circle (1.4pt);
\filldraw[fill=white, draw=black, line width=0.65pt] (2.10, 11.96) circle (1.4pt);
\draw[line width=0.8pt] (2.10, 11.96) -- (2.56, 11.96);
\node[anchor=west, align=left, font=\fontsize{6.9}{8.0}\selectfont\bfseries] at (2.66, 11.98)
    {$|I_k| < 10\;\mathrm{mA}$: Freeze Gains};
\node[align=center, font=\fontsize{6.8}{8.2}\selectfont] at (3.24, 10.46) {
    Set $\mathbf{K}_k = \mathbf{0},\;\; \mathbf{K}_{\theta,k} = \mathbf{0}$ during rest\\[1.5pt]
    Enforces $\|\mathbf{F}_k\|_2 = \|\mathbf{A}_k\|_2 \le 1$ (Passivity)
};
\draw[arr] (3.24, 13.56) -- (3.24, 13.06);


% --- STAGE 2: Dual EKF Backbone (x = 6.84 to 17.16, 10.32 cm wide — plenty of equation air!) ---
\filldraw[colhdr, fill=hdrblue] (6.84, 20.46) rectangle (17.16, 20.98);
\node[font=\fontsize{8.5}{9.8}\selectfont\bfseries] at (12.00, 20.72) {2.\ Dual EKF Backbone ($980\;\mathrm{FLOPs/Step}$)};

% 2A: Fast State Filter (x = 7.04 to 16.96, 9.92 cm wide, y = 14.34 to 19.74)
\draw[subcard] (7.04, 14.34) rectangle (16.96, 19.74);
\filldraw[fill=hdrblue, draw=black, line width=0.55pt, rounded corners=1.5pt] (7.20, 19.18) rectangle (16.80, 19.62);
\node[font=\fontsize{7.8}{9.0}\selectfont\bfseries] at (12.00, 19.40)
    {Fast State Filter ($\tau_{\mathrm{state}}\sim 10^1\text{--}10^2\;\mathrm{s}$, Joseph Form)};

% 1. Time Update (Predict) — 4.42 cm wide!
\draw[mblk] (7.20, 17.00) rectangle (11.62, 18.96);
\node[align=center, font=\fontsize{6.1}{7.4}\selectfont] at (9.41, 17.98) {
    \textbf{1.\ Time Update (Predict)}\\[2pt]
    $\hat{\mathbf{x}}_k^- = \mathbf{A}_{k-1}\hat{\mathbf{x}}_{k-1}^+ + \mathbf{B}_{k-1}I_{k-1} + \mathbf{g}_{k-1}$\\[1.8pt]
    $\mathbf{P}_k^- = \mathbf{A}_{k-1}\mathbf{P}_{k-1}^+\mathbf{A}_{k-1}^T + \mathbf{Q}$
};

% 2. Joseph Measurement Update — 4.84 cm wide (zero overflow!)
\draw[mblk] (11.96, 17.00) rectangle (16.80, 18.96);
\node[align=center, font=\fontsize{6.0}{7.3}\selectfont] at (14.38, 17.98) {
    \textbf{2.\ Joseph Measurement Update}\\[2pt]
    $\mathbf{K}_k = \mathbf{P}_k^-\mathbf{H}_k^T(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^T + R_{\mathrm{eff},k})^{-1}$\\[1.8pt]
    $\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k e_{k,\mathrm{PIDKF}}$,\; $\mathbf{P}_k^+$ (Joseph)
};
\draw[arr, line width=0.8pt] (11.62, 17.98) -- (11.96, 17.98);

% 3. Nonlinear ECM Output Block
\draw[mblk, fill=cmlp!55] (7.20, 14.54) rectangle (16.80, 16.36);
\node[align=center, font=\fontsize{7.1}{8.5}\selectfont] at (12.00, 15.45) {
    \textbf{3.\ Nonlinear ECM Measurement Model}\\[2pt]
    $\hat{V}_{t,\mathrm{ECM},k}^- = V_{\mathrm{ocv}}(\hat{z}_k^-) - \hat{U}_{1,k}^- - \hat{U}_{2,k}^- - \hat{V}_{h,k}^- - I_k\hat{R}_{0,k}^-$
};
\draw[arr, line width=0.8pt] (9.41, 17.00) -- (9.41, 16.36);
\node[fill=white, inner sep=1pt, font=\fontsize{6.6}{7.6}\selectfont\bfseries] at (9.82, 16.68) {$\hat{\mathbf{x}}_k^-$};

% Timescale Separation Arrows (y = 13.42 to 14.34)
\draw[arr] (9.56, 14.34) -- (9.56, 13.42);
\node[fill=cdekf, inner sep=1pt, font=\fontsize{7.2}{8.2}\selectfont\bfseries] at (9.12, 13.88) {$\hat{z}_k^+$};
\draw[arr] (14.44, 13.42) -- (14.44, 14.34);
\node[fill=cdekf, inner sep=1pt, font=\fontsize{7.2}{8.2}\selectfont\bfseries] at (14.90, 13.88) {$\hat{\boldsymbol{\theta}}_k^+$};
\node[draw=black, line width=0.45pt, fill=white, rounded corners=1.5pt, inner sep=1.6pt, font=\fontsize{6.8}{7.8}\selectfont\bfseries]
    at (12.00, 13.88) {$\tau_{\mathrm{state}} / \tau_{\mathrm{SOH}} \ll 10^{-4}$};

% 2B: Slow Parameter Filter (x = 7.04 to 16.96, y = 9.48 to 13.42)
\draw[subcard] (7.04, 9.48) rectangle (16.96, 13.42);
\filldraw[fill=hdrblue, draw=black, line width=0.55pt, rounded corners=1.5pt] (7.20, 12.86) rectangle (16.80, 13.30);
\node[font=\fontsize{7.8}{9.0}\selectfont\bfseries] at (12.00, 13.08)
    {Slow Parameter Filter ($\tau_{\mathrm{SOH}}\sim 10^5\text{--}10^6\;\mathrm{s}$, Random Walk)};

\node[align=center, font=\fontsize{6.9}{8.3}\selectfont] at (12.00, 11.16) {
    $\hat{\boldsymbol{\theta}}_k = \left[\hat{R}_{0,k},\; 1/\hat{Q}_{\mathrm{act},k}\right]^T$,\quad $\hat{\boldsymbol{\theta}}_k^+ = \hat{\boldsymbol{\theta}}_k^- + \mathbf{K}_{\theta,k} e_{k,\mathrm{PIDKF}}$\\[3pt]
    Fr\'echet Jacobian (Eq.\ 17):\; $\mathbf{C}_{\theta,k} = \left[-I_k,\; -\dfrac{\eta\Delta t I_{k-1}}{3600}V_{\mathrm{ocv}}'(\hat{z}_k^+)\right]$\\[3pt]
    Thermal Inversion:\; $\mathrm{SOH}_{\mathrm{ref},k} \equiv \dfrac{\hat{Q}_{\mathrm{act},k}(T_k)}{Q_{\mathrm{init}}\,f_Q(T_k)} = \dfrac{1}{Q_{\mathrm{init}}\,\hat{\theta}_{2,k}\,f_Q(T_k)}$
};

% Routing Stage 1 -> Stage 2
\draw[arr] (6.06, 15.45) -- (7.20, 15.45);
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{6.5}{7.3}\selectfont\bfseries]
    at (6.54, 15.82) {$I_k, T_k$};
\draw[darr] (6.06, 11.18) -- (7.04, 11.18);
\draw[darr] (6.54, 11.18) |- (7.20, 17.50);
\filldraw[fill=black] (6.54, 11.18) circle (1.4pt);
\node[fill=cgate, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{6.5}{7.3}\selectfont\bfseries]
    at (6.54, 13.40) {Gate};


% --- STAGE 3: Residual MLP, Butler-Volmer Passivity & Innovation Synthesis (x = 17.92 to 29.16) ---
\filldraw[colhdr, fill=hdrgreen] (17.92, 20.46) rectangle (29.16, 20.98);
\node[font=\fontsize{8.5}{9.8}\selectfont\bfseries] at (23.54, 20.72)
    {3.\ Residual MLP Compensator \& Innovation Synthesis};

% Subcard 3A: MLP Topology (x = 18.12 to 23.82, width 5.70 cm!)
\draw[subcard] (18.12, 14.66) rectangle (23.82, 19.74);
\filldraw[fill=hdrgreen, draw=black, line width=0.55pt, rounded corners=1.5pt] (18.24, 19.18) rectangle (23.70, 19.62);
\node[font=\fontsize{6.9}{8.0}\selectfont\bfseries] at (20.97, 19.40)
    {MLP ($6\to 16\to 8\to 1$ GELU, $560\;\mathrm{FLOPs}$)};

\foreach \i/\lbl/\yy in {1/{$I_k$}/18.58, 2/{$T_k$}/18.00, 3/{$\hat{z}_k^-$}/17.42, 4/{$\hat{U}_{1,k}^-$}/16.84, 5/{$\hat{U}_{2,k}^-$}/16.26, 6/{$\mathrm{SOH}_{\mathrm{ref},k}$}/15.68} {
    \filldraw[fill=hdrblue, draw=black, line width=0.6pt] (19.52, \yy) circle (0.12cm);
    \node[anchor=east, font=\fontsize{6.3}{7.2}\selectfont\bfseries] at (19.38, \yy) {\lbl};
}

% Hidden Layer 1: 4 nodes + explicit vertical ellipsis (x = 20.82)
\foreach \yj in {18.36, 17.74, 16.44, 15.82} {
    \filldraw[fill=hdramber, draw=black, line width=0.6pt] (20.82, \yj) circle (0.12cm);
    \foreach \yy in {18.58, 18.00, 17.42, 16.84, 16.26, 15.68} {
        \draw[line width=0.26pt, black!40] (19.64, \yy) -- (20.70, \yj);
    }
}
\node[font=\fontsize{8.5}{8.5}\selectfont\bfseries] at (20.82, 17.10) {$\vdots$};
\node[align=center, font=\fontsize{6.1}{6.8}\selectfont\bfseries] at (20.82, 18.84) {$16$\\[-1.5pt]{\fontsize{5.3}{6.0}\selectfont (GELU)}};

% Hidden Layer 2: 3 nodes + explicit vertical ellipsis (x = 22.12)
\foreach \yk in {18.20, 17.12, 16.04} {
    \filldraw[fill=hdrteal, draw=black, line width=0.6pt] (22.12, \yk) circle (0.12cm);
    \foreach \yj in {18.36, 17.74, 16.44, 15.82} {
        \draw[line width=0.26pt, black!40] (20.94, \yj) -- (22.00, \yk);
    }
}
\node[font=\fontsize{8.5}{8.5}\selectfont\bfseries] at (22.12, 17.66) {$\vdots$};
\node[align=center, font=\fontsize{6.1}{6.8}\selectfont\bfseries] at (22.12, 18.84) {$8$\\[-1.5pt]{\fontsize{5.3}{6.0}\selectfont (GELU)}};

% Output Neuron connected straight through right border to Subcard 3B
\filldraw[fill=hdrgreen, draw=black, line width=0.75pt] (23.26, 17.12) circle (0.16cm);
\foreach \yk in {18.20, 17.12, 16.04} {
    \draw[line width=0.38pt, black!55] (22.24, \yk) -- (23.10, 17.12);
}
\node[font=\fontsize{6.7}{7.6}\selectfont\bfseries] at (23.28, 17.62) {$\mathcal{NN}(\mathbf{u}_k)$};
\draw[arr, line width=0.9pt] (23.42, 17.12) -- (24.16, 17.12);
\node[font=\fontsize{6.4}{7.4}\selectfont\bfseries] at (20.97, 14.94) {$257\;\text{Params} = 1.03\;\mathrm{KB}$ Flash (Zero Leakage)};

% Subcard 3B: Butler-Volmer Passivity Sector (x = 24.16 to 28.96, width 4.80 cm!)
\draw[subcard] (24.16, 14.66) rectangle (28.96, 19.74);
\filldraw[fill=hdrgreen, draw=black, line width=0.55pt, rounded corners=1.5pt] (24.28, 19.18) rectangle (28.84, 19.62);
\node[font=\fontsize{7.2}{8.3}\selectfont\bfseries] at (26.56, 19.40) {Butler--Volmer Passivity Sector};

\fill[cforbid] (26.56, 16.96) rectangle (28.44, 18.28);
\fill[cforbid] (24.68, 15.64) rectangle (26.56, 16.96);
\fill[cadmit] (24.68, 16.96) -- (26.56, 16.96) -- (26.56, 17.46) -- (24.68, 18.08) -- cycle;
\fill[cadmit] (26.56, 16.96) -- (28.44, 16.96) -- (28.44, 15.84) -- (26.56, 16.46) -- cycle;
\draw[line width=1.0pt, draw=black] (24.68, 18.08) -- (26.56, 17.46) -- (26.56, 16.46) -- (28.44, 15.84);
\draw[line width=0.65pt, densely dashed, draw=black!75] (24.68, 18.16) -- (28.44, 18.16);
\draw[line width=0.65pt, densely dashed, draw=black!75] (24.68, 15.76) -- (28.44, 15.76);
\draw[arr, line width=0.8pt] (24.52, 16.96) -- (28.64, 16.96);
\draw[arr, line width=0.8pt] (26.56, 15.52) -- (26.56, 18.46);
\node[font=\fontsize{6.8}{7.6}\selectfont\bfseries] at (28.56, 17.18) {$I_k$};
\node[font=\fontsize{6.8}{7.6}\selectfont\bfseries] at (27.22, 18.50) {$\delta\hat{V}_{t,\mathrm{comp},k}$};
\node[font=\fontsize{6.6}{7.6}\selectfont\bfseries] at (25.62, 17.34) {Charge OK};
\node[font=\fontsize{6.6}{7.6}\selectfont\bfseries] at (27.52, 16.52) {Disch.\ OK};
\node[font=\fontsize{5.9}{6.8}\selectfont] at (27.50, 17.78) {Clamped ($I_k\delta\hat{V}\!>\!0$)};
\node[font=\fontsize{6.4}{7.4}\selectfont\bfseries] at (26.56, 14.94)
    {$|\delta\hat{V}|\le V_{\mathrm{pol}}^{\max} \le 20.0\;\mathrm{mV}$;\; $I_k\cdot\delta\hat{V}\le 0$};


% --- DEDICATED CONFLICT-FREE SIGNAL BUSES ---
% 1. Direct BMS Bus cleanly routed along y = 20.08 above ALL subcards -> V_meas down x=29.06 to Sigma_2!
\draw[arr] (6.06, 16.40) -- (6.54, 16.40) -- (6.54, 20.08) -- (29.06, 20.08) -- (29.06, 13.88) -- (27.74, 13.88);
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1.5pt, inner sep=1.6pt, font=\fontsize{6.8}{7.8}\selectfont\bfseries]
    at (16.50, 20.08) {Direct BMS Bus: $(I_k, T_k)\to$ MLP Inputs \& $V_{\mathrm{meas},k}\to \Sigma_2$};

\filldraw[fill=black] (17.54, 20.08) circle (1.4pt);
\draw[line width=0.85pt] (17.54, 20.08) -- (17.54, 18.02);
\draw[arr, line width=0.85pt] (17.54, 18.60) -- (18.12, 18.60);
\draw[arr, line width=0.85pt] (17.54, 18.02) -- (18.12, 18.02);
\filldraw[fill=black] (17.54, 18.60) circle (1.3pt);

% 2. Fast State Output \hat{x}_k^- straight horizontally into MLP pins 3-5
\draw[arr, line width=0.9pt] (16.96, 16.86) -- (18.12, 16.86);
\node[fill=white, draw=black, line width=0.4pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{6.6}{7.4}\selectfont\bfseries]
    at (17.54, 17.26) {$\hat{\mathbf{x}}_k^-$};

% 3. Dedicated V_{t,ECM,k}^- rail along y = 13.88 BELOW MLP into Sigma_1
\draw[arr, line width=0.9pt] (16.80, 15.00) -- (17.24, 15.00) -- (17.24, 13.88) -- (24.90, 13.88);
\node[fill=cmlp, draw=black, line width=0.45pt, rounded corners=1.2pt, inner sep=1.4pt, font=\fontsize{6.7}{7.7}\selectfont\bfseries]
    at (20.97, 13.88) {$\hat{V}_{t,\mathrm{ECM},k}^-$};

% 4. SOH_{ref,k} exits Slow Filter along x=17.54 with IEEE bridge hop over y=13.88 into MLP pin 6
\draw[line width=0.9pt] (16.96, 11.16) -- (17.54, 11.16) -- (17.54, 13.75);
\draw[line width=0.9pt] (17.54, 13.75) arc (-90:90:0.13cm);
\draw[arr, line width=0.9pt] (17.54, 14.01) -- (17.54, 15.70) -- (18.12, 15.70);
\node[fill=white, draw=black, line width=0.45pt, rounded corners=1pt, inner sep=1.2pt, font=\fontsize{6.3}{7.2}\selectfont\bfseries]
    at (17.54, 12.40) {$\mathrm{SOH}_{\mathrm{ref}}$};

% 5. Exact Cascaded Summing Junctions along y = 13.88
\filldraw[fill=white, draw=black, line width=0.85pt] (25.12, 13.88) circle (0.22cm);
\node[font=\fontsize{7.4}{8.4}\selectfont\bfseries] at (25.12, 13.88) {$\Sigma_1$};
\draw[arr, line width=0.9pt] (25.12, 14.66) -- (25.12, 14.10);
\node[anchor=west, font=\fontsize{6.6}{7.6}\selectfont\bfseries] at (25.24, 14.36) {$\delta\hat{V}_{t,\mathrm{comp},k}$};
\node[font=\fontsize{6.6}{7.4}\selectfont\bfseries] at (24.76, 14.08) {$+$};
\node[font=\fontsize{6.6}{7.4}\selectfont\bfseries] at (24.88, 14.24) {$+$};

\draw[arr, line width=0.9pt] (25.34, 13.88) -- (27.30, 13.88);
\node[fill=cmlp, inner sep=1.2pt, font=\fontsize{6.6}{7.6}\selectfont\bfseries] at (26.32, 13.88) {$\hat{V}_{t,\mathrm{PIDKF},k}^-$};

\filldraw[fill=white, draw=black, line width=0.85pt] (27.52, 13.88) circle (0.22cm);
\node[font=\fontsize{7.4}{8.4}\selectfont\bfseries] at (27.52, 13.88) {$\Sigma_2$};
\node[font=\fontsize{7.0}{7.8}\selectfont\bfseries] at (27.14, 14.08) {$-$};
\node[font=\fontsize{7.0}{7.8}\selectfont\bfseries] at (27.90, 14.08) {$+$};
\node[font=\fontsize{6.4}{7.4}\selectfont\bfseries] at (28.38, 14.12) {$V_{\mathrm{meas},k}$};

% Innovation down to Subcard 3C
\draw[arr, line width=0.9pt] (27.52, 13.66) -- (27.52, 13.10);
\node[anchor=east, font=\fontsize{6.6}{7.6}\selectfont\bfseries] at (27.40, 13.38) {$e_{k,\mathrm{PIDKF}}$};

% Subcard 3C: Innovation Update, Exogenous Decoupling & Mean-Square UUB (x = 18.12 to 28.96, 10.84 cm wide!)
\draw[subcard] (18.12, 9.48) rectangle (28.96, 13.10);
\filldraw[fill=hdrgreen, draw=black, line width=0.55pt, rounded corners=1.5pt] (18.24, 12.56) rectangle (28.84, 12.98);
\node[font=\fontsize{7.1}{8.2}\selectfont\bfseries] at (23.54, 12.77)
    {Compensated Innovation Update, Exogenous Decoupling \& Stability Bound};
\node[align=center, font=\fontsize{6.5}{7.8}\selectfont] at (23.54, 11.02) {
    $\hat{V}_{t,\mathrm{PIDKF},k}^- = \hat{V}_{t,\mathrm{ECM},k}^- + \delta\hat{V}_{t,\mathrm{comp},k}$,\quad $e_{k,\mathrm{PIDKF}} = V_{\mathrm{meas},k} - \hat{V}_{t,\mathrm{PIDKF},k}^-$\\[2.2pt]
    $R_{\mathrm{eff},k} = R_v + \sigma_{\mathrm{NN}}^2 + \sigma_{\mathrm{approx}}^2 = 1.04\times 10^{-3}\;\mathrm{V}^2$\\[1.4pt]
    {\fontsize{6.0}{7.0}\selectfont ($R_v = 10^{-3}\;\mathrm{V}^2$,\; MLP Error $\sigma_{\mathrm{NN}} = 5.8\;\mathrm{mV}$,\; Truncation $\sigma_{\mathrm{approx}} = 1.9\;\mathrm{mV}$)}\\[2.4pt]
    \textbf{Exogenous Decoupling:}\; $\mathbf{H}_{\mathrm{PIDKF},k}\approx\mathbf{H}_k,\;\mathbf{C}_{\theta,\mathrm{PIDKF},k}\approx\mathbf{C}_{\theta,k}$ \;\textbf{($-37.3\%$ MCU Cycles)}\\[2.4pt]
    \textbf{Mean-Square UUB (Eqs.\ 24--25):}\; $\limsup_{k\to\infty}\mathbb{E}[\|\mathbf{e}_k^-\|^2] \le n\bar{p} + \left(\dfrac{c\bar{A}\bar{K}\bar{\varepsilon}}{1-\rho}\right)^2 < \infty$
};

% Clean Unobstructed Bottom Feedback Return Rail (y = 8.90)
\draw[line width=0.9pt] (23.54, 9.48) -- (23.54, 8.90) -- (6.54, 8.90);
\draw[arr, line width=0.9pt] (12.00, 8.90) -- (12.00, 9.48);
\filldraw[fill=black] (12.00, 8.90) circle (1.4pt);
\draw[arr, line width=0.9pt] (6.54, 8.90) -- (6.54, 17.98) -- (7.20, 17.98);
\node[fill=cgate, draw=black, line width=0.5pt, rounded corners=1.5pt, inner sep=1.6pt, font=\fontsize{7.0}{8.0}\selectfont\bfseries]
    at (17.54, 8.90) {Closed-Loop Innovation Feedback: $e_{k,\mathrm{PIDKF}}$};


% =========================================================================
% PANEL (b): Protocol, Embedded Latency Bar Chart & 10-Phase Matrix (y = 0.0 to 8.25)
% =========================================================================
\draw[outerbox] (0.0, 0.0) rectangle (29.40, 8.25);
\node[anchor=west, font=\fontsize{10.8}{12.2}\selectfont\bfseries] at (0.24, 7.86)
    {(b) Dual-Tier Validation Protocol, 8-Estimator Benchmark Latency, STM32F407 Profiling \& 10-Phase Matrix};

% Pillar 1: Zero-Leakage Protocol (x = 0.24 to 8.10, y = 4.76 to 7.48)
\draw[subcard, fill=cplant] (0.24, 4.76) rectangle (8.10, 7.48);
\filldraw[fill=hdrslate, draw=black, line width=0.65pt] (0.24, 7.00) rectangle (8.10, 7.48);
\node[font=\fontsize{8.0}{9.2}\selectfont\bfseries] at (4.17, 7.24) {Zero-Leakage Datasets \& Experimental Split};
\node[align=center, font=\fontsize{7.2}{8.6}\selectfont] at (4.17, 5.88) {
    CALCE (\texttt{INR18650-20R}, \texttt{ANR26650M1-B})\\[1.5pt]
    Oxford EIS \& NASA PCoE Diagnostics\\[1.5pt]
    Cell Batch~A (Calibrate/Train) $\perp$ Batch~B (Test)\\[1.5pt]
    Arbin BT-2000 Uncertainty: $\sigma_{\mathrm{SOC,ref}} \le \pm 0.036\%$
};

% Pillar 2: Embedded PGFPlots Log-Scale Cortex-M4 Latency Chart (x = 8.46 to 21.10, 12.64 cm wide!)
% Explicit exact bar annotations (zero nodes-near-coords bar-shift mismatch, white backdrop over x=20 dashed line!)
\draw[subcard, fill=white] (8.46, 4.76) rectangle (21.10, 7.48);
\filldraw[fill=hdrblue, draw=black, line width=0.65pt] (8.46, 7.00) rectangle (21.10, 7.48);
\node[font=\fontsize{7.8}{9.0}\selectfont\bfseries] at (14.78, 7.24)
    {8 Benchmarked Estimators Across 3 Paradigms (Cortex-M4F $\mu\mathrm{s}$, Log Scale)};

\begin{axis}[
    at={(10.06cm, 5.04cm)},
    anchor=south west,
    xbar=0pt,
    bar shift=0pt,
    xmode=log,
    xmin=1, xmax=26000,
    ymin=0.35, ymax=8.65,
    width=10.60cm,
    height=1.84cm,
    scale only axis,
    bar width=4.6pt,
    axis lines*=left,
    xtick={1, 10, 100, 1000},
    xticklabels={$1$, $10$, $10^2$, $10^3$},
    ytick={1, 2, 3, 4, 5, 6, 7, 8},
    yticklabels={BiLSTM, PINN, \textbf{PIDKF}, UKF, DEKF, AEKF, EKF, SMO},
    yticklabel style={font=\fontsize{6.2}{7.0}\selectfont, text=black},
    xticklabel style={font=\fontsize{6.2}{7.0}\selectfont, text=black, yshift=2pt},
    axis line style={line width=0.65pt, draw=black},
    xmajorgrids=true,
    grid style={dotted, black!30},
    clip=false
]
    \draw[densely dashed, black!80, line width=0.7pt] (axis cs:20, 0.35) -- (axis cs:20, 8.65);
    \node[font=\fontsize{5.8}{6.6}\selectfont\bfseries, anchor=south west] at (axis cs:21.5, 7.75) {RT $<20\;\mu\mathrm{s}$};

    \addplot[fill=barslate, draw=black, line width=0.5pt] coordinates {
        (1.67, 8) (3.87, 7) (12.42, 4)
    };
    \addplot[fill=barblue, draw=black, line width=0.5pt] coordinates {
        (5.24, 6) (8.63, 5)
    };
    \addplot[fill=baramber, draw=black, line width=0.5pt] coordinates {
        (127.98, 2) (2440.48, 1)
    };
    \addplot[fill=bargreen, draw=black, line width=0.7pt] coordinates {
        (14.20, 3)
    };

    % Exact aligned numeric labels centered on each horizontal bar with white pills over dashed line!
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner xsep=1pt, inner ysep=0.2pt, fill=white, fill opacity=0.92, text opacity=1] at (axis cs:1.80, 8) {$1.67\;\mu\mathrm{s}$};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner xsep=1pt, inner ysep=0.2pt, fill=white, fill opacity=0.92, text opacity=1] at (axis cs:4.18, 7) {$3.87\;\mu\mathrm{s}$};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner xsep=1pt, inner ysep=0.2pt, fill=white, fill opacity=0.92, text opacity=1] at (axis cs:5.66, 6) {$5.24\;\mu\mathrm{s}$};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner xsep=1pt, inner ysep=0.2pt, fill=white, fill opacity=0.92, text opacity=1] at (axis cs:9.30, 5) {$8.63\;\mu\mathrm{s}$};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner xsep=1pt, inner ysep=0.2pt, fill=white, fill opacity=0.92, text opacity=1] at (axis cs:13.4, 4) {$12.42\;\mu\mathrm{s}$};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner xsep=1pt, inner ysep=0.2pt, fill=white, fill opacity=0.94, text opacity=1] at (axis cs:15.4, 3) {\textbf{$14.20\;\mu\mathrm{s}$ ($172\times$ vs.\ LSTM)}};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner sep=1pt] at (axis cs:138, 2) {$127.98\;\mu\mathrm{s}$};
    \node[anchor=west, font=\fontsize{5.9}{6.7}\selectfont\bfseries, inner sep=1pt] at (axis cs:2640, 1) {$2.44\;\mathrm{ms}$};
\end{axis}

% Pillar 3: Bare-Metal MCU Profiling + Crisp IC Package (x = 21.46 to 29.16, 7.70 cm wide!)
\draw[subcard, fill=cmlp] (21.46, 4.76) rectangle (29.16, 7.48);
\filldraw[fill=hdrgreen, draw=black, line width=0.65pt] (21.46, 7.00) rectangle (29.16, 7.48);
\node[font=\fontsize{8.0}{9.2}\selectfont\bfseries] at (25.31, 7.24) {Bare-Metal ARM Cortex-M4F Bench};

\filldraw[fill=hdrslate, draw=black, line width=0.7pt] (21.82, 5.32) rectangle (22.46, 6.44);
\foreach \p in {5.48, 5.74, 6.00, 6.26} {
    \draw[line width=0.6pt] (21.66, \p) -- (21.82, \p);
    \draw[line width=0.6pt] (22.46, \p) -- (22.62, \p);
}
\node[font=\fontsize{6.4}{7.2}\selectfont\bfseries] at (22.14, 5.88) {M4F};
\node[anchor=west, align=left, font=\fontsize{6.8}{8.2}\selectfont] at (22.74, 5.88) {
    STM32F407VG ($168\;\mathrm{MHz}$, IEEE-754 FPU)\\
    Zero Heap / MISRA-C:2012 / GCC \texttt{-O3}\\
    $1,540\;\mathrm{FLOPs}$, $2,386\;\text{cycles}$ (\texttt{DWT})\\
    $8.2\;\mathrm{KB}$ Flash, $0.85\;\mathrm{KB}$ RAM ($4.97\;\mu\mathrm{s}$ M7)
};

\draw[arr, line width=0.85pt] (8.10, 6.12) -- (8.46, 6.12);
\draw[arr, line width=0.85pt] (21.10, 6.12) -- (21.46, 6.12);
\draw[line width=0.7pt] (4.17, 4.76) -- (4.17, 4.50) -- (25.31, 4.50) -- (25.31, 4.76);
\draw[arr, line width=0.8pt] (14.78, 4.76) -- (14.78, 4.26);


% --- 10-Phase Experimental Matrix Grid (2 rows x 5 columns, y = 0.24 to 4.26, 5.54 cm wide each!) ---
% Row 1: y = 2.32 to 4.26
\draw[phasecard] (0.24, 2.32) rectangle (5.78, 4.26);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (0.24, 3.82) rectangle (5.78, 4.26);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (3.01, 4.04) {01 $\mid$ Dynamic Cycles};
\node[align=center, font=\fontsize{6.9}{8.3}\selectfont] at (3.01, 3.07) {
    UDDS, US06, FUDS, DST\\
    Fresh Cell ($100\%$ SOH, $25^\circ\mathrm{C}$)\\
    \textbf{EKF: $0.065\%$ $\mid$ DEKF: $0.191\%$}
};

\draw[phasecard] (6.08, 2.32) rectangle (11.62, 4.26);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (6.08, 3.82) rectangle (11.62, 4.26);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (8.85, 4.04) {02 $\mid$ Multi-Stage Aging};
\node[align=center, font=\fontsize{6.9}{8.3}\selectfont] at (8.85, 3.07) {
    $\mathrm{SOH}\;100\% \to 95\% \to 90\% \to 80\%$\\
    EKF Surge: $45.7\times$ ($2.973\%$)\\
    \textbf{DEKF: $0.585\%$ $\mid$ PIDKF: $0.482\%$}
};

\draw[phasecard] (11.92, 2.32) rectangle (17.46, 4.26);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (11.92, 3.82) rectangle (17.46, 4.26);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (14.69, 4.04) {03 $\mid$ Thermal Regimes};
\node[align=center, font=\fontsize{6.9}{8.3}\selectfont] at (14.69, 3.07) {
    Cold ($0^\circ\mathrm{C}$) \& Hot ($45^\circ\mathrm{C}$)\\
    DEKF $0^\circ\mathrm{C}$: $2.254\%$ $\mid$ SMO: $0.929\%$\\
    \textbf{Aged PIDKF ($0^\circ\mathrm{C}$): $0.842\%$ (NMC)}
};

\draw[phasecard] (17.76, 2.32) rectangle (23.30, 4.26);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (17.76, 3.82) rectangle (23.30, 4.26);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (20.53, 4.04) {04 $\mid$ Initial Convergence};
\node[align=center, font=\fontsize{6.9}{8.3}\selectfont] at (20.53, 3.07) {
    Initial Offset $\Delta\mathrm{SOC}_0 = -40\%$\\
    Closed-Loop Riccati Contraction\\
    \textbf{EKF/AEKF: $7/6\;\mathrm{s}$ $\mid$ PIDKF: $18.0\;\mathrm{s}$}
};

\draw[phasecard] (23.60, 2.32) rectangle (29.16, 4.26);
\filldraw[fill=hdrblue, draw=black, line width=0.6pt] (23.60, 3.82) rectangle (29.16, 4.26);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (26.38, 4.04) {05 $\mid$ Radar Trade-Off};
\node[align=center, font=\fontsize{6.9}{8.3}\selectfont] at (26.38, 3.07) {
    5-Axis Pareto Envelope:\\
    Accuracy, Aging, Thermal,\\
    Convergence \& MCU Efficiency
};

% Row 2: y = 0.24 to 2.18
\draw[phasecard] (0.24, 0.24) rectangle (5.78, 2.18);
\filldraw[fill=hdrteal, draw=black, line width=0.6pt] (0.24, 1.74) rectangle (5.78, 2.18);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (3.01, 1.96) {06 $\mid$ LFP Chemistry};
\node[align=center, font=\fontsize{6.8}{8.2}\selectfont] at (3.01, 0.99) {
    $20\%\text{--}80\%$ Plateau $+V_h$ ($\pm 30\;\mathrm{mV}$)\\
    Gramian $\lambda_{\min}(\mathcal{G}_o) \ge 1.21\!\times\!10^{-3}$\\
    \textbf{Aged PIDKF: $0.149\%$ ($0.158\%$ CALCE)}
};

\draw[phasecard] (6.08, 0.24) rectangle (11.62, 2.18);
\filldraw[fill=hdramber, draw=black, line width=0.6pt] (6.08, 1.74) rectangle (11.62, 2.18);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (8.85, 1.96) {07 $\mid$ Sensor Faults};
\node[align=center, font=\fontsize{6.8}{8.2}\selectfont] at (8.85, 0.99) {
    $+0.25\;\mathrm{A}$ Hall Bias \& $25\;\mathrm{mV}$ EMI\\
    SMO Drift Clamping: $1.024\%$\\
    \textbf{EKF: $1.817\%$ $\mid$ PIDKF: $1.428\%$ RMSE}
};

\draw[phasecard] (11.92, 0.24) rectangle (17.46, 2.18);
\filldraw[fill=hdrteal, draw=black, line width=0.6pt] (11.92, 1.74) rectangle (17.46, 2.18);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (14.69, 1.96) {08 $\mid$ EIS Diagnostics};
\node[align=center, font=\fontsize{6.8}{8.2}\selectfont] at (14.69, 0.99) {
    $100\;\mathrm{kHz}\text{--}1\;\mathrm{mHz}$ Randles--CPE\\
    $R_0$ ($+50\%$), $R_{\mathrm{ct}}$ ($+150\%$), $\alpha=0.88$\\
    \textbf{RMS Fit Error: $1.1\%\text{--}1.8\%$}
};

\draw[phasecard] (17.76, 0.24) rectangle (23.30, 2.18);
\filldraw[fill=hdramber, draw=black, line width=0.6pt] (17.76, 1.74) rectangle (23.30, 2.18);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (20.53, 1.96) {09 $\mid$ 8S Pack Scaling};
\node[align=center, font=\fontsize{6.8}{8.2}\selectfont] at (20.53, 0.99) {
    Dispersion ($\sigma_Q=3.0\%,\;\sigma_R=10.0\%$)\\
    Core Thermal $\Delta T\!=\!+3.2^\circ\mathrm{C}$ ($1.24\times$ Fade)\\
    \textbf{Weakest Cell 4: $0.337\%$ (Dist.\ DEKF)}
};

\draw[phasecard, fill=cmlp, line width=0.9pt] (23.60, 0.24) rectangle (29.16, 2.18);
\filldraw[fill=hdrgreen, draw=black, line width=0.9pt] (23.60, 1.74) rectangle (29.16, 2.18);
\node[font=\fontsize{7.5}{8.6}\selectfont\bfseries] at (26.38, 1.96) {10 $\mid$ Hybrid PIDKF};
\node[align=center, font=\fontsize{6.8}{8.2}\selectfont] at (26.38, 0.99) {
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
    data = build_pgfplots_data()
    target_folders = ["results", "manuscript", "submission_journal_of_energy_storage"]
    compile_vector_and_png(get_mechanism_tex(data), "fig0_electrochemical_mechanism", target_folders, dpi=600)
    compile_vector_and_png(get_framework_tex(), "fig0_pidkf_framework", target_folders, dpi=600)
