# -*- coding: utf-8 -*-
tex_path = 'manuscript/manuscript_jeecs.tex'
with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Smooth rephrasing of LFP state-space to eliminate underfull badness 1389
target_lfp_text = r"""For LFP, the augmented 4-state state-space model is formulated as $\mathbf{x}_k^{\text{LFP}} = [\text{SOC}_k, U_{1,k}, U_{2,k}, V_{h,k}]^T$, governed by $\mathbf{x}_{k+1}^{\text{LFP}} = \mathbf{A}_k^{\text{LFP}} \mathbf{x}_k^{\text{LFP}} + \mathbf{B}_k^{\text{LFP}} I_k + \mathbf{g}_k + \mathbf{w}_k$, with transition matrix $\mathbf{A}_k^{\text{LFP}} = \operatorname{diag}\left(1, e^{-\Delta t/\tau_1}, e^{-\Delta t/\tau_2}, e^{-\beta|\Delta z_k|}\right)$, deterministic input vector $\mathbf{B}_k^{\text{LFP}} = [B_{1,k}, B_{2,k}, B_{3,k}, 0]^T$, nonlinear hysteresis excitation vector $\mathbf{g}_k = [0, 0, 0, (1 - e^{-\beta |\Delta z_k|}) M(z_k) \operatorname{sgn}(I_k)]^T$, and terminal voltage observation:"""

replacement_lfp_text = r"""For LFP chemistry, the state vector is augmented with dynamic hysteresis: $\mathbf{x}_k^{\text{LFP}} = [\text{SOC}_k, U_{1,k}, U_{2,k}, V_{h,k}]^T$. The discrete evolution follows $\mathbf{x}_{k+1}^{\text{LFP}} = \mathbf{A}_k^{\text{LFP}} \mathbf{x}_k^{\text{LFP}} + \mathbf{B}_k^{\text{LFP}} I_k + \mathbf{g}_k + \mathbf{w}_k$ with $\mathbf{A}_k^{\text{LFP}} = \operatorname{diag}(1, e^{-\Delta t/\tau_1}, e^{-\Delta t/\tau_2}, e^{-\beta|\Delta z_k|})$, $\mathbf{B}_k^{\text{LFP}} = [B_{1,k}, B_{2,k}, B_{3,k}, 0]^T$, and nonlinear excitation $\mathbf{g}_k = [0, 0, 0, (1 - e^{-\beta |\Delta z_k|}) M(z_k) \operatorname{sgn}(I_k)]^T$. The terminal voltage satisfies:"""

if target_lfp_text in text:
    text = text.replace(target_lfp_text, replacement_lfp_text)
    print("LFP state text replaced!")
else:
    print("LFP state text NOT found!")

# 2. Relocate Table 7 before Table 6 so Table 7 lands on Page 9
tab7_block = r"""\begin{table*}[t]
\caption{Series 8S pack benchmark with cell-to-cell parameter inconsistency\label{tab:7_pack}}
\centering{%
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc}
\toprule
Estimation Architecture & Target State & RMSE (\%) & MaxE (\%) & Safety Margin Risk \\
\midrule
Lumped Pack Observer & Mean Pack SOC & \textbf{0.128} & 0.443 & Moderate \\
Distributed DEKF & Mean Pack SOC & 0.149 & \textbf{0.336} & None (Optimal) \\
Lumped Pack Observer & Weakest Cell SOC & 1.744 & 2.302 & High (Overdischarge / Plating) \\
Distributed DEKF & Weakest Cell SOC & \textbf{0.337} & \textbf{0.963} & Guaranteed Tracking \\
\bottomrule
\end{tabular*}
}%
\end{table*}"""

if tab7_block in text:
    text = text.replace(tab7_block, "")
    # Place right before Phase 7 subsection
    target_pos = r"""\subsection{Phase 7: Sensor Corruption and Environmental Stress Testing}"""
    text = text.replace(target_pos, tab7_block + "\n\n" + target_pos)
    print("Table 7 moved before Phase 7!")
else:
    print("Table 7 block NOT found!")

# 3. Format Appendix A and Appendix B with zero overfull and zero underfull
target_apps = r"""\appendix
\section{State-Space Matrices and Analytic Jacobians\label{app:jacobians}}
For the 3-state NMC ($\mathbf{x}_k \in \mathbb{R}^3$) and 4-state LFP ($\mathbf{x}_k^{\text{LFP}} \in \mathbb{R}^4$) models, the state matrices and observation Jacobians are:
\begin{align}
\mathbf{A}_k^{\text{NMC}} &= \operatorname{diag}(1, a_1, a_2), \quad \mathbf{H}_k^{\text{NMC}} = \begin{bmatrix} V_{\text{ocv}}'(\widehat{\text{SOC}}_k^-) & -1 & -1 \end{bmatrix}, \label{eqn:app_nmc}\\
\mathbf{A}_k^{\text{LFP}} &= \operatorname{diag}(1, a_1, a_2, a_h), \quad \mathbf{H}_k^{\text{LFP}} = \begin{bmatrix} V_{\text{ocv}}'(\widehat{\text{SOC}}_k^-) & -1 & -1 & -1 \end{bmatrix}, \label{eqn:app_lfp}
\end{align}
where $a_i = e^{-\Delta t/\tau_i}$, $a_h = e^{-\beta|\Delta z_k|}$, deterministic inputs $\mathbf{B}_k^{\text{NMC}} = [B_{1,k}, B_{2,k}, B_{3,k}]^T$ via Eq.~\eqref{eqn:Bk_mat}, $\mathbf{B}_k^{\text{LFP}} = [\mathbf{B}_k^{\text{NMC}}; 0]$, and $\mathbf{g}_k = [0, 0, 0, (1 - a_h) M(z_k) \operatorname{sgn}(I_k)]^T$. In DEKF, the Fréchet parameter sensitivity matrix $\mathbf{C}_{\theta,k} = \left. \frac{d V_k}{d \boldsymbol{\theta}} \right|_{\hat{\boldsymbol{\theta}}_k^-} \in \mathbb{R}^{1 \times 2}$ is:
\begin{equation}\label{eqn:app_C_theta}
\mathbf{C}_{\theta,k} = \begin{bmatrix} -I_k & -\frac{\eta \Delta t I_{k-1}}{3600} V_{\text{ocv}}'(\widehat{\text{SOC}}_k^+) \end{bmatrix}.
\end{equation}

\section{Lyapunov Stability, Finite-Time Reaching, and Error Boundedness of SMO\label{app:lyapunov}}
Define the terminal voltage estimation error as sliding manifold $S(t) = V_{\text{meas}}(t) - \hat{V}_t(t)$. Differentiating yields continuous error dynamics $\dot{S} = \Delta \dot{f}(t) - \gamma(\hat{z}) \operatorname{sat}(S/\mu)$, where $|\Delta \dot{f}(t)| \le \Phi < \infty$ bounds lumped modeling uncertainties, and effective injection gain is $\gamma(\hat{z}) \equiv \mathbf{H}\mathbf{L} = l_1 V_{\text{ocv}}'(\hat{z}) - l_2 - l_3$ for NMC ($\gamma = l_1 V_{\text{ocv}}' - l_2 - l_3 - l_h$ for LFP). Because $V_{\text{ocv}}' \ge \alpha > 0$ ($\alpha = 0.481\text{ V}$ for NMC, $\approx 0.08\text{ V}$ for LFP), choosing $l_1 > (l_2 + l_3 + \Phi + \eta_s)/\alpha$ ensures $\gamma(\hat{z}) \ge \gamma_{\min} \ge \Phi + \eta_s > 0$ with reachability margin $\eta_s > 0$.

Consider Lyapunov candidate $V(S) = \frac{1}{2} S^2$.
1.~\emph{Finite-Time Reaching ($|S| > \mu$)}: Since $\operatorname{sat}(S/\mu) = \operatorname{sgn}(S)$, the derivative satisfies $\dot{V} = S \Delta \dot{f} - \gamma |S| \le -(\gamma_{\min} - \Phi)|S| \le -\eta_s |S| = -\eta_s \sqrt{2V}$. Integrating $\frac{dV}{\sqrt{V}} \le -\eta_s \sqrt{2} dt$ proves that trajectories reach boundary layer $\mathcal{B}_\mu \equiv \{S \mid |S| \le \mu\}$ in finite time:
\begin{equation}\label{eqn:t_reach}
t_{\text{reach}} \le \frac{|S(0)| - \mu}{\eta_s}.
\end{equation}
2.~\emph{Boundary Layer Invariance ($|S| \le \mu$)}: Inside $\mathcal{B}_\mu$, $\dot{S} = -(\gamma/\mu)S + \Delta \dot{f}$, giving $\dot{V} \le -(\gamma_{\min}/\mu)S^2 + \Phi |S| = -|S|[(\gamma_{\min}/\mu)|S| - \Phi]$. Hence $\dot{V} < 0$ whenever $|S| > \mu\Phi/\gamma_{\min}$. Because $\gamma_{\min} > \Phi$, the compact set $\Omega_{\text{ult}} = \{S \mid |S| \le \mu\Phi/\gamma_{\min} < \mu\} \subset \mathcal{B}_\mu$ is positively invariant, eliminating chattering.
3.~\emph{Uniform Ultimate Boundedness (UUB) of Error Vector $\mathbf{e}_x$}: Within $\mathcal{B}_\mu$, equivalent control satisfies $|u_{\text{eq}}| \le \Phi/\gamma_{\min}$. Polarization errors $\dot{e}_{U_i} = -\frac{1}{\tau_i} e_{U_i} - l_{i+1} u_{\text{eq}} + d_{U_i}$ are exponentially Input-to-State Stable (ISS) with bound $|e_{U_i}| \le \tau_i(|l_{i+1}|\frac{\Phi}{\gamma_{\min}} + D_{U_i}) \equiv \bar{e}_{U_i}$. By the Mean Value Theorem, $V_{\text{ocv}}(z) - V_{\text{ocv}}(\hat{z}) = V_{\text{ocv}}'(\xi) e_z$ with $\xi \in (\hat{z}, z)$. Thus, the State-of-Charge error is strictly bounded:
\begin{align}\label{eqn:e_z_bound}
|e_z(t)| &\le \frac{|S(t)| + \sum_{i=1}^2 |e_{U_i}(t)| + |v(t)|}{V_{\text{ocv}}'(\xi)} \nonumber\\
&\le \frac{\mu + \bar{e}_{U_1} + \bar{e}_{U_2} + V_{\text{noise}}}{\alpha}.
\end{align}
Because $V_{\text{ocv}}'(\xi) \ge \alpha > 0$ uniformly on $z \in [0, 1]$, the denominator is strictly bounded away from zero. Consequently, full error vector $\|\mathbf{e}_x(t)\|$ is Uniformly Ultimately Bounded (UUB), completing the proof."""

replacement_apps = r"""\appendix
\section{State-Space Matrices and Analytic Jacobians\label{app:jacobians}}
For the 3-state NMC ($\mathbf{x}_k \in \mathbb{R}^3$) and 4-state LFP ($\mathbf{x}_k^{\text{LFP}} \in \mathbb{R}^4$) models, the state transition matrices and observation Jacobians are:
\begin{align}
\mathbf{A}_k^{\text{NMC}} &= \operatorname{diag}(1, a_1, a_2), \quad \mathbf{A}_k^{\text{LFP}} = \operatorname{diag}(1, a_1, a_2, a_h), \label{eqn:app_A}\\
\mathbf{H}_k^{\text{NMC}} &= \begin{bmatrix} V_{\text{ocv}}'(\widehat{\text{SOC}}_k^-) & -1 & -1 \end{bmatrix}, \label{eqn:app_H_nmc}\\
\mathbf{H}_k^{\text{LFP}} &= \begin{bmatrix} V_{\text{ocv}}'(\widehat{\text{SOC}}_k^-) & -1 & -1 & -1 \end{bmatrix}, \label{eqn:app_H_lfp}
\end{align}
where $a_i = e^{-\Delta t/\tau_i}$, $a_h = e^{-\beta|\Delta z_k|}$, deterministic inputs are $\mathbf{B}_k^{\text{NMC}} = [B_{1,k}, B_{2,k}, B_{3,k}]^T$ via Eq.~\eqref{eqn:Bk_mat}, $\mathbf{B}_k^{\text{LFP}} = [\mathbf{B}_k^{\text{NMC}}; 0]$, and $\mathbf{g}_k = [0, 0, 0, (1 - a_h) M(z_k) \operatorname{sgn}(I_k)]^T$. In DEKF, the Fréchet parameter sensitivity matrix $\mathbf{C}_{\theta,k} = \left. \frac{d V_k}{d \boldsymbol{\theta}} \right|_{\hat{\boldsymbol{\theta}}_k^-} \in \mathbb{R}^{1 \times 2}$ is:
\begin{equation}\label{eqn:app_C_theta}
\mathbf{C}_{\theta,k} = \begin{bmatrix} -I_k & -\frac{\eta \Delta t I_{k-1}}{3600} V_{\text{ocv}}'(\widehat{\text{SOC}}_k^+) \end{bmatrix}.
\end{equation}

\section{Lyapunov Stability, Finite-Time Reaching, and Error Boundedness of SMO\label{app:lyapunov}}
Define the terminal voltage estimation error as sliding manifold $S(t) = V_{\text{meas}}(t) - \hat{V}_t(t)$. Differentiating yields continuous error dynamics $\dot{S} = \Delta \dot{f}(t) - \gamma(\hat{z}) \operatorname{sat}(S/\mu)$, where $|\Delta \dot{f}(t)| \le \Phi < \infty$ bounds lumped modeling uncertainties, and effective injection gain is $\gamma(\hat{z}) \equiv \mathbf{H}\mathbf{L} = l_1 V_{\text{ocv}}'(\hat{z}) - l_2 - l_3$ for NMC ($\gamma = l_1 V_{\text{ocv}}' - l_2 - l_3 - l_h$ for LFP). Because $V_{\text{ocv}}' \ge \alpha > 0$ ($\alpha = 0.481\text{ V}$ for NMC, $\approx 0.08\text{ V}$ for LFP), choosing feedback gains such that $l_1 > (l_2 + l_3 + \Phi + \eta_s)/\alpha$ ensures $\gamma(\hat{z}) \ge \gamma_{\min} \ge \Phi + \eta_s > 0$ with reachability margin $\eta_s > 0$.

Consider Lyapunov candidate $V(S) = \frac{1}{2} S^2$.

\textbf{1. Finite-Time Reaching ($|S| > \mu$)}:
Since $\operatorname{sat}(S/\mu) = \operatorname{sgn}(S)$, the derivative satisfies $\dot{V} = S \Delta \dot{f} - \gamma |S| \le -(\gamma_{\min} - \Phi)|S| \le -\eta_s |S| = -\eta_s \sqrt{2V}$. Integrating $\frac{dV}{\sqrt{V}} \le -\eta_s \sqrt{2} dt$ demonstrates that trajectories reach boundary layer $\mathcal{B}_\mu \equiv \{S \mid |S| \le \mu\}$ in finite time:
\begin{equation}\label{eqn:t_reach}
t_{\text{reach}} \le \frac{|S(0)| - \mu}{\eta_s}.
\end{equation}

\textbf{2. Boundary Layer Invariance ($|S| \le \mu$)}:
Inside $\mathcal{B}_\mu$, error dynamics become $\dot{S} = -(\gamma/\mu)S + \Delta \dot{f}$. The Lyapunov derivative satisfies $\dot{V} \le -(\gamma_{\min}/\mu)S^2 + \Phi |S| < 0$ whenever $|S| > \mu\Phi/\gamma_{\min}$. Since $\gamma_{\min} > \Phi$, the compact set $\Omega_{\text{ult}} \equiv \{S \mid |S| \le \mu\Phi/\gamma_{\min} < \mu\} \subset \mathcal{B}_\mu$ is positively invariant, eliminating chattering while guaranteeing high tracking precision.

\textbf{3. Uniform Ultimate Boundedness (UUB) of Error Vector $\mathbf{e}_x$}:
Within $\mathcal{B}_\mu$, equivalent control satisfies $|u_{\text{eq}}| \le \Phi/\gamma_{\min}$. Polarization errors $\dot{e}_{U_i} = -\frac{1}{\tau_i} e_{U_i} - l_{i+1} u_{\text{eq}} + d_{U_i}$ are exponentially Input-to-State Stable (ISS) with bound $|e_{U_i}| \le \tau_i(|l_{i+1}|\frac{\Phi}{\gamma_{\min}} + D_{U_i}) \equiv \bar{e}_{U_i}$. By the Mean Value Theorem, $V_{\text{ocv}}(z) - V_{\text{ocv}}(\hat{z}) = V_{\text{ocv}}'(\xi) e_z$ with $\xi \in (\hat{z}, z)$. Thus, the State-of-Charge error is strictly bounded:
\begin{align}\label{eqn:e_z_bound}
|e_z(t)| &\le \frac{|S(t)| + \sum_{i=1}^2 |e_{U_i}(t)| + |v(t)|}{V_{\text{ocv}}'(\xi)} \nonumber\\
&\le \frac{\mu + \bar{e}_{U_1} + \bar{e}_{U_2} + V_{\text{noise}}}{\alpha}.
\end{align}
Because $V_{\text{ocv}}'(\xi) \ge \alpha > 0$ uniformly on $z \in [0, 1]$, the denominator is strictly positive and bounded away from zero. Consequently, the full error vector $\|\mathbf{e}_x(t)\|$ is Uniformly Ultimately Bounded (UUB), completing the proof."""

if target_apps in text:
    text = text.replace(target_apps, replacement_apps)
    print("Appendices updated cleanly!")
else:
    print("Appendices block NOT found!")

with open(tex_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(text)
print("Updated successfully!")
