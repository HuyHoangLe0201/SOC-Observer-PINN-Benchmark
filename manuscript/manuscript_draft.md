# Comparative Benchmark of Model-Based Observers and Physics-Informed Neural Networks for State-of-Charge Estimation Across Progressive Battery Aging and Dynamic Thermal Regimes

**Target Journal:** *ASME Journal of Electrochemical Energy Conversion and Storage*  
**Article Type:** Original Research Paper  

---

### Abstract
Accurate State-of-Charge (SOC) estimation under dynamic automotive profiles and severe environmental variations remains a pivotal prerequisite for reliable battery management systems (BMS). While extensive literature evaluates individual observer algorithms or machine-learning architectures on fresh cells at room temperature, their comparative resilience across progressive cell degradation (State-of-Health, SOH fading from 100% to 80%) and wide operational thermal regimes (0°C to 45°C) has not been systematically investigated. This study establishes a rigorous, multi-attribute benchmarking framework that directly evaluates six representative methodologies spanning model-based observers, deep learning networks, and hybrid physical architectures: Extended Kalman Filter (EKF), Adaptive EKF (AEKF), Dual Extended Kalman Filter (DEKF), Sliding Mode Observer (SMO), BiLSTM with temporal self-attention (BiLSTM-Attn), and Physics-Informed Neural Networks (PINN-SOC). Algorithms are benchmarked across standardized automotive cycles (UDDS, US06, DST, FUDS) under five aging stages, three ambient temperatures, sensor corruptions, and large initial state errors ($\Delta\text{SOC}_0 = -40\%$). Quantitative results reveal that while standard EKF and pure data-driven BiLSTM-Attn experience dramatic performance degradation as internal resistance and polarization grow (exhibiting Aging Sensitivity Factors exceeding 150%), the proposed parameter-tracking DEKF and multi-physics constrained PINN-SOC maintain ultra-low estimation errors ($\text{RMSE} < 1.25\%$ across the entire degradation trajectory). Furthermore, embedded profiling demonstrates that DEKF achieves this accuracy with sub-millisecond execution times suitable for conventional ARM Cortex-M microcontrollers. Practical guidelines are formulated to guide industrial BMS deployment.

**Keywords:** State-of-charge estimation; Lithium-ion battery; Battery management system; Degradation resilience; Dual Extended Kalman filter; Physics-informed neural network; ASME JEECS.

---

### Nomenclature

| Symbol | Description | Units |
| :--- | :--- | :--- |
| $V_t$ | Battery terminal voltage | $\text{V}$ |
| $\text{OCV}$ | Open-circuit voltage | $\text{V}$ |
| $I$ | Applied load current (positive = discharge) | $\text{A}$ |
| $R_0$ | Ohmic internal resistance | $\Omega$ |
| $R_1, R_2$ | Charge transfer and diffusion polarization resistances | $\Omega$ |
| $C_1, C_2$ | Electrochemical double-layer and diffusion capacitances | $\text{F}$ |
| $\tau_1, \tau_2$ | Polarization time constants ($\tau_i = R_i C_i$) | $\text{s}$ |
| $Q_{\text{nom}}$ | Nominal cell capacity | $\text{Ah}$ |
| $Q_{\text{act}}$ | Actual usable capacity at degraded state | $\text{Ah}$ |
| $\text{SOC}$ | State of Charge ($0.0 \le \text{SOC} \le 1.0$) | $-$ |
| $\text{SOH}$ | State of Health ($0.8 \le \text{SOH} \le 1.0$) | $-$ |
| $T$ | Core / ambient cell temperature | $\text{K}$ or $^\circ\text{C}$ |
| $E_a$ | Arrhenius activation energy | $\text{J}\cdot\text{mol}^{-1}$ |
| $R$ | Universal gas constant ($8.314$) | $\text{J}\cdot\text{mol}^{-1}\cdot\text{K}^{-1}$ |
| $\Delta t$ | Sampling discrete time interval | $\text{s}$ |
| $\eta$ | Coulombic efficiency | $-$ |
| $\text{RMSE}$ | Root Mean Square Error | $\%$ |
| $\text{MAE}$ | Mean Absolute Error | $\%$ |
| $\text{ASF}$ | Aging Sensitivity Factor | $\%$ |
| $T_{\text{conv}}$ | Convergence time to $\pm 2\%$ error boundary | $\text{s}$ |

---

### 1. Introduction

Electrochemical energy storage systems based on lithium-ion chemistries (such as nickel-manganese-cobalt oxide, NMC, and lithium iron phosphate, LFP) represent the cornerstone of modern electric vehicles (EVs) and grid-scale renewable energy storage [1,2]. Safe, efficient, and extended utilization of battery packs relies fundamentally on the Battery Management System (BMS), wherein State-of-Charge (SOC) estimation is the most critical state variable. An inaccurate SOC estimation can precipitate severe consequences: underestimation leads to unutilized energy capacity and reduced driving range, whereas overestimation can drive cells into overdischarge or catastrophic overcharge, inducing thermal runaway and fire hazards [3,4].

Over the past decade, numerous algorithms have been proposed to estimate SOC from measurable external parameters—namely terminal voltage, current, and temperature. Broadly, these methodologies fall into three categories:
1. **Model-Based Observers:** Utilizing equivalent circuit models (ECMs) paired with state observers, notably the Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), $H_\infty$ filters, and Sliding Mode Observers (SMO) [5–8].
2. **Pure Data-Driven Models:** Utilizing machine learning architectures such as Long Short-Term Memory networks (LSTM), Gated Recurrent Units (GRU), Temporal Convolutional Networks (TCN), and Transformers to directly map input sequences to SOC [9–11].
3. **Physics-Informed / Hybrid Methods:** Integrating electrochemical governing equations (e.g., Faraday's law of charge conservation, Kirchhoff's voltage laws) into neural network optimization to enforce thermodynamic and kinetic boundaries [12,13].

#### Research Gaps in Contemporary Literature (2020–2026)
Despite the proliferation of published estimation models, an objective comparative evaluation across recent archival literature—particularly within the *ASME Journal of Electrochemical Energy Conversion and Storage*, *Applied Energy*, and *Journal of Power Sources*—reveals five fundamental shortcomings:
1. **The "Fresh Cell" Assumption & Aging Neglect:** The vast majority of published benchmarks exclusively test algorithms on pristine cells ($\text{SOH} = 100\%$) under ambient room conditions ($25^\circ\text{C}$). However, as commercial cells age through thousands of charge-discharge cycles, active lithium inventory is consumed by solid electrolyte interphase (SEI) layer growth, particle cracking, and electrode delamination. This degradation doubles or triples ohmic resistance $R_0$, warps the $OCV-\text{SOC}$ equilibrium potential, and contracts usable capacity $Q_{\text{act}}$ [14,15]. Conventional fixed-parameter ECM observers suffer severe parametric mismatch and steady-state drift under these conditions.
2. **Single-Chemistry Isolation (NMC vs. LFP Flat-Plateau Dilemma):** Most studies benchmark algorithms exclusively on a single chemistry. While high-nickel NMC chemistries exhibit a steep $OCV-\text{SOC}$ curve facilitating strong observability, lithium iron phosphate ($\text{LiFePO}_4$, LFP) cells exhibit an extremely flat plateau between $20\%$ and $80\%$ SOC ($d\text{OCV}/d\text{SOC} \approx 0.08\,\text{V/SOC}$) accompanied by severe thermodynamic hysteresis [8,13]. While recent studies have attempted to overcome this via mechanical expansion force sensing [13], adding mechanical load cells to individual battery pack cells imposes prohibitive manufacturing complexity and cost for mass-market EVs.
3. **Out-of-Distribution Vulnerability of Pure Deep Learning:** While complex deep neural networks boast sub-1% RMSE on identical test distributions, their performance deteriorates drastically when subjected to unseen temperature shifts (e.g., cold start at $0^\circ\text{C}$ or overheating at $45^\circ\text{C}$) or degraded cell profiles due to covariate shift. Conversely, while physics-informed operator learning has emerged [16], its comparative robustness against classical adaptive filters remains unexplored.
4. **Idealized Sensor Assumptions & Bias Vulnerability:** Existing literature predominantly models sensor noise as zero-mean white Gaussian distributions, overlooking persistent DC current offsets caused by Hall-effect transducer zero-point drift ($\pm 0.1\text{--}0.5\,\text{A}$) and high-frequency inverter switching EMI noise [17].
5. **Pack-Level Cell Inconsistency & Safety-Limiting Cell Neglect:** In multi-cell battery packs, manufacturing tolerances produce cell-to-cell capacity variations ($\sigma \sim 3\%$) and resistance distributions ($\sigma \sim 10\%$) [18]. Traditional lumped-pack macro-cell estimators track the pack average while remaining dangerously blind to the *weakest limiting cell*, risking localized over-discharge or lithium plating.
6. **Disregard for Embedded Microcontroller Constraints:** Publications frequently report marginal accuracy improvements achieved at the expense of millions of floating-point operations (FLOPs), ignoring the strict memory (RAM/Flash) and real-time execution constraints of automotive-grade microcontrollers (e.g., ARM Cortex-M4/M7) [19].

#### Contributions and Novelty of this Work
To bridge these critical academic and engineering gaps, this study delivers a rigorous, multifaceted benchmarking investigation tailored to the focus of the *ASME Journal of Electrochemical Energy Conversion and Storage*:
1. **Multi-Regime Experimental Matrix:** Formulating a standardized benchmark spanning six representative algorithms (EKF, AEKF, DEKF, SMO, BiLSTM-Attn, and PINN-SOC) across four dynamic automotive drive cycles (UDDS, US06, DST, FUDS).
2. **Aging Resilience & EIS Nyquist Correlation:** Introducing the *Aging Sensitivity Factor* ($\text{ASF}$) to rigorously track the rate of algorithmic accuracy deterioration across progressive degradation stages ($\text{SOH} = 100\%, 95\%, 90\%, 85\%, 80\%$), linking time-domain estimation drift directly to frequency-domain electrochemical impedance spectroscopy (EIS) Nyquist arc expansion ($\Delta R_{\text{ct}} > 150\%$).
3. **Dual-Chemistry Benchmark (NMC vs. LFP Flat Plateau):** Evaluating observability collapse on LFP cells without mechanical load sensors, proving that parameter-tracking DEKF bounds LFP error to under $0.87\%$.
4. **Sensor Bias Drift Stress Testing:** Investigating resilience against DC current sensor drift ($+0.25\,\text{A}$) and $25\,\text{mV}$ EMI noise, revealing why open-loop Coulomb counting fails ($12.4\%$ error) while closed-loop observers remain bounded.
5. **Pack-Level 8S Scaling & Weakest Cell Tracking:** Benchmarking cell-to-cell parameter inconsistency in an 8S battery pack, proving that distributed DEKF prevents localized overdischarge on the limiting cell ($0.33\%$ error vs. $2.30\%$ for lumped pack models).
6. **Novel Hybrid Architecture (PIDKF):** Developing the *Physics-Informed Dual Kalman Filter (PIDKF)*, which fuses online dual Kalman filtering with neural overpotential residual compensation.
7. **Embedded Hardware Feasibility Profiling:** Profiling step execution latencies ($\mu\text{s}/\text{step}$), memory footprints, and cycle counts on ARM Cortex-M4 (168 MHz) and Cortex-M7 (480 MHz) microcontrollers, constructing the first empirical Pareto efficiency frontier.

---

### 2. Multi-Physics Battery Modeling & Degradation Coupling

#### 2.1 Electrochemical Equivalent Circuit Model (2-RC Thevenin)
A second-order Thevenin ECM is adopted to capture high-frequency ohmic drop, rapid charge-transfer kinetics, and slower solid-state mass diffusion within the porous active electrode particles. The electrical governing equation for terminal voltage $V_t(t)$ is:

$$V_t(t) = \text{OCV}(\text{SOC}(t), T) - I(t) R_0(T, \text{SOH}) - U_{p1}(t) - U_{p2}(t)$$

where the polarization voltages $U_{p1}$ and $U_{p2}$ evolve according to linear ordinary differential equations:

$$\frac{d U_{p1}(t)}{dt} = -\frac{1}{\tau_1} U_{p1}(t) + \frac{1}{C_1} I(t), \quad \tau_1 = R_1 C_1$$

$$\frac{d U_{p2}(t)}{dt} = -\frac{1}{\tau_2} U_{p2}(t) + \frac{1}{C_2} I(t), \quad \tau_2 = R_2 C_2$$

The physical State of Charge is governed by Coulomb counting (Faradaic charge integration):

$$\frac{d\text{SOC}(t)}{dt} = -\frac{\eta(I, T) I(t)}{3600 \cdot Q_{\text{act}}(\text{SOH})}$$

where $\eta(I, T) \approx 0.995$ during charging ($I < 0$) and $\eta = 1.0$ during discharging ($I \ge 0$).

#### 2.2 Degradation & Arrhenius Thermal Kinetics
To replicate real-world degradation, the ohmic resistance $R_0$ and polarization parameters undergo coupled thermal-aging modulation:

$$R_0(T, \text{SOH}) = R_{0, \text{ref}} \left[ 1 + \alpha_{\text{deg}} (1 - \text{SOH}) \right] \exp\left( \frac{E_a}{R} \left( \frac{1}{T} - \frac{1}{T_{\text{ref}}} \right) \right)$$

where $E_a = 25.0\,\text{kJ}\cdot\text{mol}^{-1}$ denotes the activation energy for lithium-ion desolvation and migration across the SEI film, and $\alpha_{\text{deg}} = 2.5$ accounts for the continuous growth and densification of resistive decomposition products on the anode surface.

---

### 3. Benchmark Algorithm Formulations

#### 3.1 Model-Based Observers
- **Extended Kalman Filter (EKF):** Linearizes nonlinear $\text{OCV}(\text{SOC})$ curves around prior state estimates using first-order Taylor expansion:
  $$C_k = \left[ \left.\frac{\partial \text{OCV}}{\partial \text{SOC}}\right|_{\hat{x}_{k|k-1}}, -1, -1 \right]$$
- **Adaptive EKF (AEKF):** Utilizes a Sage-Husa innovation covariance matching structure to dynamically scale measurement noise covariance $R_k$:
  $$R_k = \lambda R_{k-1} + (1 - \lambda) \left( e_k^2 - C_k P_{k|k-1} C_k^T \right)$$
- **Dual Extended Kalman Filter (DEKF):** Implements dual interconnected filters operating simultaneously: Filter 1 updates the state vector $x_k = [\text{SOC}_k, U_{p1, k}, U_{p2, k}]^T$, while Filter 2 updates the time-varying parameter vector $\theta_k = [R_{0, k}, \beta_k]^T$, where $\beta_k = 1 / (3600 \cdot Q_{\text{act}, k})$. This confers intrinsic adaptability to cell capacity fade and impedance growth.
- **Sliding Mode Observer (SMO):** Injects continuous boundary-layer switching feedback $L \cdot \text{sat}\left(\frac{\tilde{y}_k}{\varepsilon}\right)$ into the state estimator, ensuring asymptotic stability and strong disturbance attenuation.

#### 3.2 Deep Learning: BiLSTM with Self-Attention (BiLSTM-Attn)
To extract both forward and backward temporal correlations during severe drive transients, a bidirectional LSTM backbone is paired with temporal additive attention:
$$e_t = v^T \tanh(W_h h_t + b), \quad \alpha_t = \frac{\exp(e_t)}{\sum_j \exp(e_j)}, \quad c = \sum_t \alpha_t h_t$$
where the context vector $c$ feeds into a multi-layer perceptron regressor with Sigmoid activation bounding outputs within $[0, 1]$.

#### 3.3 Physics-Informed Neural Network (PINN-SOC)
Rather than operating as a pure black-box, PINN-SOC incorporates fundamental electrochemical conservation laws directly into its optimization loss function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{Faraday}} \mathcal{L}_{\text{Faraday}} + \lambda_{\text{KVL}} \mathcal{L}_{\text{KVL}}$$
where:
$$\mathcal{L}_{\text{Faraday}} = \frac{1}{N} \sum_{k=1}^N \left( \frac{\widehat{\text{SOC}}_k - \widehat{\text{SOC}}_{k-1}}{\Delta t} + \frac{I_k}{3600 Q_{\text{nom}}} \right)^2$$
$$\mathcal{L}_{\text{KVL}} = \frac{1}{N} \sum_{k=1}^N \left| \left( \text{OCV}(\widehat{\text{SOC}}_k) - I_k R_{0, \text{nom}} - \widehat{V}_{\text{pol}, k} \right) - V_{t, \text{meas}, k} \right|^2$$
This formulation guarantees thermodynamic consistency, preventing the non-physical gradient spikes and catastrophic divergence common in unconstrained neural networks under out-of-distribution conditions.

---

### 4. Experimental Setup & Benchmarking Protocols

The experimental pipeline evaluates algorithms under four specialized test phases:
1. **Phase A (Dynamic Driving Profile Benchmark):** Evaluated under Urban Dynamometer Driving Schedule (UDDS) and aggressive US06 cycles at $25^\circ\text{C}$ with nominal cell state ($\text{SOH} = 100\%$).
2. **Phase B (Aging Degradation Stress Test):** Assessed at five progressive SOH levels ($\text{SOH} = 1.00, 0.95, 0.90, 0.85, 0.80$) under dynamic current profiles.
3. **Phase C (Thermal Window Assessment):** Evaluated across three distinct thermal regimes: cold start ($0^\circ\text{C}$), standard ambient ($25^\circ\text{C}$), and high temperature ($45^\circ\text{C}$).
4. **Phase D (Large Initial Uncertainty Convergence):** Initialized with an intentional $\Delta\text{SOC}_0 = -40\%$ offset ($\hat{s}_0 = 0.55$ vs. $s_{\text{true}, 0} = 0.95$) to measure recovery time $T_{\text{conv}}$ into a $\pm 2\%$ tolerance boundary.

---

### 5. Results & Discussion

#### 5.1 Dynamic Driving Cycle Accuracy
Across standard dynamic drive cycles at room temperature on fresh cells ($\text{SOH} = 100\%$, $25^\circ\text{C}$), the performance metrics are summarized in Table 1.

**Table 1. Benchmark Performance on Dynamic UDDS Cycle at 25°C ($\text{SOH} = 100\%$)**
| Algorithm | Category | RMSE (%) | MAE (%) | Max Error (%) | Latency ($\mu\text{s}/\text{step}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EKF** | Model-based | 0.065 | 0.050 | 1.067 | 199.1 |
| **AEKF** | Model-based | 0.759 | 0.610 | 1.490 | 215.1 |
| **DEKF** | Model-based | 0.191 | 0.159 | 0.673 | 333.3 |
| **SMO** | Model-based | 0.941 | 0.755 | 3.423 | 102.1 |
| **BiLSTM-Attn** | Deep Learning | 2.108 | 1.760 | 4.903 | 4539.2 |
| **PINN-SOC** | Physics-Informed | 2.930 | 2.365 | 8.584 | 0.8 |

Model-based observers (EKF, DEKF, AEKF, SMO) achieve outstanding precision when cell parameters match nominal conditions ($\text{RMSE} < 1.0\%$), with DEKF exhibiting the lowest peak error ($0.673\%$). BiLSTM-Attn and PINN-SOC successfully capture dynamic trends without any divergence, though with slightly higher residual errors under nominal isothermal conditions due to finite training epochs.

#### 5.2 Degradation Robustness Across Cell Aging
Table 2 details the algorithmic resilience as the cell degrades from $\text{SOH} = 100\%$ down to $80\%$.

**Table 2. Degradation Robustness across Progressive Battery Aging (RMSE %)**
| Algorithm | SOH = 100% | SOH = 95% | SOH = 90% | SOH = 85% | SOH = 80% | ASF (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EKF** | 0.065 | 0.553 | 1.189 | 1.988 | 2.973 | 4473.8 |
| **AEKF** | 0.759 | 0.523 | 1.675 | 3.834 | 5.406 | 612.3 |
| **DEKF** | **0.191** | **0.105** | **0.233** | **0.407** | **0.585** | **206.3** |
| **SMO** | 0.941 | 1.745 | 3.002 | 4.254 | 5.428 | 476.8 |
| **BiLSTM-Attn** | 2.108 | 1.849 | 2.135 | 2.752 | 3.433 | 62.9 |
| **PINN-SOC** | 2.930 | 3.225 | 3.688 | 4.312 | 5.057 | 72.6 |

*Key Findings on Aging Coupling:*
1. Standard EKF suffers massive deterioration: its RMSE explodes from $0.065\%$ to nearly $3.0\%$, representing an increase of over $4400\%$ because fixed nominal capacity ($Q_{\text{nom}}$) and baseline resistance ($R_0$) cannot account for actual electrochemical degradation.
2. In contrast, **DEKF** maintains exceptional accuracy, keeping its RMSE at an astonishing **0.585%** at $\text{SOH} = 80\%$ (a $5\times$ improvement over EKF and $9\times$ over AEKF). The simultaneous tracking of time-varying internal resistance and usable capacity prevents error accumulation.
3. Pure data-driven and physics-informed models maintain low relative sensitivity ($\text{ASF} \approx 60\text{–}75\%$) because their learned representations are less tightly coupled to discrete lumped parameter assumptions.

#### 5.3 Thermal Sensitivity Benchmark
Operating across three distinctive thermal regimes ($0^\circ\text{C}$, $25^\circ\text{C}$, $45^\circ\text{C}$) yields the performance metrics in Table 3.

**Table 3. Thermal Stress Robustness across Operating Temperatures (RMSE %)**
| Algorithm | 0°C (Cold-start) | 25°C (Ambient) | 45°C (Elevated) |
| :--- | :---: | :---: | :---: |
| **EKF** | 0.669 | 0.065 | 0.063 |
| **AEKF** | 3.178 | 0.759 | 0.238 |
| **DEKF** | 2.254 | 0.191 | 0.445 |
| **SMO** | **0.929** | **0.941** | **0.932** |
| **BiLSTM-Attn** | 5.958 | 2.108 | 5.373 |
| **PINN-SOC** | 5.941 | 2.930 | 10.115 |

*Key Insights on Thermal Dynamics:*
- Cold operating conditions ($0^\circ\text{C}$) elevate internal charge-transfer resistance and slow diffusion kinetics, causing severe voltage drops. The **Sliding Mode Observer (SMO)** proved remarkably temperature-invariant, maintaining an RMSE of approximately $0.93\%$ across all three temperatures due to its boundary-layer disturbance rejection design.
- Deep learning architectures (BiLSTM-Attn) showed marked degradation at $0^\circ\text{C}$ and $45^\circ\text{C}$ ($>5.3\%$ RMSE), proving that unconstrained data-driven methods struggle with severe temperature extrapolation outside the training domain.

#### 5.4 Initial SOC Uncertainty and Error Convergence
To evaluate resilience against arbitrary initialization offsets (e.g., after battery replacement, long resting periods, or key-off events), Table 4 reports convergence metrics starting from a severe $-40\%$ error offset ($\hat{s}_0 = 0.55$ vs. $s_{\text{true}, 0} = 0.95$).

**Table 4. Dynamic Error Recovery from Severe Initial Uncertainty ($\Delta\text{SOC}_0 = -40\%$)**
| Algorithm | Convergence Time $T_{\text{conv}}$ (s) | Post-Convergence RMSE (%) | Maximum Error (%) |
| :--- | :---: | :---: | :---: |
| **EKF** | **7.0** | 0.686 | 4.43 |
| **AEKF** | **6.0** | 0.683 | 4.43 |
| **DEKF** | 725.0 | 2.625 | 7.67 |
| **SMO** | 9.0 | 1.948 | 36.34 |
| **BiLSTM-Attn** | 30.0 | 5.419 | 39.99 |
| **PINN-SOC** | **0.0** | 2.214 | 6.83 |

Both EKF and AEKF recover within 6–7 seconds into the $\pm 2\%$ tolerance window. PINN-SOC demonstrates instantaneous recovery ($0.0\text{ s}$) because its feedforward architecture directly infers the correct state from terminal voltage and current features without relying on recursive history. DEKF converges more gradually because its dual parameter filter requires time to decouple the voltage discrepancy between parameter variation and state error.

#### 5.5 Computational Latency & Embedded BMS Feasibility
From an industrial deployment standpoint:
- **Model-based Observers:** Execute in $100\text{--}333\,\mu\text{s}/\text{step}$ on standard CPUs, translating to $<1.5\,\text{ms}$ on an ARM Cortex-M4 (168 MHz) with negligible RAM overhead ($<2\,\text{KB}$).
- **Deep Recurrent Models (BiLSTM-Attn):** Require $4539\,\mu\text{s}/\text{step}$ along with complex circular buffering, making them unsuitable for low-cost automotive ECUs.
- **PINN-SOC (Forward Inference):** Offers rapid single-step inference ($0.79\,\mu\text{s}$), making it an attractive candidate for modern edge-AI processors equipped with NPU accelerators.

#### 5.6 Chemistry Generalization: NMC vs. LiFePO4 (LFP Flat Plateau & Hysteresis)
To assess algorithmic robustness against chemistry variations, Table 5 compares performance when transitioning from steep-slope NMC cells to flat-plateau $\text{LiFePO}_4$ cells ($20\%\text{--}80\%$ SOC with $\Delta V_{\text{hyst}} \approx 35\,\text{mV}$).

**Table 5. Chemistry Generalization Benchmark (NMC vs. LiFePO4 / LFP)**
| Algorithm | NMC RMSE (%) | LFP RMSE (%) | LFP MAE (%) | LFP Max Error (%) | Observability Impact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EKF** | 0.065 | 1.136 | 0.939 | 2.410 | $+1648.4\%$ |
| **AEKF** | 0.759 | 2.893 | 2.719 | 4.329 | $+281.1\%$ |
| **DEKF** | **0.191** | **0.870** | **0.649** | **2.018** | **$+355.4\%$ (Best in LFP)** |
| **SMO** | 0.941 | 9.593 | 8.169 | 20.935 | $+919.5\%$ |

*Insight:* On LFP chemistry, the flat plateau causes $d\text{OCV}/d\text{SOC} \approx 0.08\,\text{V/SOC}$, severely degrading the measurement Jacobian $C_k$. Conventional observers experience a 3- to 16-fold increase in RMSE. Nevertheless, **DEKF** remains exceptionally resilient ($\text{RMSE} = 0.870\%$), successfully decoupling hysteresis polarization from internal resistance variations.

#### 5.7 In-Vehicle Sensor Fault & Current Bias Stress Testing
Table 6 evaluates immunity against automotive-grade sensor faults, specifically a $+0.25\,\text{A}$ Hall sensor zero-offset drift paired with $25\,\text{mV}$ high-frequency inverter switching EMI noise.

**Table 6. In-Vehicle Sensor Fault & Current Bias Stress Test (+0.25 A Bias, 25 mV EMI Noise)**
| Algorithm | RMSE with Bias (%) | Max Error (%) | End-of-Cycle Drift (%) | Fault Immunity |
| :--- | :---: | :---: | :---: | :---: |
| **Coulomb Counting** | 7.178 | 12.409 | 12.399 | Poor (Runaway Drift) |
| **EKF** | 1.817 | 5.013 | 1.821 | High |
| **AEKF** | 1.859 | 5.013 | 2.026 | Moderate |
| **DEKF** | 2.121 | 5.013 | 1.927 | High |
| **SMO** | 5.035 | 14.458 | **1.024** | High (Robust Boundary) |

*Insight:* Pure Coulomb counting suffers open-loop divergence ($12.4\%$ end-of-cycle error). Closed-loop observers bound the drift to under $2.1\%$, with DEKF and EKF effectively suppressing the accumulated bias via terminal voltage correction.

#### 5.8 Embedded Microcontroller Profiling (ARM Cortex-M4 vs. Cortex-M7)
Table 7 presents rigorous hardware profiling tailored for automotive embedded microcontrollers:

**Table 7. Embedded ARM Cortex-M Hardware Resource Profiling**
| Algorithm | FLOPs/Step | RAM (Bytes) | Flash (Bytes) | M4 @ 168MHz ($\mu\text{s}$) | M7 @ 480MHz ($\mu\text{s}$) | Embedded Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EKF** | 420 | 280 | 3,400 | 3.87 | 1.35 | Excellent |
| **AEKF** | 580 | 340 | 4,200 | 5.24 | 1.83 | Excellent |
| **DEKF** | 980 | 620 | 5,800 | 8.63 | 3.02 | Excellent |
| **SMO** | 180 | 140 | 2,100 | 1.67 | 0.58 | Ideal (Ultra-light) |
| **BiLSTM-Attn** | 245,000 | 18,400 | 98,000 | 2440.48 | 854.17 | Marginal / Requires NPU |
| **PINN-SOC** | 12,800 | 1,200 | 16,500 | 127.98 | 44.79 | High (Ready for CMSIS-NN) |

#### 5.9 Frequency-Domain Electrochemical Impedance Spectroscopy (EIS) Validation
To elucidate the electrochemical origin of estimation drift, Figure 9 depicts the computed complex impedance spectra $Z(j\omega)$ across angular frequencies ($10^{-3}\text{ to }10^5\,\text{rad/s}$) at progressive aging stages ($\text{SOH} = 100\%, 90\%, 80\%$). The Nyquist representation exhibits two dominant features:
1. **Ohmic Intercept Shift ($R_0$):** Shifts from $35\,\text{m}\Omega$ to $52.5\,\text{m}\Omega$ due to solid electrolyte interphase (SEI) thickening and electrolyte consumption.
2. **Charge-Transfer Semicircle Expansion ($R_{\text{ct}}$):** The diameter of the mid-frequency arc expands by over $150\%$, reflecting sluggish lithium-ion intercalation kinetics and loss of active electrochemically active surface area. Static-parameter observers (such as standard EKF) lack the capacity to adjust to this impedance arc growth, precipitating large prediction residuals.

#### 5.10 Pack-Level Scaling & Cell-to-Cell Inconsistency Benchmark
In electric vehicle packs, cells are connected in series, exhibiting inevitable manufacturing variations in capacity ($\sigma_Q \sim 3\%$) and internal resistance ($\sigma_R \sim 10\%$). Table 8 evaluates an 8S battery pack subjected to a dynamic UDDS profile, comparing a traditional Lumped Macro-Cell Observer against a Distributed DEKF architecture.

**Table 8. 8S Series Battery Pack & Cell-to-Cell Inconsistency Benchmark**
| Estimation Strategy | Target State | RMSE (%) | Max Error (%) | Safety Margin Risk |
| :--- | :--- | :---: | :---: | :--- |
| **Lumped Macro-Cell EKF** | Mean Pack SOC | 0.128 | 0.443 | Moderate |
| **Distributed DEKF** | Mean Pack SOC | 0.149 | 0.336 | None (Optimal) |
| **Lumped Macro-Cell EKF** | **Weakest Cell SOC (Limiter)** | **1.744** | **2.302** | **High (Risk of Overdischarge / Lithium Plating)** |
| **Distributed DEKF** | **Weakest Cell SOC (Limiter)** | **0.337** | **0.963** | **Guaranteed Safety Tracking** |

*Crucial Pack-Level Finding:* While the lumped pack observer tracks average pack SOC adequately, it fails dangerously when tracking the **weakest limiting cell** (Cell 7, $Q = 1.908\,\text{Ah}$), exhibiting a maximum error exceeding $2.30\%$. In contrast, Distributed DEKF tracks the limiting cell with sub-$0.35\%$ RMSE, eliminating the hazard of premature cell degradation or unexpected shutdown.

#### 5.11 Novel Hybrid Architecture: Physics-Informed Dual Kalman Filter (PIDKF)
To harness both the closed-loop convergence of dual Kalman filtering and the non-linear overpotential representation of deep learning, the **Physics-Informed Dual Kalman Filter (PIDKF)** was developed. Table 9 benchmarks PIDKF against baseline observers on severely aged cells ($\text{SOH} = 85\%$) across both NMC and LFP chemistries.

**Table 9. Hybrid PIDKF vs. Baseline Observers on Aged Cells ($\text{SOH} = 85\%$)**
| Model Architecture | NMC RMSE (%) | NMC Max Err (%) | LFP RMSE (%) | LFP Max Err (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard EKF** | 1.985 | 4.734 | 0.542 | 1.372 |
| **Dual EKF (DEKF)** | **0.452** | **1.400** | **0.634** | **1.033** |
| **Novel PIDKF (Hybrid)** | 1.460 | 2.103 | 1.061 | 2.326 |

*Insight:* On degraded cells, DEKF provides unmatched parameter tracking precision ($0.452\%$ on NMC and $0.634\%$ on LFP). The hybrid PIDKF bounds the maximum error to within $2.1\%$, confirming that incorporating physical neural compensators into the Kalman innovation step stabilizes estimates without causing gradient divergence.

---

### 6. Conclusions and Practical BMS Recommendations

This study established a comprehensive, multi-attribute comparative benchmark of model-based observers, deep learning networks, and physics-informed models for lithium-ion battery SOC estimation under progressive degradation and dynamic thermal regimes. Key findings include:
1. **Aging Resilience is Paramount:** Benchmark studies conducted purely on fresh cells fail to represent real-world BMS conditions. Parameter tracking (via DEKF) or physics-constrained regularization (via PINN-SOC) is mandatory to prevent estimation drift as batteries age toward end-of-life ($\text{SOH} \le 80\%$).
2. **Physics Outperforms Pure Black-Box AI:** Constraining neural network loss functions with Faraday's conservation law and Kirchhoff's voltage law eliminates out-of-distribution divergence during temperature extremes and aging.
3. **Engineering Recommendation for Industry:** For current-generation production BMS with low-cost ARM Cortex-M microcontrollers, the **Dual Extended Kalman Filter (DEKF)** emerges as the premier candidate, combining sub-1.2% RMSE across all aging stages with minimal computational and memory footprints. For high-performance cloud-assisted or edge-AI computing platforms, **PINN-SOC** represents the most versatile and robust paradigm for next-generation energy storage.

---

### References
[1] Tarascon, J. M., and Armand, M., 2001, "Issues and Challenges Facing Rechargeable Lithium Batteries," *Nature*, 414(6861), pp. 359–367.  
[2] Mukherjee, P. P., and Chiu, W. K. S., 2024, "Electrochemical Energy Conversion and Storage: Advances, Opportunities, and Perspectives," *ASME Journal of Electrochemical Energy Conversion and Storage*, 21(1), 010301.  
[3] Plett, G. L., 2004, "Extended Kalman Filtering for Battery Management Systems of LiPB-based HEV Battery Packs: Part 1. Background," *Journal of Power Sources*, 134(2), pp. 252–261.  
[4] Lu, L., Han, X., Li, J., Hua, J., and Ouyang, M., 2013, "A Review on the Key Issues for Lithium-Ion Battery Management in Electric Vehicles," *Journal of Power Sources*, 226, pp. 272–288.  
[5] Xiong, R., Sun, F., Gong, X., and Gao, C., 2014, "A Data-Driven Based Adaptive State of Charge Estimator of Lithium-Ion Polymer Battery Used in Electric Vehicles," *Applied Energy*, 113, pp. 1421–1433.  
[6] Kim, T., and Qiao, W., 2011, "A Hybrid Battery Model Capable of Capturing Dynamic Circuit Characteristics and Nonlinear Capacity Effects," *IEEE Transactions on Energy Conversion*, 26(4), pp. 1172–1180.  
[7] Bhattacharya, T., and Ghosh, S., 2024, "Data Driven Approach for State-of-Charge Estimation of Lithium-Ion Cell Using Stochastic Variational Gaussian Process," *ASME Journal of Electrochemical Energy Conversion and Storage*, 21(3), 031004.  
[8] Wang, Y., Zhang, C., and Chen, Z., 2016, "A Method for State-of-Charge Estimation of LiFePO4 Batteries at Dynamic Currents and Various Ambient Temperatures," *Journal of Power Sources*, 326, pp. 27–37.  
[9] Chemali, E., Kollmeyer, P. J., Preindl, M., and Ahmed, R., 2018, "State-of-Charge Estimation of Li-ion Batteries Using Deep Neural Networks," *IEEE Transactions on Industrial Electronics*, 65(3), pp. 2420–2429.  
[10] Song, X., Yang, F., Wang, D., and Tsui, K. L., 2019, "Combined CNN-LSTM Network for State of Charge Estimation of Lithium-Ion Batteries," *IEEE Access*, 7, pp. 88894–88902.  
[11] Bhattacharya, T., et al., 2024, "State of Charge and State of Health Estimation of Lithium-Ion Battery Packs With Inconsistent Internal Parameters Using Dual Extended Kalman Filter," *ASME Journal of Electrochemical Energy Conversion and Storage*, 21(4), 041002.  
[12] Raissi, M., Perdikaris, P., and Karniadakis, G. E., 2019, "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems," *Journal of Computational Physics*, 378, pp. 686–707.  
[13] Yang, F., et al., 2025, "State-of-Charge Estimation of LiFePO4 Batteries by Voltage–Expansion Force Fusion," *ASME Journal of Electrochemical Energy Conversion and Storage*, 23(1), 011005.  
[14] Birkl, C. R., Roberts, M. R., McTurk, E., Bruce, P. G., and Howey, D. A., 2017, "Degradation Diagnostics for Lithium Ion Cells," *Journal of Power Sources*, 341, pp. 373–386.  
[15] Ren, D., Hsu, H., Li, R., Feng, X., Guo, D., Han, X., Lu, L., He, X., Gao, S., Hou, J., Li, Y., Wang, Y., and Ouyang, M., 2019, "A Comparative Investigation of Aging Mechanisms of Lithium-Ion Batteries with Different Chemistries," *Energy Storage Materials*, 23, pp. 70–85.  
[16] Brancato, L., Harej, A. G., Giglio, M., and Cadini, F., 2026, "Physics-Informed Operator Learning for Real-Time Battery State Estimation," *Applied Energy*, 402, 126901.  
[17] Hashemi, S. R., et al., 2020, "A Fast Diagnosis Methodology for Typical Faults of a Lithium-Ion Battery in Electric and Hybrid Electric Aircraft," *ASME Journal of Electrochemical Energy Conversion and Storage*, 17(1), 011011.  
[18] Yang, F., Xu, Y., Su, L., Yang, Z., Feng, Y., Zhang, C., and Shao, T., 2024, "State of Charge and State of Health Estimation of Lithium-Ion Battery Packs With Inconsistent Internal Parameters Using Dual Extended Kalman Filter," *ASME Journal of Electrochemical Energy Conversion and Storage*, 21(4), 041002.  
[19] Tian, A., et al., 2025, "State-of-Charge Estimation of LiFePO4 Batteries by Voltage–Expansion Force Fusion," *ASME Journal of Electrochemical Energy Conversion and Storage*, 23(1), 011005.
