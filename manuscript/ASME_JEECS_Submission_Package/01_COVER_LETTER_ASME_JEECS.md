# COVER LETTER FOR SUBMISSION TO ASME JEECS

**Date:** November 4, 2026  

**To:**  
Prof. Partha P. Mukherjee  
Editor-in-Chief  
*ASME Journal of Electrochemical Energy Conversion and Storage*  
School of Mechanical Engineering, Purdue University  
West Lafayette, IN 47907, USA  

---

**Dear Professor Mukherjee and the Editorial Board,**

We are pleased to submit our original research manuscript entitled:

> **"Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes"**

by Huy Hoang Le and Kim-Anh Nguyen (Corresponding Author), for publication consideration as a full **Research Paper** in the *ASME Journal of Electrochemical Energy Conversion and Storage* (JEECS).

---

### Motivation and Core Contributions

Accurate online State-of-Charge (SOC) estimation is foundational for driving range prediction, fast-charging safety, and lifetime preservation in automotive battery management systems (BMS). While battery state estimation has generated thousands of publications, existing literature exhibits five critical limitations:
1. **Beginning-of-Life Bias:** Algorithmic comparisons predominantly evaluate pristine cells ($100\%$ SOH) at room temperature ($25^\circ\text{C}$), completely obscuring observer degradation under real-world operational aging.
2. **Thermal Kinetics Decoupling:** Non-isothermal conditions ($0^\circ\text{C}$ to $45^\circ\text{C}$) that dramatically alter electrolyte conductivity and interfacial Butler–Volmer charge-transfer kinetics are rarely investigated comparatively.
3. **Flat Plateau Observability Loss:** Tracking over the zero-slope biphasic plateau of $\text{LiFePO}_4$ (LFP) without auxiliary mechanical strain sensors remains an unresolved challenge.
4. **Sensor Bias and Inverter EMI Stress:** Simultaneous current sensor Hall bias drift and PWM inverter noise are neglected in pristine bench tests.
5. **Embedded Hardware Disconnect:** Algorithms are typically benchmarked on high-performance desktop CPUs/GPUs, with little validation of cycle-accurate constraints on automotive microcontrollers (ARM Cortex-M) or series pack scalability under cell-to-cell parameter inconsistency.

To resolve these challenges, our study establishes an exhaustive multi-attribute benchmark evaluating 6 representative state estimation paradigms across 10 operational stress phases:
* **The Aging Sensitivity Factor (ASF):** We formulate and quantify the ASF, demonstrating that uncompensated Extended Kalman Filters (EKF) suffer a staggering **45-fold accuracy collapse (+4,473.8% ASF)** as SOH degrades from $100\%$ down to $80\%$, whereas parameter-tracking Dual EKF (DEKF) confines lifespan RMSE to **$0.585\%$**.
* **The Frequency-Domain Impedance Bridge:** We establish a direct analytical and experimental bridge connecting time-domain observer tracking drift to frequency-domain Electrochemical Impedance Spectroscopy (EIS) Nyquist spectra ($100\text{ kHz}$ to $1\text{ mHz}$), proving that a $>150\%$ expansion in charge-transfer arc ($R_{\text{ct}}$) drives the observed time-domain drift.
* **The Hybrid Physics-Informed Dual Kalman Filter (PIDKF):** We propose a lightweight hybrid architecture augmenting DEKF co-estimation with physics-informed parameter bounds, dynamic hysteresis tracking, and neural polarization compensation. PIDKF achieves **$0.414\%$ RMSE on degraded NMC** and **$0.149\%$ on degraded LFP** ($85\%$ SOH), with bounded-input bounded-state (BIBS) stochastic stability analytically proven.
* **Pareto Frontier on Automotive Microcontrollers:** We conduct cycle-accurate execution profiling on ARM Cortex-M4 (168 MHz) and Cortex-M7 (480 MHz) microcontrollers using CMSIS-DSP, mapping the computational complexity (FLOPs, SRAM, latency) versus accuracy trade-off. PIDKF achieves Pareto optimality, executing in just **$14.20\;\mu\text{s}$** per time step (well within the $100\text{ ms}$ control loop of industrial BMS).

---

### Alignment with ASME JEECS Scope

This work directly aligns with the mission of the *ASME Journal of Electrochemical Energy Conversion and Storage*, specifically within the technological focus areas of:
* *Electrochemical energy storage and conversion*
* *Battery management systems, modeling, state estimation, and lifetime prognosis*
* *Thermal management, degradation mechanics, and safety tolerance*
* *Applied physics-informed machine learning for energy devices*

Our paper provides both deep theoretical rigor (Filippov solutions, Hermann–Krener observability, Lyapunov sliding manifold invariance, and Joseph covariance stabilization) and actionable practical guidelines for EV and grid-scale storage engineers.

---

### Statements and Declarations

1. **Originality:** This manuscript is an original work that has not been published previously and is not under consideration for publication elsewhere, in whole or in part.
2. **Authorship & Consent:** All co-authors have actively contributed to the research, reviewed the manuscript, and approved its submission to ASME JEECS.
3. **Conflicts of Interest:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.
4. **Data Availability:** All benchmark cycle datasets (Samsung INR18650-20R and CALCE A123 26650) are open-access, and the algorithmic implementation codebase is available for full reproducibility.
5. **Funding:** This research received no external commercial funding.

We believe this paper will be of strong interest to the multidisciplinary readership of ASME JEECS across mechanical engineering, electrochemical systems, and automotive control. Thank you very much for your time and consideration of our manuscript.

Sincerely yours,

**Kim-Anh Nguyen, Ph.D.** *(Corresponding Author)*  
Faculty of Electrical Engineering  
University of Science and Technology, The University of Danang  
54 Nguyen Luong Bang, Da Nang 550000, Vietnam  
Email: `nkanh@dut.udn.vn`  
ORCID: 0000-0003-3408-847X  

**Huy Hoang Le**  
R&D Department, PowerMore Ltd.  
Da Nang 550000, Vietnam  
ORCID: 0009-0009-2138-5468  
