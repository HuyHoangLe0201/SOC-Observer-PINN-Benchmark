# COVER LETTER FOR SUBMISSION TO JOURNAL OF ENERGY STORAGE

**Date:** November 8, 2026  

**To:**  
Prof. Dirk Uwe Sauer  
Editor-in-Chief  
*Journal of Energy Storage* (Elsevier)  
Chair of Electrochemical Energy Conversion and Storage Systems,  
RWTH Aachen University, Aachen, Germany  

---

**Dear Professor Sauer and the Editorial Board,**

We are pleased to submit our original research manuscript entitled:

> **"Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes"**

by Huy Hoang Le and Kim-Anh Nguyen (Corresponding Author), for publication consideration as a full **Research Paper** in the *Journal of Energy Storage* (Elsevier).

---

### Motivation and Core Contributions

Accurate and robust State-of-Charge (SOC) estimation across operational aging ($100\%$ down to $80\%$ SOH) and thermal variations ($0^\circ\text{C}$ to $45^\circ\text{C}$) is critical for both electric vehicle traction batteries and stationary Battery Energy Storage Systems (BESS). However, existing literature predominantly benchmarks state observers on pristine cells at $25^\circ\text{C}$, obscuring catastrophic estimator drift when electrochemical impedance expands during real-world lifespan degradation.

In this work, we present an exhaustive dual-tier benchmark comprising direct laboratory cycler experiments (Tier 1) and calibrated multi-physical co-simulation (Tier 2). We evaluate six representative estimation paradigms across ten challenging operational phases:
1. **Classical & Adaptive Observers:** Extended Kalman Filter (EKF), Adaptive EKF (AEKF with Sage-Husa covariance tracking), Dual Extended Kalman Filter (DEKF with co-estimated capacity and ohmic resistance), and Sliding Mode Observer (SMO with boundary-layer chattering attenuation).
2. **Deep Learning & Physics-Informed Architectures:** Bidirectional LSTM with Multi-Head Self-Attention (BiLSTM-Attn) and Physics-Informed Neural Networks (PINN-SOC) embedding Faradaic charge conservation and Kirchhoff's voltage law into backpropagation.
3. **Proposed Hybrid Architecture (PIDKF):** A Physics-Informed Dual Kalman Filter coupling dynamic hysteresis tracking on flat-plateau $\text{LiFePO}_4$ (LFP) with neural overpotential compensation on NMC chemistry.

### Key Quantitative Findings

- **Aging Vulnerability:** While pristine EKF achieves $0.065\%$ RMSE at Beginning-of-Life, uncompensated aging to $80\%$ SOH induces a **45.7-fold error surge** (RMSE escalates to $2.973\%$, $+4473.8\%$ Aging Sensitivity Factor) due to interfacial charge-transfer resistance ($R_{\text{ct}}$) expansion ($>150\%$).
- **Lifespan Resilience:** Online parameter-tracking DEKF confines aged RMSE to $0.585\%$ across the battery lifespan.
- **Chemistry Complementarity:** The proposed PIDKF achieves Pareto-optimal tracking on degraded cells ($85\%$ SOH): $\text{RMSE} = 0.414\%$ on NMC ($0.428\%$ on CALCE commercial aged cells) and $\mathbf{0.149\%}$ on LFP ($0.158\%$ experimental), bounding thermal-extreme errors on NASA PCoE cells to $0.842\%$ ($4^\circ\text{C}$) and $0.326\%$ ($43^\circ\text{C}$).
- **Embedded MCU Execution:** Bare-metal profiling on ARM Cortex-M microcontrollers demonstrates execution times of $14.20\;\upmu\mathrm{s}$ on Cortex-M4 ($2,386$ DWT clock cycles) and an estimated $4.97\;\upmu\mathrm{s}$ on Cortex-M7 ($0.23\%$ CPU loading for 16 series cells at $100\text{ ms}$), proving direct viability for industrial Battery Management Systems.

---

### Scope Alignment with *Journal of Energy Storage*

This manuscript is squarely aligned with the aims and scope of the *Journal of Energy Storage*, addressing:
- Battery management and state estimation for stationary and mobile energy storage;
- Electrochemical degradation and impedance evolution across thermal regimes;
- Chemistry-specific algorithms for nickel-manganese-cobalt (NMC) and lithium iron phosphate (LFP) cells;
- Hardware-in-the-loop embedded deployment on automotive-grade microcontrollers.

---

### Declarations

- **Novelty & Non-Duplication:** This manuscript is original, has not been published previously, and is not currently under consideration for publication elsewhere.
- **Author Agreement:** All authors have approved the manuscript and agree with its submission to the *Journal of Energy Storage*.
- **Data Availability & Reproducibility:** All datasets (CALCE, NASA Ames PCoE, Oxford), calibrated 2-RC model parameters, and optimized CMSIS-DSP C kernels are openly available on GitHub and permanently openly hosted on GitHub (`DOI: 10.5281/zenodo.14048921`).
- **Conflict of Interest:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

### Suggested Reviewers

1. **Prof. Gregory L. Plett**  
   Department of Electrical and Computer Engineering, University of Colorado Colorado Springs, USA  
   *Email:* gplett@uccs.edu  
   *Expertise:* Kalman filtering, dual estimation, battery modeling for energy storage systems.

2. **Prof. Minggao Ouyang**  
   State Key Laboratory of Automotive Safety and Energy, Tsinghua University, Beijing, China  
   *Email:* ouymg@tsinghua.edu.cn  
   *Expertise:* Battery degradation, thermal runaway, State-of-Health and State-of-Charge co-estimation.

3. **Prof. Rui Xiong**  
   School of Mechanical Engineering, Beijing Institute of Technology, Beijing, China  
   *Email:* rxiong@bit.edu.cn  
   *Expertise:* Machine learning in battery systems, adaptive filtering, pack-level state estimation.

4. **Dr. Christoph R. Birkl**  
   Department of Engineering Science, University of Oxford, Oxford, UK  
   *Email:* christoph.birkl@eng.ox.ac.uk  
   *Expertise:* Electrochemical impedance spectroscopy, degradation diagnostics in lithium-ion cells.

---

We thank you for your time and consideration of our manuscript.

Sincerely,

**Kim-Anh Nguyen, Ph.D.** *(Corresponding Author)*  
Faculty of Electrical Engineering  
University of Science and Technology, The University of Danang  
54 Nguyen Luong Bang, Da Nang 550000, Vietnam  
*Email:* nkanh@dut.udn.vn  
*Tel:* +84 236 3842308  

**Huy Hoang Le, M.Sc.**  
R&D Department, PowerMore Ltd.  
Da Nang 550000, Vietnam  
*Email:* hoang.lehuy@powermore.vn  
