# COVER LETTER FOR SUBMISSION TO JOURNAL OF ENERGY STORAGE (ELSEVIER)

**Date:** November 8, 2026

**To:**  
The Editor-in-Chief  
*Journal of Energy Storage*  
Elsevier Science  

**Subject:** Submission of Original Research Article: *"Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes"*

**Authors:**  
Huy Hoang Le$^1$, Kim-Anh Nguyen$^{2,*}$  
$^1$ R&D Department, PowerMore Ltd., Da Nang 550000, Vietnam  
$^2$ Faculty of Electrical Engineering, University of Science and Technology, The University of Danang, Da Nang 550000, Vietnam  
$^*$ Corresponding Author: Kim-Anh Nguyen (Email: `nkanh@dut.udn.vn`)

---

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript entitled **"Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes"** for consideration as a regular research article in the *Journal of Energy Storage*.

### Context and Motivation
Accurate and robust State-of-Charge (SOC) estimation remains a paramount prerequisite for safety, optimal utilization, and lifespan extension of modern Battery Energy Storage Systems (BESS) and electric vehicles (EVs). Although the literature is replete with algorithmic proposals—ranging from classical Extended Kalman Filters (EKF), Adaptive filters (AEKF), and Sliding Mode Observers (SMO) to data-driven deep recurrent networks (BiLSTM-Attn) and Physics-Informed Neural Networks (PINN)—the overwhelming majority of published benchmarks restrict their validation to pristine, unaged cells operated at room temperature ($25^\circ\text{C}$). 

In practical stationary storage and vehicular operations, however, lithium-ion batteries inevitably undergo continuous capacity fade, severe solid-electrolyte interphase (SEI) growth, charge-transfer impedance rise, wide environmental temperature fluctuations ($0^\circ\text{C}$ to $45^\circ\text{C}$), and cell-to-cell manufacturing and thermal inconsistency. Crucially, algorithmic failure modes and parameter cross-coupling that emerge under these coupled degradation-thermal stresses remain poorly quantified.

### Key Contributions and Findings of this Work
In accordance with the editorial scope of the *Journal of Energy Storage*, this work presents a systematic, dual-tier experimental and calibrated co-simulation benchmark across 10 operational phases:
1. **Uncovering Severe Aging Degradation Divergence:** We demonstrate that under progressive aging ($100\% \to 80\%$ SOH), standard uncompensated EKF exhibits a **45.7-fold RMSE escalation** ($0.065\%$ to $2.973\%$, Aging Sensitivity Factor $+4473.8\%$) because unmodeled ohmic and polarization overpotential growth ($I \cdot \Delta R_0$) is erroneously misattributed to thermodynamic open-circuit voltage (OCV) shifts. Adaptive covariance tuning (AEKF) fails to prevent this deterioration ($5.406\%$ RMSE at $80\%$ SOH). In contrast, dual-state-parameter tracking (DEKF) dynamically tracks internal resistance and active capacity online, confining full-lifespan RMSE to $0.585\%$.
2. **Thermal Extremes and Chemistry Disparities:** We evaluate algorithmic robustness across $0^\circ\text{C}$, $25^\circ\text{C}$, and $45^\circ\text{C}$, and across distinct cathode chemistries: sloping-OCV $\text{LiNi}_{0.8}\text{Co}_{0.15}\text{Al}_{0.05}\text{O}_2$ (NMC) and flat-plateau $\text{LiFePO}_4$ (LFP). We identify the physical mechanism of cold-temperature parameter cross-coupling in DEKF ($2.254\%$ error at $0^\circ\text{C}$) caused by unmodeled $R_1$ surge. Furthermore, we establish the analytical conditions under which dynamic hysteresis tracking restores practical observability on flat LFP plateaus ($\lambda_{\min}(\mathcal{G}_o) \ge 1.21 \times 10^{-3}$), whereas unretrained neural networks experience severe out-of-distribution drift.
3. **Formulation and Validation of the Hybrid PIDKF:** We propose a novel *Physics-Informed Dual Kalman Filter (PIDKF)* that integrates an online parameter-tracking DEKF core with an exogenous physics-constrained neural overpotential compensator governed by Butler–Volmer/Tafel dissipative passivity. The neural network absorbs thermal-kinetic residuals without destabilizing the Kalman filter. We prove discrete-time Input-to-State Stability (ISS) and mean-square uniform ultimate boundedness (UUB). Experimental validation on aged CALCE cells ($85\%$ SOH) demonstrates $0.428\%$ RMSE (NMC) and $0.158\%$ RMSE (LFP), while evaluations on NASA PCoE cells under thermal stress ($4^\circ\text{C}$ and $43^\circ\text{C}$) maintain RMSE $<0.85\%$, outperforming all benchmarked baselines.
4. **Pack-Level Inconsistency and Bare-Metal Embedded Profiling:** We evaluate an 8S series pack subjected to realistic cell-to-cell dispersion and thermal gradients, demonstrating that lumped string observers incur dangerous $2.302\%$ peak errors on the weakest cell, which distributed DEKF safely confines to $0.337\%$ RMSE. Finally, we provide exact bare-metal execution timings measured on ARM Cortex-M4 microcontrollers ($168\text{ MHz}$) using ARM GCC 12.3 and CMSIS-DSP, mapping out the multi-attribute Pareto efficiency frontier for industrial BMS deployment.

### Formatting and "Your Paper Your Way" Policy
In alignment with Elsevier's *"Your Paper Your Way"* submission guidelines, this manuscript is submitted in a clean, universal, unformatted single-column LaTeX layout (`article` class, 11pt, 2.5cm margins, standard numbering, fully integrated high-resolution figures, and comprehensive reference citations) to facilitate unimpeded, thorough technical evaluation by the reviewers.

We confirm that:
- This manuscript is original, has not been published previously, and is not currently under consideration for publication elsewhere.
- All authors have reviewed and approved the submitted version of the manuscript.
- All experimental datasets utilized are publicly accessible (CALCE, Oxford, NASA PCoE), and the full reproducible simulation platform and bare-metal embedded profiling code are archived on Zenodo (DOI: `10.5281/zenodo.14028912`) and GitHub.
- There are no conflicts of interest to declare.

### Suggested Expert Reviewers
1. **Prof. Gregory L. Plett** (University of Colorado Colorado Springs, USA)  
   *Expertise:* Extended Kalman Filtering, Dual Kalman Filter, Battery Management Systems, Li-ion ECM modeling.  
   *Email:* `gplett@uccs.edu`
2. **Prof. Rui Xiong** (Beijing Institute of Technology, China)  
   *Expertise:* Battery state estimation, machine learning for energy storage, electro-thermal co-estimation.  
   *Email:* `rxiong@bit.edu.cn`
3. **Prof. Dirk Uwe Sauer** (RWTH Aachen University, Germany)  
   *Expertise:* Battery aging, impedance spectroscopy (EIS), pack-level battery diagnostics and storage systems.  
   *Email:* `dirkuwe.sauer@isea.rwth-aachen.de`
4. **Prof. Datong Liu** (Harbin Institute of Technology, China)  
   *Expertise:* Physics-informed neural networks, battery degradation prognostics, Kalman filtering.  
   *Email:* `liudatong@hit.edu.cn`

Thank you very much for your time, consideration, and management of the peer-review process.

Sincerely,

**Kim-Anh Nguyen, Ph.D.** *(Corresponding Author)*  
Faculty of Electrical Engineering, University of Science and Technology, The University of Danang  
54 Nguyen Luong Bang St., Lien Chieu District, Da Nang 550000, Vietnam  
Email: `nkanh@dut.udn.vn`  

**Huy Hoang Le**  
R&D Department, PowerMore Ltd., Da Nang 550000, Vietnam  
Email: `hoang.le@powermore.vn`
