# COVER LETTER FOR SUBMISSION TO JOURNAL OF ENERGY STORAGE (ELSEVIER)

**Date:** November 19, 2026

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
Accurate and robust State-of-Charge (SOC) estimation remains a foundational prerequisite for safety, optimal utilization, and lifespan extension in Battery Energy Storage Systems (BESS) and electric vehicles (EVs). Although the literature spans classical Extended Kalman Filters (EKF), Adaptive EKF (AEKF), Unscented Kalman Filters (UKF), Dual EKF (DEKF), Sliding Mode Observers (SMO), deep sequence networks (BiLSTM-Attn), and Physics-Informed Neural Networks (PINN-SOC), most published benchmarks evaluate pristine cells near room temperature ($25^\circ\text{C}$).

In field operation, however, lithium-ion batteries experience simultaneous capacity fade down to $80\%$ state-of-health (SOH), solid-electrolyte interphase (SEI) and charge-transfer resistance growth, ambient excursions ($0^\circ\text{C}$ to $45^\circ\text{C}$), flat-plateau $\text{LiFePO}_4$ (LFP) observability limits, sensor faults, and pack-level thermal-aging inconsistency. How model-based, data-driven, and physics-informed observers break down under strictly blind fresh-prior initialization ($100\%$ SOH, $25^\circ\text{C}$, zero oracle health leakage)---and how they fit on automotive microcontrollers---has remained inadequately quantified.

### Key Contributions and Findings of this Work
Aligned with the editorial scope of the *Journal of Energy Storage*, this work delivers a dual-tier benchmark pairing **Tier 1 public laboratory cycler datasets** (held-out **XJTU Batch-5** NMC532 dynamic Random Walk tests and **NASA Ames PCoE** aging, $4^\circ\text{C}\text{--}43^\circ\text{C}$ thermal, and EIS spectra) with **Tier 2 calibrated electro-thermal-aging co-simulation** across 10 operational stress phases:

1. **Quantifying Progressive Aging Divergence:** Strictly initialized from fresh priors without oracle aging leakage, fixed-parameter EKF exhibits a **55.9-fold RMSE escalation** ($0.067\% \to 3.745\%$, Aging Sensitivity Factor $+5489.6\%$) from $100\%$ down to $80\%$ SOH as unmodeled ohmic and charge-transfer impedance growth ($R_{\text{ct}}$ broadened by $150\%$ in the plant law and corroborated directly via Randles--CPE fitting on real NASA \texttt{B0006} EIS spectra, $22.37 \to 38.89\;\mathrm{m}\Omega$) is misattributed to OCV shifts. Dual parameter tracking (DEKF) lowers $80\%$ SOH error to $2.073\%$, while our proposed hybrid **PIDKF** uniquely holds severely aged $80\%$ SOH error below $1\%$ ($0.928\pm0.027\%$).
2. **Thermal Extremes, Flat-Plateau LFP Observability, and Pack Scaling:** On biphasic LFP cells, dynamic hysteresis co-estimation removes $\pm 30\;\mathrm{mV}$ charge--discharge voltage bias to cut fresh tracking error nearly eightfold ($0.401\% \to 0.049\text{--}0.051\%$) alongside strictly positive-definite 30-s LTV observability Gramians ($\lambda_{\min}(\mathcal{G}_o) \ge 1.09\times 10^{-7}$ on UDDS). Across $0^\circ\text{C}\text{--}45^\circ\text{C}$, PIDKF attains $2.526\pm0.098\%$ ($0^\circ\text{C}$) and $0.573\pm0.019\%$ ($45^\circ\text{C}$). In an 8S series pack with cell-to-cell SEI/thermal inconsistency, distributed PIDKF reduces weakest-cell \#4 RMSE to $0.544\%$ ($0.223\%$ mean pack RMSE), eliminating lumped-observer overdischarge risk.
3. **Physics-Informed Dual Kalman Filter (PIDKF) & Tier 1 Validation:** PIDKF augments an online DEKF core ($\hat{R}_{0,k}, 1/\hat{Q}_{\text{act},k}$) with a $257$-parameter Butler--Volmer-passivity neural overpotential compensator conditioned on fast-converging $\hat{R}_{0,k}^-$ and protected via *exogenous sensitivity decoupling* (avoiding neural-Jacobian gain destabilization). Five-seed Tier 2 ablation demonstrates $0.554\pm0.018\%$ RMSE (aged $85\%$ NMC) and $0.593\pm0.048\%$ (aged $85\%$ LFP). On held-out Tier 1 cyclers, PIDKF achieves $1.425\text{--}2.750\%$ RMSE on XJTU Random Walk tests, $0.636\text{--}1.715\%$ on NASA \texttt{B0007} ($24^\circ\text{C}$), and cuts cold $4^\circ\text{C}$ error by $63\%$ vs.\ EKF under NASA \texttt{B0055/B0056} capacity collapse, supported by discrete ISS/UUB proofs.
4. **Static ANSI-C99 Reference Kernels & Analytical ARMv7E-M Sizing:** Zero-heap C99 implementations with exported `float32` weights (`embedded_c/pidkf_weights.h`, PyTorch parity $\le 2.64\times 10^{-5}\;\mathrm{V}$) audit PIDKF at **$364\;\mathrm{B}$ RAM**, **$7,428\;\mathrm{B}$ ($7.25\;\mathrm{KB}$) Flash**, **$1,540$ FLOPs/step**, and analytical **$17.74\;\upmu\mathrm{s}$** latency at $168\;\mathrm{MHz}$ Cortex-M4F ($4.28\;\upmu\mathrm{s}$ at $480\;\mathrm{MHz}$ Cortex-M7; $0.28\%$ CPU load for $16$ cells at $10\;\mathrm{Hz}$), sharply contrasting with BiLSTM-Attn ($16.96\;\mathrm{ms}$, infeasible for real-time packs).

### Formatting and Integrity Confirmation
In alignment with Elsevier's *"Your Paper Your Way"* policy, this manuscript is submitted in clean, universal single-column LaTeX layout (`article` class, 11pt, 2.5cm margins). We confirm that:
- This manuscript is original, has not been published previously, and is not under consideration elsewhere.
- All authors have reviewed and approved the final manuscript.
- All Tier 1 datasets (XJTU Batch-5, NASA Ames PCoE), benchmark scripts, EIS tools, and static ANSI-C99 kernels are openly hosted on GitHub (`https://github.com/HuyHoangLe0201/SOC-Observer-PINN-Benchmark`).
- There are no competing interests or external funding conflicts to declare.

### Suggested Expert Reviewers
1. **Prof. Gregory L. Plett** (University of Colorado Colorado Springs, USA)  
   *Expertise:* Extended Kalman Filtering, Dual Kalman Filter, Battery Management Systems, Li-ion ECM modeling.  
   *Email:* `gplett@uccs.edu`
2. **Prof. Rui Xiong** (Beijing Institute of Technology, China)  
   *Expertise:* Battery state estimation, machine learning for energy storage, electro-thermal co-estimation.  
   *Email:* `rxiong@bit.edu.cn`
3. **Prof. David A. Howey** (University of Oxford, United Kingdom)  
   *Expertise:* Battery modeling, parameter estimation, electrochemical impedance spectroscopy (EIS), BMS diagnostics.  
   *Email:* `david.howey@eng.ox.ac.uk`
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
