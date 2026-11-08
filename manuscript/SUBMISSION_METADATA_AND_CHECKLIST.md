# SUBMISSION METADATA AND COMPLIANCE CHECKLIST
## ASME Journal of Electrochemical Energy Conversion and Storage (JEECS)

---

### 1. MANUSCRIPT METADATA

| Item | Details |
| :--- | :--- |
| **Journal Title** | *ASME Journal of Electrochemical Energy Conversion and Storage* (JEECS) |
| **Editor-in-Chief** | Prof. Partha P. Mukherjee (Purdue University) |
| **Manuscript Title** | **Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes** |
| **Manuscript Type** | Full Archival Research Paper |
| **Subject / Technical Area** | Electrochemical Energy Storage; Battery Management Systems; Modeling, State Estimation, and Diagnostics; Physics-Informed Machine Learning |
| **Target Page Count** | Exactly 13 Pages (Standard ASME 2-Column Archival Format via `asmejour.cls`) |
| **Total Words** | ~9,800 words |
| **Total Figures** | 11 Figures (All full column width: $239.8\text{ pt} = 8.46\text{ cm} = 3.33\text{ inches}$, 300+ DPI) |
| **Total Tables** | 10 Tables (Fully populated with empirical benchmark data, calibrated parameters, and physical units) |
| **Total References** | 50 References (All with verified DOIs, formatted in ASME bracketed style) |

---

### 2. AUTHOR INFORMATION & AFFILIATIONS

#### **Corresponding Author**
* **Full Name:** Kim-Anh Nguyen, Ph.D.
* **Academic Title:** Associate Professor / Senior Lecturer
* **Affiliation:** Faculty of Electrical Engineering, University of Science and Technology, The University of Danang
* **Address:** 54 Nguyen Luong Bang, Lien Chieu District, Da Nang 550000, Vietnam
* **Institutional Email:** `nkanh@dut.udn.vn`
* **ORCID:** [0000-0003-3408-847X](https://orcid.org/0000-0003-3408-847X)
* **Role:** Conceptualization, Supervision, Formal Analysis, Writing – Review & Editing, Funding Acquisition.

#### **First Author**
* **Full Name:** Huy Hoang Le
* **Affiliation:** Research & Development Department, PowerMore Ltd.
* **Address:** Da Nang 550000, Vietnam
* **Email:** `hoangle@powermore.vn` / `huyhoang.le@alumni.dut.udn.vn`
* **ORCID:** [0009-0009-2138-5468](https://orcid.org/0009-0009-2138-5468)
* **Role:** Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing – Original Draft, Visualization.

---

### 3. RESEARCH HIGHLIGHTS
*(Strict ASME/Elsevier style: 4 bullet points, maximum 85 characters per bullet)*

* **Benchmark of 6 SOC estimation paradigms across 10 operational stress phases** (77 chars)
* **EKF shows 45x error surge (+4474% ASF) as SOH fades; DEKF bounds RMSE to 0.58%** (80 chars)
* **EIS Nyquist bridge connects >150% Rct expansion directly to observer drift** (77 chars)
* **Hybrid PIDKF delivers 0.41% RMSE in 14.2 us on ARM Cortex-M4 microcontrollers** (79 chars)

---

### 4. ABSTRACT

Although battery State-of-Charge (SOC) estimation has been extensively investigated, existing comparative studies predominantly evaluate fresh cells at room temperature, obscuring algorithmic vulnerabilities under operational aging and thermal extremes. This work establishes a dual-tier benchmark of six representative nonlinear observers, deep recurrent networks, and physics-informed architectures (EKF, AEKF, DEKF, SMO, BiLSTM-Attn, and PINN-SOC) combining direct experimental cycler datasets with calibrated electro-thermal-aging co-simulation. Methodologies are systematically assessed across progressive capacity fade ($100\% \rightarrow 80\%$ SOH), wide ambient temperatures ($0^\circ\text{C}$ to $45^\circ\text{C}$), flat-plateau $\text{LiFePO}_4$ thermodynamics, and 8S series pack inconsistency. Results reveal that whereas uncompensated EKF suffers a 45.7-fold RMSE escalation ($0.065\%$ to $2.973\%$) as interfacial impedance expands, online parameter-tracking DEKF confines lifespan RMSE to $0.585\%$. Furthermore, we formulate a hybrid Physics-Informed Dual Kalman Filter (PIDKF) that reveals chemistry-dependent complementarity---neural overpotential compensation dominates on sloping-OCV NMC ($0.414\%$ RMSE), whereas dynamic hysteresis tracking dominates on flat-plateau LFP ($0.149\%$ RMSE) under calibrated co-simulation for the embedded exogenous-Jacobian configuration ($0.412\%$ and $0.148\%$ for exact Jacobian in Table 9). Finally, bare-metal profiling on ARM Cortex-M microcontrollers ($14.20\;\mu\text{s}$ measured on Cortex-M4; $4.97\;\mu\text{s}$ constant-IPC clock-scaled estimate on Cortex-M7) delineates the Pareto efficiency frontier for industrial battery management systems.

**Keywords:** State-of-charge; Lithium-ion battery; Battery degradation; Dual Extended Kalman filter; Physics-informed neural network; Electrochemical impedance spectroscopy; Embedded BMS; ASME JEECS.

---

### 5. SUGGESTED INDEPENDENT REVIEWERS

The nominated peer reviewers are internationally recognized experts in battery state estimation, electrochemical modeling, and physics-informed machine learning. None of the nominated reviewers have joint publications, shared grants, or institutional affiliations with the authors within the past 5 years.

#### Reviewer 1: Prof. Gregory L. Plett
* **Title & Affiliation:** Professor of Electrical and Computer Engineering, University of Colorado Colorado Springs (UCCS), USA.
* **Email:** `gplett@uccs.edu`
* **Expertise:** Renowned pioneer of Extended Kalman Filtering, Dual Kalman Filtering, and Sigma-Point Kalman Filtering for lithium-ion battery management; author of the foundational textbook series *Battery Management Systems* (Artech House).
* **Justification:** Ideal authority to critically evaluate the algorithmic formulation of EKF, AEKF, and DEKF, covariance adaptation, and parameter observability proofs.

#### Reviewer 2: Prof. Simona Onori
* **Title & Affiliation:** Associate Professor of Energy Resources Engineering, Stanford University, Stanford, CA, USA.
* **Email:** `sonori@stanford.edu`
* **Expertise:** Nonlinear observer design for electrochemical systems, battery degradation modeling (SEI growth, lithium plating), and experimental validation under dynamic automotive load profiles.
* **Justification:** World leader in bridging electrochemical degradation phenomena with control-oriented state estimation; ideally suited to critique our EIS-time-domain degradation bridge.

#### Reviewer 3: Prof. Scott J. Moura
* **Title & Affiliation:** Professor of Civil & Environmental Engineering and Mechanical Engineering, University of California, Berkeley, USA.
* **Email:** `smoura@berkeley.edu`
* **Expertise:** PDE electrochemical battery observers, parameter co-estimation, battery thermal-electrochemical coupling, and optimal BMS control.
* **Justification:** Expert in both advanced control theory and electrochemical battery dynamics; perfectly qualified to review our Lyapunov stability proofs and distributed series pack benchmark.

#### Reviewer 4: Prof. Weihan Li
* **Title & Affiliation:** Junior Research Group Leader, Center for Ageing, Reliability and Lifetime Prediction of Electrochemical and Electronic Systems (CARL), RWTH Aachen University, Aachen, Germany.
* **Email:** `weihan.li@isea.rwth-aachen.de`
* **Expertise:** Physics-informed machine learning for battery health prognosis, digital twins, impedance spectroscopy analysis, and fast charging optimization.
* **Justification:** Leading European researcher in PINN applications for batteries; ideally suited to rigorously scrutinize the PINN-SOC and hybrid PIDKF architectural benchmarks.

#### Reviewer 5: Prof. Xiaosong Hu
* **Title & Affiliation:** Professor of Automotive Engineering, Chongqing University, Chongqing, China; Fellow of IEEE.
* **Email:** `xiaosonghu@cqu.edu.cn`
* **Expertise:** Battery state and parameter estimation algorithms, fractional-order modeling, multi-temperature state estimation, and embedded BMS hardware realization.
* **Justification:** Expert in comparative evaluation of state estimators across temperatures, sensor corruptions, and automotive drive cycles.

---

### 6. OPPOSED REVIEWERS
* **Opposed Reviewers:** *None.* (The authors welcome any unbiased, qualified reviewer assigned by the Editor-in-Chief).

---

### 7. COMPREHENSIVE COMPLIANCE CHECKLIST

- [x] **Page Budget:** Exactly 13 pages (Standard ASME double-column layout).
- [x] **Typesetting Quality:** 0 Overfull `\hbox`, 0 Underfull `\hbox`, 0 Undefined references, 0 Warnings.
- [x] **Flushend Balanced:** Balanced columns on Page 13 with no vertical overhang.
- [x] **Peer Review Enhancements:** All 8 Major Comments and Minor Comments comprehensively incorporated:
  - Direct Tier 1 PIDKF cycler testing (CALCE & NASA PCoE).
  - Dynamic lumped electro-thermal balance equation ($C_{\text{th}}\frac{dT}{dt} = \dot{Q}_{\text{gen}} - \dot{Q}_{\text{loss}}$).
  - Degrading plant parameter formulas ($R_2$, $C_1$, $C_2$) and empirical EIS calibration basis.
  - No-target-leakage mathematical proof and physics-informed constraints in PIDKF.
  - Explicit uniform non-resonance gap ($\delta_{\text{res}} \ge 0.0042 > 0$) for LFP observability under PE.
  - Concrete empirical comparison with UKF ($+4650\%$ ASF, $5.2\times$ latency penalty).
  - Cell sample diversity and ground truth uncertainty budget ($\pm 0.036\%$).
  - Module/Pack-level thermal gradient physical justification and sensing disparity.
  - Dedicated Limitations and Future Work paragraph.
