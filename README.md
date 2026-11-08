# Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14028912.svg)](https://doi.org/10.5281/zenodo.14028912)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Embedded: ARM Cortex-M](https://img.shields.io/badge/Hardware-ARM%20Cortex--M4%2FM7-0091BD.svg)](https://developer.arm.com/Processors/Cortex-M4)

---

## 📌 Overview

This repository provides the complete, self-contained, reproducible codebase and experimental benchmarking platform for the research article:

> **"Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes"**  
> *Huy Hoang Le, Kim-Anh Nguyen*  
> Submitted to **Journal of Energy Storage** (Elsevier).

While conventional battery State-of-Charge (SOC) estimation studies predominantly benchmark fresh cells at nominal room temperature ($25^\circ\text{C}$), this work establishes a rigorous, dual-tier experimental and calibrated co-simulation framework across **10 operational phases**:

1. **Nominal Driving Schedules** (UDDS, US06, FUDS, DST at $25^\circ\text{C}$, $100\%$ SOH)
2. **Progressive Aging Degradation** ($100\% \to 80\%$ SOH capacity fade and impedance growth)
3. **Wide Thermal Regimes** ($0^\circ\text{C}$, $25^\circ\text{C}$, $45^\circ\text{C}$)
4. **Initial Condition Uncertainty & Error Convergence** ($-40\%$ initial SOC shock)
5. **Multi-Attribute Radar & Pareto Trade-Off Analysis**
6. **Cross-Chemistry Disparities** (Sloping-OCV NMC vs. Flat-Plateau $\text{LiFePO}_4$ with dynamic hysteresis)
7. **Sensor Corruption & Environmental Noise** ($+0.25\text{ A}$ Hall bias drift and $25\text{ mV}$ inverter EMI)
8. **Model-Based Electrochemical Impedance Spectroscopy (EIS)** across progressive degradation stages
9. **Module & Pack-Level Scaling** (8S series pack with cell-to-cell parameter and thermal inconsistency)
10. **Bare-Metal Profiling & Hybrid PIDKF Formulation** (ARM Cortex-M bare-metal C deployment)

---

## 🔬 Key Scientific Breakthroughs

- **The 45.7-Fold Aging Escalation:** Uncompensated Extended Kalman Filters (EKF) suffer a **45.7-fold RMSE surge** ($0.065\%$ to $2.973\%$, $\text{ASF} = +4473.8\%$) as battery capacity fades to $80\%$ SOH because interfacial impedance expansion ($I \cdot \Delta R_0$) is erroneously misattributed to thermodynamic open-circuit voltage (OCV) shifts. Adaptive covariance filters (AEKF) fail to prevent this degradation ($5.406\%$ RMSE).
- **Online Parameter Adaptation (DEKF):** Online dual-tracking of ohmic resistance and active capacity confines full-lifespan RMSE to $0.585\%$.
- **Hysteresis Restores LFP Observability:** Dynamic hysteresis tracking on flat-plateau $\text{LiFePO}_4$ chemistry restores linear time-varying observability ($\lambda_{\min}(\mathcal{G}_o) \ge 1.21 \times 10^{-3}$), preventing catastrophic divergence.
- **Physics-Informed Dual Kalman Filter (PIDKF):** We formulate a hybrid filter coupling Butler–Volmer/Tafel passivity bounds with an online DEKF core. Experimental validation on degraded cells ($85\%$ SOH) achieves **$0.428\%$ RMSE (NMC)** and **$0.158\%$ RMSE (LFP)** on CALCE RPT profiles, and bounds error to $<0.85\%$ under NASA PCoE thermal extremes ($4^\circ\text{C}$ and $43^\circ\text{C}$).
- **Bare-Metal MCU Profiling:** Profiling on ARM Cortex-M4 ($168\text{ MHz}$) using ARM GCC 12.3 and CMSIS-DSP demonstrates that PIDKF executes in **$14.20\;\upmu\mathrm{s}$**, achieving the optimal Pareto efficiency frontier for automotive and stationary BMS.

---

## 📂 Repository Structure

```
.
├── README.md                      # Project documentation and reproducibility guide
├── LICENSE                        # MIT Open-Source License
├── requirements.txt               # Python package dependencies
├── environment.yml                # Conda environment definition
├── run_all_benchmarks.py          # Master 1-click verification runner
├── Cover_Letter_Journal_of_Energy_Storage.docx # Word Cover Letter
│
├── src/                           # Modular Python benchmark source code
│   ├── battery_model.py           # 2-RC ECM with Arrhenius & degradation physics
│   ├── drive_cycles.py            # Standardized drive profiles (UDDS, US06, FUDS, DST)
│   ├── models_model_based.py      # EKF, AEKF (Sage-Husa), DEKF, SMO
│   ├── models_data_driven.py      # BiLSTM with temporal self-attention
│   ├── models_physics_informed.py # Physics-Informed Neural Network (PINN-SOC)
│   ├── pidkf_hybrid.py            # Proposed hybrid PIDKF algorithm
│   ├── pack_model.py              # 8S series pack with manufacturing & thermal gradients
│   ├── eis_analyzer.py            # Randles-CPE Nyquist spectrum analyzer
│   ├── benchmark_engine.py        # Automated multi-phase evaluation engine
│   ├── nasa_data.py               # NASA PCoE battery dataset loader
│   └── visualizer.py              # Publication-grade figure generator
│
├── embedded_c/                    # Bare-metal C implementations for ARM Cortex-M
│   ├── README.md                  # Compilation & DWT profiling instructions
│   ├── ekf.h / ekf.c              # 3-state EKF with stabilized Joseph form
│   ├── dekf.h / dekf.c            # Joint state-parameter DEKF
│   └── pidkf.h / pidkf.c          # Proposed hybrid PIDKF with GELU neural compensator
│
├── data/                          # Dataset loaders & sample drive trajectories
│   ├── dataset_loader.py          # Cycler parser for CALCE and Oxford datasets
│   └── calce_sample_udds_25c.csv  # 1-second sample cycler baseline
│
├── results/                       # High-resolution publication figures & benchmark tables
│   ├── fig1_udds_tracking_errors.png
│   ├── fig2_aging_degradation.png
│   ├── fig3_thermal_robustness.png
│   ├── fig4_initial_error_convergence.png
│   ├── fig5_radar_tradeoff.png
│   ├── fig6_lfp_plateau_tracking.png
│   ├── fig7_sensor_bias_drift_stress.png
│   ├── fig8_eis_nyquist_aging.png
│   ├── fig9_pack_cell_inconsistency.png
│   ├── fig10_embedded_resource_tradeoff.png
│   └── fig11_pidkf_hybrid_benchmark.png
│
├── manuscript/                    # Complete, unformatted LaTeX manuscript (article.cls)
│   ├── manuscript_unformatted.tex # Standard unformatted LaTeX source
│   ├── references.bib             # Complete BibTeX bibliography database
│   ├── manuscript_unformatted.pdf # 31-page compiled PDF (0 warnings, 0 errors)
│   └── fig1_*.png ... fig11_*.png # Co-located figure files
│
└── submission_journal_of_energy_storage/ # Submission-ready pack for Journal of Energy Storage
    ├── Cover_Letter_Journal_of_Energy_Storage.docx
    ├── HIGHLIGHTS_JOURNAL_OF_ENERGY_STORAGE.md
    ├── manuscript.pdf
    ├── manuscript.tex
    ├── references.bib
    └── fig1_*.png ... fig11_*.png # Co-located figure files
```

---

## ⚡ Quickstart & Installation

### Option 1: Using Conda (Recommended)

```bash
git clone https://github.com/HuyHoangLe0201/SOC-Observer-PINN-Benchmark.git
cd SOC-Observer-PINN-Benchmark

conda env create -f environment.yml
conda activate soc-benchmark
```

### Option 2: Using Standard Virtualenv & Pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 1-Click Reproducibility

To verify the benchmark suite and inspect the reproduced figures and metrics:

```bash
python run_all_benchmarks.py
```

To run individual experimental modules:

```bash
# Run advanced benchmarks across chemistries and thermal extremes:
python src/run_advanced_benchmarks.py

# Run hybrid PIDKF evaluation on degraded cells:
python src/pidkf_hybrid.py

# Run pack-level 8S series simulation:
python src/pack_model.py

# Run EIS Randles-CPE Nyquist spectrum generation:
python src/eis_analyzer.py
```

---

## 📊 Summary Benchmark Results

### Table 1: Nominal UDDS Benchmark ($25^\circ\text{C}$, $100\%$ SOH, Samsung INR18650-20R)

| Algorithm | Type | RMSE (%) | MAE (%) | MaxE (%) | Cortex-M4 Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EKF** | Model-based (Fixed) | 0.065 | 0.050 | 1.067 | $3.87\;\upmu\mathrm{s}$ |
| **AEKF** | Model-based (Adaptive noise) | 0.759 | 0.610 | 1.490 | $5.24\;\upmu\mathrm{s}$ |
| **DEKF** | Model-based (Dual tracking) | 0.191 | 0.159 | 0.673 | $8.63\;\upmu\mathrm{s}$ |
| **SMO** | Nonlinear Sliding Mode | 0.941 | 0.755 | 3.423 | **$1.67\;\upmu\mathrm{s}$** |
| **BiLSTM-Attn** | Deep Sequence Network | 2.813 | 2.309 | 6.915 | $2440.48\;\upmu\mathrm{s}$ |
| **PINN-SOC** | Physics-Informed ML | 2.781 | 2.278 | 7.317 | $127.98\;\upmu\mathrm{s}$ |
| **PIDKF (Proposed)** | Hybrid Physics-Informed Dual | **0.062** | **0.048** | **0.612** | **$14.20\;\upmu\mathrm{s}$** |

### Table 2: Lifespan Capacity Fade ($100\% \to 80\%$ SOH)

| Algorithm | SOH 100% | SOH 95% | SOH 90% | SOH 85% | SOH 80% | Aging Sensitivity Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EKF** | **0.065%** | 0.553% | 1.189% | 1.988% | 2.973% | **+4473.8% (45.7x surge)** |
| **AEKF** | 0.759% | 0.523% | 1.675% | 3.834% | 5.406% | +612.3% |
| **DEKF** | 0.191% | **0.105%** | **0.233%** | **0.407%** | **0.585%** | **+206.3% (Bounded)** |
| **SMO** | 0.941% | 1.745% | 3.002% | 4.254% | 5.428% | +476.8% |
| **BiLSTM-Attn** | 2.813% | 2.917% | 3.405% | 4.202% | 5.128% | +82.3% |
| **PINN-SOC** | 2.781% | 3.081% | 3.551% | 4.177% | 4.911% | +76.6% |

---

## 🛠️ Bare-Metal C Compilation

To compile and benchmark the bare-metal C kernels for ARM Cortex-M4:

```bash
cd embedded_c
arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard ekf.c -o ekf.o
arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard dekf.c -o dekf.o
arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard pidkf.c -o pidkf.o
```

---

## 📄 Citation

If you use this benchmark platform, models, or bare-metal C code in your research, please cite:

```bibtex
@article{Le2026SOCBenchmark,
  author  = {Le, Huy Hoang and Nguyen, Kim-Anh},
  title   = {Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes},
  journal = {Journal of Energy Storage},
  year    = {2026},
  doi     = {10.5281/zenodo.14028912},
  note    = {Under Review}
}
```

---

## ⚖️ License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## ✉️ Contact

- **Huy Hoang Le** - R&D Department, PowerMore Ltd., Vietnam (Email: `hoang.le@powermore.vn`)
- **Kim-Anh Nguyen, Ph.D.** *(Corresponding Author)* - Faculty of Electrical Engineering, University of Science and Technology, The University of Danang, Vietnam (Email: `nkanh@dut.udn.vn`)
