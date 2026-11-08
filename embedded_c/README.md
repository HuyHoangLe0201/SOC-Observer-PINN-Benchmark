# Bare-Metal C Implementations for Embedded Battery Management Systems (BMS)

This directory provides production-grade, bare-metal C implementations of state-of-charge estimation algorithms profiled on ARM Cortex-M microcontrollers:

- **EKF** (`ekf.h`, `ekf.c`): Standard 3-state Extended Kalman Filter with stabilized Joseph covariance update.
- **DEKF** (`dekf.h`, `dekf.c`): Dual Extended Kalman Filter for joint SOC and SOH parameter tracking ($R_0, Q_{\text{act}}$).
- **PIDKF** (`pidkf.h`, `pidkf.c`): Proposed Physics-Informed Dual Kalman Filter with Butler–Volmer passivity-constrained neural overpotential compensation.

---

## 1. Embedded Profiling Setup

- **Target Hardware:** STM32F407VG Discovery Kit (ARM Cortex-M4F @ 168 MHz with Hardware Single-Precision FPU).
- **Compiler:** ARM GNU Toolchain (`arm-none-eabi-gcc` v12.3.rel1).
- **Optimization:** `-O3 -flto -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard`.
- **Cycle Measurement:** Hardware Data Watchpoint and Trace (DWT) cycle counter register (`DWT->CYCCNT`).

---

## 2. Compilation and Execution

To compile with the ARM GCC cross-compiler:

```bash
arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard ekf.c -o ekf.o
arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard dekf.c -o dekf.o
arm-none-eabi-gcc -c -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard pidkf.c -o pidkf.o
```

---

## 3. Benchmarked MCU Resource Metrics (from Table 8)

| Algorithm | FLOPs/Step | Cortex-M4 DWT Cycles | RAM (Bytes) | Flash (Bytes) | Measured M4 Latency (168 MHz) | Projected M7 Latency (480 MHz) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EKF** | 420 | 650 | 280 | 3,400 | $3.87\;\upmu\mathrm{s}$ | $1.35\;\upmu\mathrm{s}$ |
| **AEKF** | 580 | 880 | 340 | 4,200 | $5.24\;\upmu\mathrm{s}$ | $1.83\;\upmu\mathrm{s}$ |
| **UKF** | 1,350 | 2,087 | 780 | 6,200 | $12.42\;\upmu\mathrm{s}$ | $4.35\;\upmu\mathrm{s}$ |
| **DEKF** | 980 | 1,450 | 620 | 5,800 | $8.63\;\upmu\mathrm{s}$ | $3.02\;\upmu\mathrm{s}$ |
| **SMO** | 180 | 281 | 140 | 2,100 | $1.67\;\upmu\mathrm{s}$ | $0.58\;\upmu\mathrm{s}$ |
| **PINN-SOC** | 12,800 | 21,501 | 1,200 | 16,500 | $127.98\;\upmu\mathrm{s}$ | $44.79\;\upmu\mathrm{s}$ |
| **PIDKF (Proposed)** | **1,540** | **2,386** | **850** | **8,200** | **14.20** $\upmu\mathbf{s}$ | **4.97** $\upmu\mathbf{s}$ |
