# Bare-Metal ANSI-C99 Reference Implementations for Embedded BMS

This directory provides static-memory, dynamic-allocation-free ANSI-C99 reference kernels for state-of-charge estimation on 32-bit FPU microcontrollers (ARMv7E-M Cortex-M4F / Cortex-M7):

- **EKF** (`ekf.h`, `ekf.c`): 3-state Extended Kalman Filter with stabilized Joseph covariance update.
- **DEKF** (`dekf.h`, `dekf.c`): Dual EKF with Plett (2004) recurrent total-derivative sensitivity (`dx_dtheta`), immutable nominal capacity anchor `Q_init`, and rest-period innovation gating (`|I| < 10 mA`).
- **PIDKF** (`pidkf.h`, `pidkf.c`, `pidkf_weights.h`): Hybrid Physics-Informed Dual Kalman Filter coupling the DEKF core with the exported Seed-42 `[6 -> 16 -> 8 -> 1]` GELU residual compensator (257 `float32` parameters = 1,028 B) under Butler-Volmer passivity projection (`kappa_0 = 15 mV`, `kappa_1 = 22 mOhm`, `I_max = 6.0 A`, `I_k * delta_V <= 0`).

---

## 1. Analytical Resource & Instruction-Level Profiling Basis

- **Target ISA:** ARMv7E-M (STM32F407VG Cortex-M4F @ 168 MHz with FPv4-SP-D16 single-precision hardware FPU; projected Cortex-M7 @ 480 MHz via 1.45x dual-issue IPC scaling).
- **Methodology:** Static RAM and Flash figures are computed directly from C `struct` field sizes (`sizeof`), stack local frames, and exact `float32` weight arrays (`pidkf_weights.h`, PINN-SOC 6,690 params, BiLSTM-Attn 13,698 params). Arithmetic FLOPs/step and ARMv7E-M cycle estimates count exact FPv4-SP-D16 MAC (`VMLA`), multiply/add, hardware `VDIV`/`VSQRT`, and `expf`/`tanhf` polynomial calls (no unverified hardware `DWT->CYCCNT` timing claims are asserted).

---

## 2. Reproducing Exported Weights & Analytical Profiling

Run from the repository root:

```bash
python -u src/export_c_weights.py
```

This script trains the Seed-42 NMC compensator, exports `embedded_c/pidkf_weights.h`, verifies Python-vs-C `float32` numerical parity (`< 1e-5 V`), and outputs `results/table8_embedded_profiling.csv`.
