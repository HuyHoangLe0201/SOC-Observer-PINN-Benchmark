#!/usr/bin/env python3
"""
Master 1-Click Reproducibility & CSV Summary Runner
Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks
Across Battery Aging and Thermal Regimes (Journal of Energy Storage)

Authors: Huy Hoang Le, Kim-Anh Nguyen
GitHub: https://github.com/HuyHoangLe0201/SOC-Observer-PINN-Benchmark
"""

import os
import time
import pandas as pd


def print_banner():
    print("=" * 92)
    print("   COMPREHENSIVE BENCHMARK OF STATE-OF-CHARGE OBSERVERS & PHYSICS-INFORMED NETWORKS")
    print("                  Across Battery Aging and Thermal Regimes (J. Energy Storage)")
    print("=" * 92)


def verify_reproducible_assets(ws):
    results_dir = os.path.join(ws, "results")

    expected_figs = [
        ("fig1", (".pdf", ".png")),
        ("fig2", (".pdf", ".png")),
        ("fig3", (".pdf", ".png")),
        ("fig4", (".pdf", ".png")),
        ("fig5", (".pdf", ".png")),
        ("fig6", (".pdf", ".png")),
        ("fig7", (".pdf", ".png")),
        ("fig8", (".pdf", ".png")),
        ("fig9", (".pdf", ".png")),
        ("fig10", (".pdf", ".png")),
        ("fig11", (".pdf", ".png")),
        ("fig12", (".pdf", ".png")),
        ("fig13", (".pdf", ".png")),
    ]

    expected_csvs = [
        "table1_dynamic_udds.csv",
        "table2_aging_degradation.csv",
        "table3_thermal_robustness.csv",
        "table4_convergence.csv",
        "table5_radar_scores.csv",
        "table6_cross_chemistry.csv",
        "table7_sensor_corruption.csv",
        "table7_pack_cell_inconsistency.csv",
        "table8_embedded_profiling.csv",
        "table9_pidkf_ablation.csv",
        "eis_nasa_b0006_fit.csv",
        "tier1_xjtu_random_walk.csv",
        "tier1_nasa_thermal_aging.csv",
    ]

    print("[1/3] Verifying Generated Benchmark Figures (13 PDF + 13 PNG in results/)...")
    all_ok = True
    for base, exts in expected_figs:
        missing = []
        for ext in exts:
            if not os.path.exists(os.path.join(results_dir, base + ext)):
                missing.append(f"results/{base}{ext}")
        png_kb = os.path.getsize(os.path.join(results_dir, base + ".png")) / 1024.0 if os.path.exists(os.path.join(results_dir, base + ".png")) else 0.0
        if not missing:
            print(f"  [OK] {base:34s} ({png_kb:6.1f} KB PNG) - Verified (PDF+PNG)")
        else:
            print(f"  [MISSING] {base}: {', '.join(missing)}")
            all_ok = False

    print("\n[2/3] Verifying Quantitative Benchmark CSVs in results/...")
    for csv_name in expected_csvs:
        p = os.path.join(results_dir, csv_name)
        if os.path.exists(p):
            print(f"  [OK] {csv_name}")
        else:
            print(f"  [MISSING] {csv_name}")
            all_ok = False

    return all_ok


def summarize_from_csvs(ws):
    results_dir = os.path.join(ws, "results")
    sections = [
        ("PHASE 1: Nominal UDDS Benchmark (Tier 2 Co-Sim, 25C, 100% SOH)", "table1_dynamic_udds.csv"),
        ("PHASE 2: Progressive Aging Degradation (Tier 2 Co-Sim, 100% -> 80% SOH)", "table2_aging_degradation.csv"),
        ("PHASE 3: Thermal Robustness (Tier 2 Co-Sim, 0C / 25C / 45C)", "table3_thermal_robustness.csv"),
        ("PHASE 10A: Analytical ARMv7E-M Sizing & Static ANSI-C99 Footprint", "table8_embedded_profiling.csv"),
        ("PHASE 10B: Tier 2 Five-Seed PIDKF Ablation (85% SOH NMC & LFP)", "table9_pidkf_ablation.csv"),
        ("TIER 1A: XJTU Batch-5 Held-Out NMC532 Dynamic Random Walk", "tier1_xjtu_random_walk.csv"),
        ("TIER 1B: NASA Ames PCoE Aging & Thermal Extremes", "tier1_nasa_thermal_aging.csv"),
    ]
    print("\n[3/3] Dynamic CSV Benchmark Summary (Single Source of Truth):")
    for title, fname in sections:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            print(f"\n--- {title} ---")
            df = pd.read_csv(path)
            print(df.to_string(index=False))


def main():
    ws = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    print_banner()
    ok = verify_reproducible_assets(ws)
    summarize_from_csvs(ws)
    elapsed = time.time() - t0
    print(f"\nVerification finished in {elapsed:.2f} s | Status: {'PASS' if ok else 'INCOMPLETE'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
