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
    manuscript_dir = os.path.join(ws, "manuscript")
    sub_dir = os.path.join(ws, "submission_journal_of_energy_storage")

    expected_figs = [
        ("fig0_electrochemical_mechanism", (".pdf", ".png")),
        ("fig0_pidkf_framework", (".pdf", ".png")),
        ("fig1_udds_tracking_errors", (".pdf", ".png")),
        ("fig2_aging_degradation", (".pdf", ".png")),
        ("fig3_thermal_robustness", (".pdf", ".png")),
        ("fig4_initial_error_convergence", (".pdf", ".png")),
        ("fig5_radar_tradeoff", (".pdf", ".png")),
        ("fig6_lfp_plateau_tracking", (".pdf", ".png")),
        ("fig7_sensor_bias_drift_stress", (".pdf", ".png")),
        ("fig8_eis_nyquist_aging", (".pdf", ".png")),
        ("fig9_pack_cell_inconsistency", (".pdf", ".png")),
        ("fig10_embedded_resource_tradeoff", (".pdf", ".png")),
        ("fig11_pidkf_hybrid_benchmark", (".pdf", ".png")),
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

    print("[1/3] Verifying Generated Benchmark Figures (13 PDF + 13 PNG across all folders)...")
    all_ok = True
    for base, exts in expected_figs:
        missing = []
        for folder in (results_dir, manuscript_dir, sub_dir):
            for ext in exts:
                if not os.path.exists(os.path.join(folder, base + ext)):
                    missing.append(f"{os.path.basename(folder)}/{base}{ext}")
        png_kb = os.path.getsize(os.path.join(results_dir, base + ".png")) / 1024.0 if os.path.exists(os.path.join(results_dir, base + ".png")) else 0.0
        if not missing:
            print(f"  [OK] {base:34s} ({png_kb:6.1f} KB PNG) - Synced (PDF+PNG)")
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
    print("Submitted PDF: submission_journal_of_energy_storage/manuscript.pdf")
    print("Cover Letter:  Cover_Letter_Journal_of_Energy_Storage.docx")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
