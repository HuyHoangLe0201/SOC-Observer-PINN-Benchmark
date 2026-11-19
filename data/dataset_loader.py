"""
Battery Dataset Compatibility & Sample Trajectory Loader
Journal of Energy Storage (Elsevier)

Note on Dataset Provenance:
- Primary Tier 1 public cycler experiments in this benchmark run directly on on-disk
  XJTU Batch-5 (NMC532 Random Walk) and NASA Ames PCoE (B0005/6/7/30/55/56) files.
- This module provides generic Arbin CSV/Excel column parsing utilities and generates a
  synthetic CALCE-format compatibility CSV (`calce_sample_udds_25c.csv`) from the calibrated
  BatteryCellNMC Tier 2 plant for standalone quickstart script testing (NOT claimed as Tier 1 data).
"""

import os
import urllib.request
import numpy as np
import pandas as pd


class BatteryDatasetLoader:
    def __init__(self, data_root=None):
        if data_root is None:
            self.data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        else:
            self.data_root = os.path.abspath(data_root)
        os.makedirs(self.data_root, exist_ok=True)

    def load_calce_format(self, filepath):
        """
        Parses standardized CALCE Arbin cycler CSV / Excel export:
        Expected columns:
        - Date_Time or Test_Time(s)
        - Current(A)
        - Voltage(V)
        - Temperature(C)
        - Discharge_Capacity(Ah) / Charge_Capacity(Ah)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported format. Please supply .csv or .xlsx file.")

        # Standardize column naming
        col_map = {}
        for col in df.columns:
            clean_c = col.lower().replace(" ", "").replace("_", "")
            if "time" in clean_c and "s" in clean_c:
                col_map[col] = "time_s"
            elif "current" in clean_c and "a" in clean_c:
                col_map[col] = "current_a"
            elif "voltage" in clean_c and "v" in clean_c:
                col_map[col] = "voltage_v"
            elif "temp" in clean_c:
                col_map[col] = "temp_c"

        df_std = df.rename(columns=col_map)
        return df_std

    def export_sample_benchmark_csv(self, filename="calce_sample_udds_25c.csv"):
        """
        Generates a validated benchmark test CSV in CALCE-compatible format
        for instant laboratory reproduction by other researchers.
        """
        from drive_cycles import DriveCycles
        from battery_model import BatteryCellNMC

        target_path = os.path.join(self.data_root, filename)
        cell = BatteryCellNMC(nominal_capacity_ah=2.0, soh=1.0, temperature_c=25.0, dt=1.0)
        current = DriveCycles.generate_udds(duration_sec=3600, dt=1.0)

        n = len(current)
        v_out = np.zeros(n)
        soc_out = np.zeros(n)
        for i in range(n):
            vt, s = cell.step(current[i])
            v_out[i] = vt
            soc_out[i] = s

        df_sample = pd.DataFrame({
            "Test_Time(s)": np.arange(n),
            "Current(A)": np.round(current, 4),
            "Voltage(V)": np.round(v_out + np.random.normal(0, 0.004, n), 4),
            "Temperature(C)": 25.0,
            "True_SOC": np.round(soc_out, 5)
        })

        df_sample.to_csv(target_path, index=False)
        print(f"[DatasetLoader] Saved reference test cycle to: {target_path}")
        return target_path


if __name__ == "__main__":
    loader = BatteryDatasetLoader()
    loader.export_sample_benchmark_csv()
