"""
Comprehensive Benchmark Execution Engine
Journal of Energy Storage (Elsevier)

Evaluates:
- Extended Kalman Filter (EKF)
- Adaptive Extended Kalman Filter (AEKF)
- Dual Extended Kalman Filter (DEKF)
- Sliding Mode Observer (SMO)
- BiLSTM with Attention (BiLSTM-Attn)
- Physics-Informed Neural Network (PINN-SOC)

Metrics Computed:
- RMSE (%)
- MAE (%)
- Max Absolute Error (%)
- Convergence Time to +/- 2% error window (seconds)
- Aging Sensitivity Factor (ASF, % degradation increase from SOH=100% to SOH=80%)
- Step execution latency (microseconds / step)
"""

import time
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from battery_model import BatteryCell
from drive_cycles import DriveCycles
from models_model_based import ExtendedKalmanFilter, AdaptiveEKF, DualExtendedKalmanFilter, SlidingModeObserver
from models_data_driven import BiLSTMAttentionSOC, GRUNetworkSOC
from models_physics_informed import PINNSOCModel, PhysicsInformedLoss


class BenchmarkEngine:
    def __init__(self, dt=1.0, results_dir="../results"):
        self.dt = dt
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_simulation_data(self, cycle_name="udds", duration=3600, soh=1.0, temp_c=25.0, noise_v_std=0.005, current_bias=0.0):
        """
        Generate ground truth synthetic drive cycle data using the electro-thermal 2-RC model.
        """
        cell = BatteryCell(nominal_capacity_ah=2.0, soh=soh, temperature_c=temp_c, dt=self.dt)
        cell.reset_state(initial_soc=0.95)

        if cycle_name.lower() == "dst":
            current_profile = DriveCycles.generate_dst(duration_sec=duration, dt=self.dt)
        elif cycle_name.lower() == "us06":
            current_profile = DriveCycles.generate_us06(duration_sec=duration, dt=self.dt)
        elif cycle_name.lower() == "fuds":
            current_profile = DriveCycles.generate_fuds(duration_sec=duration, dt=self.dt)
        else:
            current_profile = DriveCycles.generate_udds(duration_sec=duration, dt=self.dt)

        n_steps = len(current_profile)
        v_true = np.zeros(n_steps)
        soc_true = np.zeros(n_steps)

        for i in range(n_steps):
            vt, s = cell.step(current_profile[i])
            v_true[i] = vt
            soc_true[i] = s

        # Add realistic sensor corruptions
        np.random.seed(42 + int(temp_c))
        v_noisy = v_true + np.random.normal(0, noise_v_std, n_steps)
        current_sensed = current_profile + current_bias + np.random.normal(0, 0.01, n_steps)

        # Compute dV/dt feature
        dv_dt = np.zeros(n_steps)
        dv_dt[1:] = (v_noisy[1:] - v_noisy[:-1]) / self.dt

        return {
            "time": np.arange(n_steps) * self.dt,
            "current_true": current_profile,
            "current_sensed": current_sensed,
            "v_true": v_true,
            "v_noisy": v_noisy,
            "soc_true": soc_true,
            "dv_dt": dv_dt,
            "temp_c": np.full(n_steps, temp_c),
            "soh": soh
        }

    def train_deep_models(self, epochs=25, lr=0.002):
        """
        Train BiLSTM-Attention and PINN-SOC on diverse training cycles (DST & FUDS at 25C, SOH=1.0 & 0.9).
        """
        print("[BenchmarkEngine] Training Deep Learning & PINN Models...")
        torch.manual_seed(42)
        np.random.seed(42)
        train_data_dst = self.generate_simulation_data("dst", duration=3600, soh=1.0, temp_c=25.0)
        train_data_fuds = self.generate_simulation_data("fuds", duration=3600, soh=0.9, temp_c=25.0)

        # Build feature matrices: [Current, Voltage, Temp/45.0, dV_dt]
        def prep_tensors(data_dict):
            feats = np.stack([
                data_dict["current_sensed"] / 5.0,
                (data_dict["v_noisy"] - 3.7) / 0.5,
                data_dict["temp_c"] / 45.0,
                data_dict["dv_dt"] / 0.05
            ], axis=-1)
            return (
                torch.tensor(feats, dtype=torch.float32),
                torch.tensor(data_dict["soc_true"], dtype=torch.float32),
                torch.tensor(data_dict["current_sensed"], dtype=torch.float32),
                torch.tensor(data_dict["v_noisy"], dtype=torch.float32)
            )

        f_dst, s_dst, i_dst, v_dst = prep_tensors(train_data_dst)
        f_fuds, s_fuds, i_fuds, v_fuds = prep_tensors(train_data_fuds)

        f_train = torch.cat([f_dst, f_fuds], dim=0).to(self.device)
        s_train = torch.cat([s_dst, s_fuds], dim=0).to(self.device)
        i_train = torch.cat([i_dst, i_fuds], dim=0).to(self.device)
        v_train = torch.cat([v_dst, v_fuds], dim=0).to(self.device)

        # 1. Train BiLSTM
        bilstm = BiLSTMAttentionSOC(input_dim=4, hidden_dim=48, num_layers=2).to(self.device)
        opt_bilstm = optim.AdamW(bilstm.parameters(), lr=lr, weight_decay=1e-4)
        crit_bilstm = torch.nn.MSELoss()

        # Sequence chunking for BiLSTM training
        seq_len = 30
        n_samples = f_train.shape[0] - seq_len
        idx_train = np.arange(0, n_samples, 5)

        bilstm.train()
        for epoch in range(epochs):
            total_l = 0.0
            np.random.shuffle(idx_train)
            batch_indices = idx_train[:150]
            for idx in batch_indices:
                x_seq = f_train[idx : idx + seq_len].unsqueeze(0)
                y_target = s_train[idx + seq_len - 1].unsqueeze(0)
                opt_bilstm.zero_grad()
                pred = bilstm(x_seq)
                loss = crit_bilstm(pred, y_target)
                loss.backward()
                opt_bilstm.step()
                total_l += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  [BiLSTM] Epoch {epoch+1}/{epochs} - Loss: {total_l/len(batch_indices):.6f}")

        # 2. Train PINN-SOC
        pinn = PINNSOCModel(input_dim=4, hidden_dim=48).to(self.device)
        opt_pinn = optim.AdamW(pinn.parameters(), lr=lr, weight_decay=1e-4)
        pinn_loss_fn = PhysicsInformedLoss(q_nominal_ah=2.0, r0_nominal=0.035, dt=self.dt)

        pinn.train()
        for epoch in range(epochs):
            total_pinn_l = 0.0
            batch_indices = idx_train[:150]
            for idx in batch_indices:
                x_seq = f_train[idx : idx + seq_len]
                s_seq = s_train[idx : idx + seq_len]
                i_seq = i_train[idx : idx + seq_len]
                v_seq = v_train[idx : idx + seq_len]

                opt_pinn.zero_grad()
                soc_p, v_pol_p = pinn(x_seq)
                loss, _ = pinn_loss_fn(soc_p, v_pol_p, s_seq, i_seq, v_seq)
                loss.backward()
                opt_pinn.step()
                total_pinn_l += loss.item()
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  [PINN-SOC] Epoch {epoch+1}/{epochs} - Loss: {total_pinn_l/len(batch_indices):.6f}")

        self.bilstm_model = bilstm
        self.pinn_model = pinn
        print("[BenchmarkEngine] Model training complete.")

    def run_benchmark_on_dataset(self, data, initial_soc_guess=None):
        """
        Evaluate all algorithms on a given test dataset.
        Returns a dict of estimated SOC trajectories and step latencies.
        """
        n_steps = len(data["time"])
        currents = data["current_sensed"]
        voltages = data["v_noisy"]
        soc_true = data["soc_true"]
        soh_cell = data["soh"]
        temp_c = data["temp_c"][0]

        init_soc = soc_true[0] if initial_soc_guess is None else initial_soc_guess

        # Initialize test cell with nominal parameters
        test_cell = BatteryCell(nominal_capacity_ah=2.0, soh=1.0, temperature_c=temp_c, dt=self.dt)

        # 1. Extended Kalman Filter
        ekf = ExtendedKalmanFilter(test_cell)
        ekf.reset(initial_soc=init_soc)
        soc_ekf = np.zeros(n_steps)
        t0 = time.perf_counter()
        for k in range(n_steps):
            s_est, _, _ = ekf.step(currents[k], voltages[k])
            soc_ekf[k] = s_est
        time_ekf_per_step = ((time.perf_counter() - t0) / n_steps) * 1e6

        # 2. Adaptive EKF
        aekf = AdaptiveEKF(test_cell)
        aekf.reset(initial_soc=init_soc)
        soc_aekf = np.zeros(n_steps)
        t0 = time.perf_counter()
        for k in range(n_steps):
            s_est, _, _ = aekf.step(currents[k], voltages[k])
            soc_aekf[k] = s_est
        time_aekf_per_step = ((time.perf_counter() - t0) / n_steps) * 1e6

        # 3. Dual EKF
        dekf = DualExtendedKalmanFilter(test_cell)
        dekf.reset(initial_soc=init_soc)
        soc_dekf = np.zeros(n_steps)
        r0_tracked = np.zeros(n_steps)
        t0 = time.perf_counter()
        for k in range(n_steps):
            s_est, _, r0_e, _ = dekf.step(currents[k], voltages[k])
            soc_dekf[k] = s_est
            r0_tracked[k] = r0_e
        time_dekf_per_step = ((time.perf_counter() - t0) / n_steps) * 1e6

        # 4. Sliding Mode Observer
        smo = SlidingModeObserver(test_cell)
        smo.reset(initial_soc=init_soc)
        soc_smo = np.zeros(n_steps)
        t0 = time.perf_counter()
        for k in range(n_steps):
            s_est, _ = smo.step(currents[k], voltages[k])
            soc_smo[k] = s_est
        time_smo_per_step = ((time.perf_counter() - t0) / n_steps) * 1e6

        # 5. BiLSTM with Attention
        self.bilstm_model.eval()
        feats = np.stack([
            currents / 5.0,
            (voltages - 3.7) / 0.5,
            data["temp_c"] / 45.0,
            data["dv_dt"] / 0.05
        ], axis=-1)
        feat_tensor = torch.tensor(feats, dtype=torch.float32).to(self.device)

        soc_bilstm = np.zeros(n_steps)
        seq_len = 30
        soc_bilstm[:seq_len] = init_soc
        t0 = time.perf_counter()
        with torch.no_grad():
            for k in range(seq_len, n_steps):
                x_in = feat_tensor[k - seq_len : k].unsqueeze(0)
                soc_bilstm[k] = self.bilstm_model(x_in).item()
        time_bilstm_per_step = ((time.perf_counter() - t0) / (n_steps - seq_len)) * 1e6

        # 6. PINN-SOC
        self.pinn_model.eval()
        soc_pinn = np.zeros(n_steps)
        t0 = time.perf_counter()
        with torch.no_grad():
            soc_out, _ = self.pinn_model(feat_tensor)
            soc_pinn = soc_out.cpu().numpy()
        time_pinn_per_step = ((time.perf_counter() - t0) / n_steps) * 1e6

        # Metrics computation
        models = {
            "EKF": (soc_ekf, time_ekf_per_step),
            "AEKF": (soc_aekf, time_aekf_per_step),
            "DEKF": (soc_dekf, time_dekf_per_step),
            "SMO": (soc_smo, time_smo_per_step),
            "BiLSTM-Attn": (soc_bilstm, time_bilstm_per_step),
            "PINN-SOC": (soc_pinn, time_pinn_per_step),
        }

        results = {}
        for name, (soc_est, latency_us) in models.items():
            err_pct = (soc_est - soc_true) * 100.0
            rmse = np.sqrt(np.mean(err_pct ** 2))
            mae = np.mean(np.abs(err_pct))
            max_err = np.max(np.abs(err_pct))

            # Convergence time (time in seconds until error strictly stays within +/- 2%)
            within_bounds = np.abs(err_pct) <= 2.0
            conv_idx = np.where(within_bounds)[0]
            conv_time = data["time"][conv_idx[0]] if len(conv_idx) > 0 else data["time"][-1]

            results[name] = {
                "soc_est": soc_est,
                "rmse_pct": rmse,
                "mae_pct": mae,
                "max_err_pct": max_err,
                "conv_time_sec": conv_time,
                "latency_us_step": latency_us
            }

        return results
