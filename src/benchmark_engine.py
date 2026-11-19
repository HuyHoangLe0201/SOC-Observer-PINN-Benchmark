"""
Unified Benchmark Engine (Tier 2 Calibrated Co-Simulation)
Journal of Energy Storage (Elsevier)

Strict fairness & zero-leakage guarantees:
1. All model-based observers (EKF, AEKF, UKF, DEKF, SMO, PIDKF) start from IDENTICAL
   fresh, room-temperature nominal parameters (100% SOH, 25 degC) unless explicitly stated.
2. Neural baselines (BiLSTM-Attn, PINN-SOC, Unconstrained MLP) are trained strictly on
   Batch A DST profiles (across 100%, 90%, 80% SOH at 25 degC) and evaluated on strictly
   held-out UDDS, FUDS, and US06 schedules.
3. Exact convergence metrics strictly distinguish t_reach^(2%), t_settle^(5%), global MaxE,
   and post-entry MaxE_dyn (for t > t_reach^(2%)).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from battery_model import BatteryCellNMC, BatteryCellLFP
from drive_cycles import DriveCycles
from models_model_based import (
    ExtendedKalmanFilter,
    AdaptiveEKF,
    UnscentedKalmanFilter,
    DualExtendedKalmanFilter,
    SlidingModeObserver,
)
from models_data_driven import BiLSTMAttentionSOC
from models_physics_informed import PINNSOCModel, PhysicsInformedLoss


def compute_metrics(soc_est, soc_true, initial_offset_mode=False):
    """
    Compute RMSE, MAE, global MaxE, and exact Phase-4 convergence indicators:
      - t_reach^(2%): earliest t >= 0 with |e_t| <= 2.0%
      - t_settle^(5%): earliest t >= 0 after which |e_tau| <= 5.0% for ALL tau >= t
      - max_dyn_pct: max_{|t > t_reach|} |e_t| (post-entry dynamic peak error)
    """
    err = (soc_est - soc_true) * 100.0
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))
    max_all = float(np.max(abs_err))

    # First-entry reach into <= 2.0% corridor
    idx_2 = np.where(abs_err <= 2.0)[0]
    t_reach_2 = float(idx_2[0]) if len(idx_2) > 0 else np.nan

    # Permanent settling time into <= 5.0% corridor: suffix maximum <= 5.0%
    # Compute reverse cumulative max of abs_err
    rev_max = np.maximum.accumulate(abs_err[::-1])[::-1]
    idx_5 = np.where(rev_max <= 5.0)[0]
    t_settle_5 = float(idx_5[0]) if len(idx_5) > 0 else np.nan

    # Post-entry dynamic peak error strictly for t > t_reach^(2%)
    if not np.isnan(t_reach_2) and int(t_reach_2) + 1 < len(abs_err):
        max_dyn = float(np.max(abs_err[int(t_reach_2) + 1 :]))
    else:
        max_dyn = max_all

    return {
        "rmse_pct": rmse,
        "mae_pct": mae,
        "max_err_pct": max_all,
        "t_reach_2": t_reach_2,
        "t_settle_5": t_settle_5,
        "max_dyn_pct": max_dyn,
        "final_err_pct": float(abs_err[-1]),
    }


class BenchmarkEngine:
    """
    Clean Tier 2 co-simulation orchestrator for Phases 1-4 and 6-7 baselines.
    """

    def __init__(self, dt=1.0, results_dir="results"):
        self.dt = float(dt)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.device = torch.device("cpu")

        self.bilstm_models = []
        self.pinn_models = []
        self.mlp_models = []

    def generate_simulation_data(
        self,
        cycle_name="udds",
        duration=3600,
        soh=1.0,
        temp_c=25.0,
        chemistry="NMC",
        initial_soc=0.95,
        i_bias=0.0,
        v_noise_std=0.005,
        i_noise_std=0.010,
        seed=42,
    ):
        if chemistry == "LFP":
            cell = BatteryCellLFP(nominal_capacity_ah=2.30, soh=soh, temperature_c=temp_c, dt=self.dt)
        else:
            cell = BatteryCellNMC(nominal_capacity_ah=2.00, soh=soh, temperature_c=temp_c, dt=self.dt)
        cell.reset_state(initial_soc=initial_soc)

        cname = cycle_name.lower()
        if cname == "udds":
            current = DriveCycles.generate_udds(duration_sec=duration, dt=self.dt)
        elif cname == "dst":
            current = DriveCycles.generate_dst(duration_sec=duration, dt=self.dt)
        elif cname == "fuds":
            current = DriveCycles.generate_fuds(duration_sec=duration, dt=self.dt)
        elif cname == "us06":
            current = DriveCycles.generate_us06(duration_sec=duration, dt=self.dt)
        else:
            raise ValueError(f"Unknown cycle {cycle_name}")

        n = len(current)
        v_true = np.zeros(n)
        soc_true = np.zeros(n)
        temp_arr = np.full(n, float(temp_c))
        for k in range(n):
            vt, s = cell.step(current[k])
            v_true[k] = vt
            soc_true[k] = s
            temp_arr[k] = cell.temperature_c

        rng = np.random.default_rng(int(seed + abs(temp_c) * 10 + round(soh * 100)))
        v_noisy = v_true + rng.normal(0.0, v_noise_std, n)
        i_sensed = current + i_bias + rng.normal(0.0, i_noise_std, n)
        dv_dt = np.diff(v_noisy, prepend=v_noisy[0]) / self.dt

        return {
            "time": np.arange(n) * self.dt,
            "current_true": current,
            "current_sensed": i_sensed,
            "v_true": v_true,
            "v_noisy": v_noisy,
            "soc_true": soc_true,
            "temp_c": temp_arr,
            "dv_dt": dv_dt,
            "soh_true": float(soh),
            "chemistry": chemistry,
        }

    def _build_pinn_features(self, data):
        v_ref = 3.3 if data["chemistry"] == "LFP" else 3.7
        return np.stack([
            data["current_sensed"] / 5.0,
            (data["v_noisy"] - v_ref) / 0.5,
            (data["temp_c"] - 25.0) / 25.0,
            data["dv_dt"] / 0.05,
            np.full_like(data["current_sensed"], data["soh_true"]),
        ], axis=-1)

    def _build_bilstm_features(self, data):
        v_ref = 3.3 if data["chemistry"] == "LFP" else 3.7
        return np.stack([
            data["current_sensed"] / 5.0,
            (data["v_noisy"] - v_ref) / 0.5,
            (data["temp_c"] - 25.0) / 25.0,
        ], axis=-1)

    def train_deep_models(self, epochs=30, lr=2e-3, seeds=(42, 43, 44, 45, 46), chemistry="NMC"):
        """
        Train BiLSTM-Attn, PINN-SOC, and Unconstrained MLP strictly on Batch A (DST across
        100%, 90%, 80% SOH at 25 degC), reserving UDDS, FUDS, and US06 as strictly unseen schedules.
        """
        print(f"[BenchmarkEngine] Training {chemistry} neural baselines across {len(seeds)} seeds on Batch A DST...")
        train_datasets = [
            self.generate_simulation_data("dst", duration=3600, soh=soh, temp_c=25.0, chemistry=chemistry, initial_soc=0.90, seed=10 + int(soh * 100))
            for soh in (1.00, 0.90, 0.80)
        ]

        seq_len = 30
        bilstm_list, pinn_list, mlp_list = [], [], []
        q_nom = 2.30 if chemistry == "LFP" else 2.00
        r0_nom = 0.022 if chemistry == "LFP" else 0.035

        # Prebuild windows across the 3 DST aging datasets
        b_seqs, b_targets = [], []
        p_seqs, p_socs, p_currs, p_volts = [], [], [], []
        for d in train_datasets:
            fb = self._build_bilstm_features(d)
            fp = self._build_pinn_features(d)
            for idx in range(0, len(fb) - seq_len, 10):
                b_seqs.append(fb[idx : idx + seq_len])
                b_targets.append(d["soc_true"][idx + seq_len - 1])
                p_seqs.append(fp[idx : idx + seq_len])
                p_socs.append(d["soc_true"][idx : idx + seq_len])
                p_currs.append(d["current_sensed"][idx : idx + seq_len])
                p_volts.append(d["v_noisy"][idx : idx + seq_len])

        B_X = torch.tensor(np.array(b_seqs), dtype=torch.float32)
        B_Y = torch.tensor(np.array(b_targets), dtype=torch.float32)
        P_X = torch.tensor(np.array(p_seqs), dtype=torch.float32)
        P_S = torch.tensor(np.array(p_socs), dtype=torch.float32)
        P_I = torch.tensor(np.array(p_currs), dtype=torch.float32)
        P_V = torch.tensor(np.array(p_volts), dtype=torch.float32)

        n_win = B_X.shape[0]
        mse = nn.MSELoss()

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)

            # 1. BiLSTM-Attn (1-layer D=32, input=3)
            bm = BiLSTMAttentionSOC(input_dim=3, hidden_dim=32, num_layers=1)
            opt_b = optim.AdamW(bm.parameters(), lr=lr, weight_decay=1e-4)
            bm.train()
            for ep in range(epochs):
                perm = np.random.permutation(n_win)[:150]
                for idx in range(0, len(perm), 30):
                    sub = perm[idx : idx + 30]
                    opt_b.zero_grad()
                    loss = mse(bm(B_X[sub]), B_Y[sub])
                    loss.backward()
                    opt_b.step()
            bm.eval()
            bilstm_list.append(bm)

            # 2. PINN-SOC ([64, 64, 32] GELU MLP, autograd L_mono + L_Faraday + L_KVL)
            pm = PINNSOCModel(input_dim=5, h1=64, h2=64, h3=32)
            opt_p = optim.AdamW(pm.parameters(), lr=lr, weight_decay=1e-4)
            pinn_loss_fn = PhysicsInformedLoss(q_nominal_ah=q_nom, r0_nominal=r0_nom, dt=self.dt, chemistry=chemistry)
            pm.train()
            for ep in range(epochs):
                perm = np.random.permutation(n_win)[:150]
                for k in perm[:30]:
                    opt_p.zero_grad()
                    loss = pinn_loss_fn(pm, P_X[k], P_S[k], P_I[k], P_V[k])
                    loss.backward()
                    opt_p.step()
            pm.eval()
            pinn_list.append(pm)

            # 3. Unconstrained MLP (same [64, 64, 32] GELU architecture, zero physics penalty)
            mm = PINNSOCModel(input_dim=5, h1=64, h2=64, h3=32)
            opt_m = optim.AdamW(mm.parameters(), lr=lr, weight_decay=1e-4)
            mm.train()
            for ep in range(epochs):
                perm = np.random.permutation(n_win)[:150]
                for idx in range(0, len(perm), 30):
                    sub = perm[idx : idx + 30]
                    opt_m.zero_grad()
                    pred_s, _ = mm(P_X[sub].reshape(-1, 5))
                    loss = mse(pred_s, P_S[sub].reshape(-1))
                    loss.backward()
                    opt_m.step()
            mm.eval()
            mlp_list.append(mm)

        if chemistry == "NMC":
            self.bilstm_models = bilstm_list
            self.pinn_models = pinn_list
            self.mlp_models = mlp_list
        return bilstm_list, pinn_list, mlp_list

    def run_benchmark_on_dataset(
        self,
        data,
        initial_soc_guess=None,
        p_init_soc=None,
        include_ukf=False,
        include_mlp=False,
        pidkf_models=None,
    ):
        """
        Evaluate model-based observers (all initialized fairly from nominal 100% SOH @ 25 degC!)
        and neural models (across all trained seeds) on a test dataset.
        """
        n_steps = len(data["time"])
        currents = data["current_sensed"]
        voltages = data["v_noisy"]
        soc_true = data["soc_true"]
        init_soc = float(soc_true[0]) if initial_soc_guess is None else float(initial_soc_guess)
        chem = data.get("chemistry", "NMC")

        # Strictly fresh nominal cell model (ZERO oracle aging/temperature parameters!)
        def make_nom_cell():
            if chem == "LFP":
                return BatteryCellLFP(nominal_capacity_ah=2.30, soh=1.0, temperature_c=25.0, dt=self.dt)
            return BatteryCellNMC(nominal_capacity_ah=2.00, soh=1.0, temperature_c=25.0, dt=self.dt)

        p_diag = (p_init_soc, 1e-4, 1e-4) if p_init_soc is not None else None

        ekf = ExtendedKalmanFilter(make_nom_cell())
        ekf.reset(init_soc, p_diag=p_diag)
        aekf = AdaptiveEKF(make_nom_cell())
        aekf.reset(init_soc, p_diag=p_diag)
        dekf = DualExtendedKalmanFilter(make_nom_cell())
        dekf.reset(init_soc, p_diag=p_diag)
        smo = SlidingModeObserver(make_nom_cell())
        smo.reset(init_soc)

        observers = {"EKF": ekf, "AEKF": aekf, "DEKF": dekf, "SMO": smo}
        if include_ukf:
            ukf = UnscentedKalmanFilter(make_nom_cell())
            ukf.reset(init_soc, p_diag=p_diag)
            observers["UKF"] = ukf

        results = {}
        for name, obs in observers.items():
            traj = np.zeros(n_steps)
            for k in range(n_steps):
                out = obs.step(currents[k], voltages[k])
                traj[k] = out[0]
            m = compute_metrics(traj, soc_true)
            m["soc_est"] = traj
            results[name] = m

        # BiLSTM-Attn multi-seed evaluation
        if self.bilstm_models:
            fb = self._build_bilstm_features(data)
            seq_len = 30
            X_wins = torch.tensor(np.array([fb[k - seq_len + 1 : k + 1] for k in range(seq_len - 1, n_steps)]), dtype=torch.float32)
            seed_metrics = []
            seed_trajs = []
            with torch.no_grad():
                for bm in self.bilstm_models:
                    s_traj = np.full(n_steps, init_soc)
                    s_traj[seq_len - 1 :] = bm(X_wins).numpy()
                    seed_trajs.append(s_traj)
                    seed_metrics.append(compute_metrics(s_traj, soc_true))
            mean_m = {k: float(np.mean([sm[k] for sm in seed_metrics])) for k in seed_metrics[0]}
            mean_m["rmse_sd"] = float(np.std([sm["rmse_pct"] for sm in seed_metrics]))
            mean_m["soc_est"] = seed_trajs[0]
            results["BiLSTM-Attn"] = mean_m

        # PINN-SOC multi-seed evaluation
        if self.pinn_models:
            fp = self._build_pinn_features(data)
            # In blind test (real field operation), latched SOH flag equals fresh nominal 1.0 unless periodic CC RPT updated it
            fp_causal = fp.copy()
            fp_causal[:, 4] = 1.0
            X_p = torch.tensor(fp_causal, dtype=torch.float32)
            seed_metrics, seed_trajs = [], []
            with torch.no_grad():
                for pm in self.pinn_models:
                    s_traj, _ = pm(X_p)
                    arr = s_traj.numpy()
                    seed_trajs.append(arr)
                    seed_metrics.append(compute_metrics(arr, soc_true))
            mean_m = {k: float(np.mean([sm[k] for sm in seed_metrics])) for k in seed_metrics[0]}
            mean_m["rmse_sd"] = float(np.std([sm["rmse_pct"] for sm in seed_metrics]))
            mean_m["soc_est"] = seed_trajs[0]
            results["PINN-SOC"] = mean_m

        if include_mlp and self.mlp_models:
            fp = self._build_pinn_features(data)
            fp[:, 4] = 1.0
            X_m = torch.tensor(fp, dtype=torch.float32)
            seed_metrics, seed_trajs = [], []
            with torch.no_grad():
                for mm in self.mlp_models:
                    s_traj, _ = mm(X_m)
                    arr = s_traj.numpy()
                    seed_trajs.append(arr)
                    seed_metrics.append(compute_metrics(arr, soc_true))
            mean_m = {k: float(np.mean([sm[k] for sm in seed_metrics])) for k in seed_metrics[0]}
            mean_m["rmse_sd"] = float(np.std([sm["rmse_pct"] for sm in seed_metrics]))
            mean_m["soc_est"] = seed_trajs[0]
            results["MLP (Unconstrained)"] = mean_m

        if pidkf_models:
            from pidkf_hybrid import PhysicsInformedDualKalmanFilter

            temps = data["temp_c"]
            seed_metrics, seed_trajs = [], []
            for pnn in pidkf_models:
                pidkf = PhysicsInformedDualKalmanFilter(make_nom_cell(), nn_model=pnn)
                pidkf.reset(init_soc, p_diag=p_diag)
                traj = np.zeros(n_steps)
                for k in range(n_steps):
                    traj[k] = pidkf.step(currents[k], voltages[k], temp_c=float(temps[k]))[0]
                seed_trajs.append(traj)
                seed_metrics.append(compute_metrics(traj, soc_true))
            mean_m = {k: float(np.mean([sm[k] for sm in seed_metrics])) for k in seed_metrics[0]}
            mean_m["rmse_sd"] = float(np.std([sm["rmse_pct"] for sm in seed_metrics]))
            mean_m["soc_est"] = seed_trajs[0]
            results["PIDKF"] = mean_m

        return results
