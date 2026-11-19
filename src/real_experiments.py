"""
Tier 1 Real Experimental Laboratory Validation Runner
Journal of Energy Storage (Elsevier)

Runs all benchmark estimators strictly on real public cycler measurements located in NASABattery/:
1. XJTU Batch-5 LiNi0.5Co0.2Mn0.3O2 (NMC532, 2.00 Ah nominal, 2.5-4.2 V) Random Walk Dynamic Cycles:
   - Strictly calibrated & neural-trained on Batch A (RW_battery-1, RW_battery-2)
   - Evaluated on strictly unseen Batch B test cells (RW_battery-3, RW_battery-4) at:
     * Fresh baseline (~100% of initial measured capacity Q_init)
     * Aged degraded checkpoint (~86% SOH)
2. NASA Ames PCoE 18650 (LiCoO2, 2.00 Ah rated) Aging & Extreme Thermal Validation:
   - Calibrated strictly on B0005 cycle 0 at 24 degC
   - Evaluated on unseen cells across thermal & aging stress:
     * B0007 (24 degC, 2A CC): Fresh (~100%) and Aged (~81% SOH, cycle 112)
     * B0055 & B0056 (4 degC, 2A CC cold extreme, usable cold Q = 1.32-1.34 Ah vs 2.00 Ah rated)
     * B0029 & B0030 (43 degC, 4A CC hot extreme, cycle 30 ~89% of 2.00 Ah)
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.interpolate import PchipInterpolator
import torch
import torch.nn as nn
import torch.optim as optim

import nasa_data
from models_data_driven import BiLSTMAttentionSOC
from models_physics_informed import PINNSOCModel
from pidkf_hybrid import ResidualCompensatorMLP, build_feature_vector

XJTU_BATCH5_DIR = os.environ.get(
    "XJTU_BATCH5_DIR",
    r"D:\bách khoa Đà Nẵng\nghiencuukhoahoc\baotri\data\NASABattery\Battery Dataset\Batch-5",
)


def extract_xjtu_cycle(cell_idx, cyc_idx):
    """Extract monotonic 1.0-Hz post-charge rest + discharge + final rest segment from XJTU .mat."""
    p = os.path.join(XJTU_BATCH5_DIR, f"RW_battery-{cell_idx}.mat")
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    r = m["data"][cyc_idx]
    t = np.atleast_1d(r.relative_time_min) * 60.0
    i = np.atleast_1d(r.current_A)
    v = np.atleast_1d(r.voltage_V)
    tc = np.atleast_1d(r.temperature_C)
    resets = np.where(np.diff(t) < -1.0)[0] + 1
    segs = np.split(np.arange(len(t)), resets)
    idxs = np.concatenate(segs[1:])
    t_cat = []
    off = 0.0
    for sg in segs[1:]:
        ts = t[sg] - t[sg][0]
        t_cat.append(ts + off)
        off += (ts[-1] + 1.0)
    t_all = np.concatenate(t_cat)
    tg = np.arange(0.0, t_all[-1], 1.0)
    gidx = np.clip(np.searchsorted(t_all, tg, side="right") - 1, 0, len(t_all) - 1)
    # Sample-align I, V, T so instantaneous ohmic step edges are preserved without linear blurring
    i_dis = -i[idxs][gidx]
    v_u = v[idxs][gidx]
    tc_u = tc[idxs][gidx]
    return tg, i_dis, v_u, tc_u, str(r.description)


def get_xjtu_rpt_capacity(cell_idx, target_cyc_idx):
    """Measure true RPT 1C capacity nearest to target_cyc_idx."""
    p = os.path.join(XJTU_BATCH5_DIR, f"RW_battery-{cell_idx}.mat")
    m = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    d = m["data"]
    best_q = None
    best_dist = 1e9
    q_init = None
    for idx, r in enumerate(d):
        if "test capacity" in str(r.description).lower():
            t = np.atleast_1d(r.relative_time_min) * 60.0
            i = np.atleast_1d(r.current_A)
            v = np.atleast_1d(r.voltage_V)
            res = np.where(np.diff(t) < -1.0)[0] + 1
            segs = np.split(np.arange(len(t)), res)
            q_seg = max(float(np.atleast_1d(r.capacity_Ah)[sg[-1]]) for sg in segs if np.min(i[sg]) < -1.5 and np.min(v[sg]) < 2.6)
            if q_init is None:
                q_init = q_seg
            if abs(idx - target_cyc_idx) < best_dist:
                best_dist = abs(idx - target_cyc_idx)
                best_q = q_seg
    return float(best_q), float(q_init)


class EmpiricalBatteryCell:
    """Wrapper providing monotonic PCHIP OCV, derivative, and nominal 2-RC parameters from Batch A."""

    def __init__(self, knots, ocv_vals, q_nominal, r0, r1, c1, r2, c2, dt=1.0, chemistry="NMC"):
        self.chemistry = chemistry
        self.knots = np.array(knots, dtype=float)
        self.ocv_vals = np.array(ocv_vals, dtype=float)
        self.pchip = PchipInterpolator(self.knots, self.ocv_vals)
        self.dpchip = self.pchip.derivative()
        self.q_nominal = float(q_nominal)
        self.r0_base = float(r0)
        self.r1_base = float(r1)
        self.c1_base = float(c1)
        self.r2_base = float(r2)
        self.c2_base = float(c2)
        self.dt = float(dt)

    def get_eta(self, current_a):
        return 1.0 if current_a >= 0.0 else 0.995

    def get_ocv(self, soc):
        return float(self.pchip(np.clip(soc, 0.0, 1.0)))

    def get_docv_dsoc(self, soc):
        return float(max(0.05, self.dpchip(np.clip(soc, 0.0, 1.0))))


def calibrate_empirical_model(i_segs, v_segs, soc_segs, knots, r0_lb=0.032, r0_fixed=None, q_nom=2.0, masks=None):
    n_k = len(knots)
    inc0 = np.diff(np.linspace(2.68, 4.195, n_k), prepend=2.68)
    inc0[0] = 2.68
    if masks is None:
        masks = [np.ones_like(v, dtype=bool) for v in v_segs]

    if r0_fixed is None:
        p0 = np.concatenate([inc0, [max(r0_lb + 0.008, 0.042), 0.015, 500.0, 0.020, 4000.0]])
        lb = np.concatenate([[2.2] + [0.002] * (n_k - 1), [r0_lb, 0.002, 15.0, 0.002, 200.0]])
        ub = np.concatenate([[3.8] + [1.2] * (n_k - 1), [0.120, 0.150, 12000.0, 0.150, 50000.0]])
    else:
        p0 = np.concatenate([inc0, [0.020, 500.0, 0.025, 3500.0]])
        lb = np.concatenate([[2.2] + [0.002] * (n_k - 1), [0.002, 15.0, 0.002, 200.0]])
        ub = np.concatenate([[3.8] + [1.2] * (n_k - 1), [0.150, 12000.0, 0.150, 50000.0]])

    def sim(params):
        vals = np.cumsum(params[:n_k])
        pchip = PchipInterpolator(knots, vals)
        if r0_fixed is None:
            r0, r1, c1, r2, c2 = params[n_k:]
        else:
            r0 = float(r0_fixed)
            r1, c1, r2, c2 = params[n_k:]
        e1, e2 = np.exp(-1.0 / (r1 * c1)), np.exp(-1.0 / (r2 * c2))
        out = []
        for ik, vk, sk, mk in zip(i_segs, v_segs, soc_segs, masks):
            u1 = lfilter([r1 * (1.0 - e1)], [1.0, -e1], ik)
            u2 = lfilter([r2 * (1.0 - e2)], [1.0, -e2], ik)
            out.append(((pchip(sk) - ik * r0 - u1 - u2) - vk)[mk])
        return np.concatenate(out)

    res = least_squares(sim, p0, bounds=(lb, ub), max_nfev=800)
    rms_mv = float(np.sqrt(np.mean(sim(res.x) ** 2))) * 1000.0
    vals = np.cumsum(res.x[:n_k])
    if r0_fixed is None:
        r0, r1, c1, r2, c2 = res.x[n_k:]
    else:
        r0 = float(r0_fixed)
        r1, c1, r2, c2 = res.x[n_k:]
    cell = EmpiricalBatteryCell(knots, vals, q_nom, r0, r1, c1, r2, c2)
    return cell, rms_mv


def eval_observers_real(nom_cell, i_meas, v_meas, soc_true, temp_arr, pidkf_nn=None, p0_theta=None):
    from models_model_based import ExtendedKalmanFilter, AdaptiveEKF, DualExtendedKalmanFilter, SlidingModeObserver
    from pidkf_hybrid import PhysicsInformedDualKalmanFilter

    if p0_theta is None:
        p0_theta = np.diag([1e-3, 1e-2])

    ekf = ExtendedKalmanFilter(nom_cell)
    ekf.reset(soc_true[0])
    aekf = AdaptiveEKF(nom_cell)
    aekf.reset(soc_true[0])
    dekf = DualExtendedKalmanFilter(nom_cell, p0_theta=p0_theta, q_theta=np.diag([1e-7, 1e-6]))
    dekf.reset(soc_true[0])
    smo = SlidingModeObserver(nom_cell)
    smo.reset(soc_true[0])

    n = len(i_meas)
    res_dict = {}
    for name, obs in [("EKF", ekf), ("AEKF", aekf), ("DEKF", dekf), ("SMO", smo)]:
        est = np.array([obs.step(i_meas[k], v_meas[k])[0] for k in range(n)])
        err = (est - soc_true) * 100.0
        res_dict[name] = (float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err))))

    if pidkf_nn is not None:
        pidkf = PhysicsInformedDualKalmanFilter(
            nom_cell, nn_model=pidkf_nn, p0_theta=p0_theta, q_theta=np.diag([1e-7, 1e-6]), sigma_nn=0.010, sigma_approx=0.002
        )
        pidkf.reset(soc_true[0])
        est_p = np.array([pidkf.step(i_meas[k], v_meas[k], temp_c=float(temp_arr[k]))[0] for k in range(n)])
        err_p = (est_p - soc_true) * 100.0
        res_dict["PIDKF"] = (float(np.sqrt(np.mean(err_p ** 2))), float(np.max(np.abs(err_p))))
    return res_dict


def run_all_real_experiments():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    # =========================================================================
    # PART A: XJTU Batch-5 Random Walk Real Cycler Experiments (NMC532, 2.0 Ah)
    # =========================================================================
    knots_x = np.array([0.0, 0.02, 0.05, 0.09, 0.14, 0.21, 0.30, 0.42, 0.56, 0.70, 0.84, 0.94, 1.0])
    _, i_rpt, v_rpt, _ = extract_xjtu_cycle(1, 20)[:4]
    q_rpt_1, q_init_1 = get_xjtu_rpt_capacity(1, 20)
    soc_rpt_1 = np.clip(1.0 - (np.cumsum(i_rpt) / 3600.0) / q_rpt_1, 0.0, 1.0)
    mask_rpt = (i_rpt > 0.1) | (soc_rpt_1 > 0.05)

    _, i_rw1, v_rw1, _ = extract_xjtu_cycle(1, 0)[:4]
    q_rw1, _ = get_xjtu_rpt_capacity(1, 0)
    soc_rw1 = np.clip(1.0 - (np.cumsum(i_rw1) / 3600.0) / q_rw1, 0.0, 1.0)
    _, i_rw2, v_rw2, _ = extract_xjtu_cycle(2, 0)[:4]
    q_rw2, _ = get_xjtu_rpt_capacity(2, 0)
    soc_rw2 = np.clip(1.0 - (np.cumsum(i_rw2) / 3600.0) / q_rw2, 0.0, 1.0)

    xjtu_nom_cell, xjtu_fit_mv = calibrate_empirical_model(
        [i_rw1, i_rw2, i_rpt],
        [v_rw1, v_rw2, v_rpt],
        [soc_rw1, soc_rw2, soc_rpt_1],
        knots_x,
        r0_lb=0.032,
        q_nom=q_init_1,
        masks=[np.ones_like(v_rw1, bool), np.ones_like(v_rw2, bool), mask_rpt],
    )
    print(f"[XJTU Batch A Calibrated Model] Q_init={q_init_1:.3f} Ah, Fit RMS={xjtu_fit_mv:.2f} mV, "
          f"R0={xjtu_nom_cell.r0_base*1000:.1f} mOhm, R1={xjtu_nom_cell.r1_base*1000:.1f} mOhm, R2={xjtu_nom_cell.r2_base*1000:.1f} mOhm")

    # Gather Batch A random-walk cycles from RW_1 & RW_2 across fresh, mid, and aged stages to train NN baselines & PIDKF
    train_cycles_a = []
    for cid, cids_cyc in [(1, [0, 2, 100, 175, 195]), (2, [0, 2, 150, 260, 282])]:
        for cidx in cids_cyc:
            _, ik, vk, tk, _ = extract_xjtu_cycle(cid, cidx)
            q_true, _ = get_xjtu_rpt_capacity(cid, cidx)
            sk = np.clip(1.0 - (np.cumsum(ik) / 3600.0) / q_true, 0.0, 1.0)
            train_cycles_a.append((ik, vk, tk, sk, q_true / q_init_1))

    # Build PIDKF causal residual dataset on Batch A
    from models_model_based import DualExtendedKalmanFilter
    X_pid, Y_pid, I_pid = [], [], []
    b_X, b_Y, p_X, p_S, p_I, p_V, p_SOH = [], [], [], [], [], [], []
    seq_len = 30
    for ik, vk, tk, sk, soh_c in train_cycles_a:
        dekf_a = DualExtendedKalmanFilter(xjtu_nom_cell, p0_theta=np.diag([1e-3, 1e-2]), q_theta=np.diag([1e-7, 1e-6]))
        dekf_a.reset(sk[0])
        for k in range(len(ik)):
            _, xp, r0p, _ = dekf_a.predict_state_prior(ik[k])
            dekf_a.step(ik[k], vk[k])
            v_ecm_ref = xjtu_nom_cell.get_ocv(sk[k]) - ik[k] * r0p - xp[1, 0] - xp[2, 0]
            dv_res = vk[k] - v_ecm_ref
            X_pid.append(build_feature_vector(ik[k], tk[k], xp[0, 0], xp[1, 0], xp[2, 0], r0p, "NMC"))
            Y_pid.append(np.clip(dv_res, -0.15, 0.15))
            I_pid.append(ik[k])

        # Features for BiLSTM & PINN
        dv_dt = np.diff(vk, prepend=vk[0])
        fb = np.stack([ik / 5.0, (vk - 3.7) / 0.5, (tk - 25.0) / 25.0], axis=-1)
        fp = np.stack([ik / 5.0, (vk - 3.7) / 0.5, (tk - 25.0) / 25.0, dv_dt / 0.05, np.full_like(ik, soh_c)], axis=-1)
        for j in range(0, len(ik) - seq_len, 25):
            b_X.append(fb[j : j + seq_len])
            b_Y.append(sk[j + seq_len - 1])
            p_X.append(fp[j : j + seq_len])
            p_S.append(sk[j : j + seq_len])
            p_I.append(ik[j : j + seq_len])
            p_V.append(vk[j : j + seq_len])

    X_pid_t = torch.tensor(np.array(X_pid, dtype=np.float32))
    Y_pid_t = torch.tensor(np.array(Y_pid, dtype=np.float32))
    I_pid_t = torch.tensor(np.array(I_pid, dtype=np.float32))
    B_X = torch.tensor(np.array(b_X, dtype=np.float32))
    B_Y = torch.tensor(np.array(b_Y, dtype=np.float32))
    P_X = torch.tensor(np.array(p_X, dtype=np.float32))
    P_S = torch.tensor(np.array(p_S, dtype=np.float32))

    # Evaluate across 5 seeds on strictly held-out XJTU Batch B cells: RW_battery-3 & RW_battery-4 (Fresh & Aged ~86% SOH)
    seeds = (42, 43, 44, 45, 46)
    pid_models, bilstm_models, pinn_models = [], [], []
    mse = nn.MSELoss()
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        pm = ResidualCompensatorMLP(kappa_0=0.015, kappa_1=0.022, i_max=6.0)
        opt = optim.AdamW(pm.parameters(), lr=1e-2, weight_decay=1e-5)
        for _ in range(300):
            opt.zero_grad()
            raw = pm.raw_output(X_pid_t)
            vmax = pm.kappa_0 + pm.kappa_1 * torch.clamp(torch.abs(I_pid_t), max=pm.i_max)
            pred = torch.clamp(raw, -vmax, vmax)
            loss = torch.mean((pred - Y_pid_t) ** 2) + 0.5 * torch.mean(torch.relu(torch.abs(raw) - vmax) ** 2 + (torch.relu(I_pid_t * pred) ** 2))
            loss.backward()
            opt.step()
        pm.eval()
        pid_models.append(pm)

        bm = BiLSTMAttentionSOC(input_dim=3, hidden_dim=32, num_layers=1)
        opt_b = optim.AdamW(bm.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(25):
            perm = np.random.permutation(len(B_X))[:150]
            for ix in range(0, len(perm), 30):
                sub = perm[ix : ix + 30]
                opt_b.zero_grad()
                l = mse(bm(B_X[sub]), B_Y[sub])
                l.backward()
                opt_b.step()
        bm.eval()
        bilstm_models.append(bm)

        pnn = PINNSOCModel(input_dim=5, h1=64, h2=64, h3=32)
        opt_p = optim.AdamW(pnn.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(25):
            perm = np.random.permutation(len(P_X))[:150]
            for ix in perm[:30]:
                opt_p.zero_grad()
                sp, _ = pnn(P_X[ix])
                l = mse(sp, P_S[ix])
                l.backward()
                opt_p.step()
        pnn.eval()
        pinn_models.append(pnn)

    xjtu_rows = []
    for lbl, cell_id, cyc_id in [
        ("XJTU RW_3 Fresh (~100% SOH)", 3, 1),
        ("XJTU RW_4 Fresh (~100% SOH)", 4, 1),
        ("XJTU RW_3 Aged (~86% SOH)", 3, 326),
        ("XJTU RW_4 Aged (~87% SOH)", 4, 238),
    ]:
        _, ik, vk, tk, _ = extract_xjtu_cycle(cell_id, cyc_id)
        q_true, q_i = get_xjtu_rpt_capacity(cell_id, cyc_id)
        sk = np.clip(1.0 - (np.cumsum(ik) / 3600.0) / q_true, 0.0, 1.0)
        soh_true = q_true / q_i

        # Deterministic observers once + 5-seed PIDKF
        base_res = eval_observers_real(xjtu_nom_cell, ik, vk, sk, tk, pidkf_nn=None)
        pid_rmses, pid_maxes = [], []
        for pm in pid_models:
            r = eval_observers_real(xjtu_nom_cell, ik, vk, sk, tk, pidkf_nn=pm)
            pid_rmses.append(r["PIDKF"][0])
            pid_maxes.append(r["PIDKF"][1])

        fb = np.stack([ik / 5.0, (vk - 3.7) / 0.5, (tk - 25.0) / 25.0], axis=-1)
        X_bw = torch.tensor(np.array([fb[j - seq_len + 1 : j + 1] for j in range(seq_len - 1, len(ik))], dtype=np.float32))
        bi_rmses = []
        with torch.no_grad():
            for bm in bilstm_models:
                sp = np.full(len(ik), sk[0])
                sp[seq_len - 1 :] = bm(X_bw).numpy()
                bi_rmses.append(float(np.sqrt(np.mean(((sp - sk) * 100.0) ** 2))))

        dv_dt = np.diff(vk, prepend=vk[0])
        fp = np.stack([ik / 5.0, (vk - 3.7) / 0.5, (tk - 25.0) / 25.0, dv_dt / 0.05, np.ones_like(ik)], axis=-1)
        X_pt = torch.tensor(fp, dtype=torch.float32)
        pi_rmses = []
        with torch.no_grad():
            for pnn in pinn_models:
                sp, _ = pnn(X_pt)
                pi_rmses.append(float(np.sqrt(np.mean(((sp.numpy() - sk) * 100.0) ** 2))))

        xjtu_rows.append({
            "Test Cell & Condition": lbl,
            "SOH (%)": round(soh_true * 100.0, 1),
            "EKF RMSE (%)": round(base_res["EKF"][0], 3),
            "AEKF RMSE (%)": round(base_res["AEKF"][0], 3),
            "DEKF RMSE (%)": round(base_res["DEKF"][0], 3),
            "SMO RMSE (%)": round(base_res["SMO"][0], 3),
            "BiLSTM RMSE (%)": f"{np.mean(bi_rmses):.3f} +/- {np.std(bi_rmses):.3f}",
            "PINN-SOC RMSE (%)": f"{np.mean(pi_rmses):.3f} +/- {np.std(pi_rmses):.3f}",
            "PIDKF RMSE (%)": f"{np.mean(pid_rmses):.3f} +/- {np.std(pid_rmses):.3f}",
            "PIDKF MaxE (%)": round(float(np.mean(pid_maxes)), 3),
        })

    df_xjtu = pd.DataFrame(xjtu_rows)
    df_xjtu.to_csv(os.path.join(results_dir, "tier1_xjtu_random_walk.csv"), index=False)
    print("\n=== TIER 1 REAL CYCLER RESULTS: XJTU Batch-5 Random Walk (Unseen Cells RW_3, RW_4) ===")
    print(df_xjtu.to_string(index=False))

    # =========================================================================
    # PART B: NASA PCoE 18650 Aging & Thermal Extremes (24C, 43C, 4C)
    # =========================================================================
    d5 = nasa_data.resample_uniform(nasa_data.discharge_cycles("B0005")[0], dt=1.0)
    s5 = np.clip(1.0 - (np.cumsum(d5["i"]) / 3600.0) / d5["capacity_ah"], 0.0, 1.0)
    m5 = (d5["i"] > 0.1) | (s5 > 0.04)
    # Anchor ohmic R0 = 0.04467 Ohm strictly from B0005 fresh EIS Re (eliminates CC collinearity)
    r0_b0005_eis = float(nasa_data.impedance_records("B0005")[0]["Re"])
    nasa_nom_cell, nasa_fit_mv = calibrate_empirical_model(
        [d5["i"]], [d5["v"]], [s5], knots_x, r0_fixed=r0_b0005_eis, q_nom=d5["capacity_ah"], masks=[m5]
    )
    print(f"\n[NASA B0005 Calibrated Model (EIS-anchored)] Fit RMS={nasa_fit_mv:.2f} mV, "
          f"R0={nasa_nom_cell.r0_base*1000:.2f} mOhm, R1={nasa_nom_cell.r1_base*1000:.1f} mOhm, R2={nasa_nom_cell.r2_base*1000:.1f} mOhm")

    # Train lightweight PIDKF compensator on NASA Batch A (B0005 fresh & aged, B0029 @ 43C, B0053 @ 4C)
    NASA_ROOT = nasa_data.DATA_ROOT
    def load_nasa_extra_dis(sub, cid, dis_idx):
        m = sio.loadmat(os.path.join(NASA_ROOT, sub, f"{cid}.mat"), squeeze_me=True, struct_as_record=False)
        cs = [c for c in m[cid].cycle if str(c.type).strip() == "discharge"]
        c = cs[dis_idx]
        d = c.data
        t = np.atleast_1d(np.asarray(d.Time, dtype=float))
        v = np.atleast_1d(np.asarray(d.Voltage_measured, dtype=float))
        i = -np.atleast_1d(np.asarray(d.Current_measured, dtype=float))
        tp = np.atleast_1d(np.asarray(d.Temperature_measured, dtype=float))
        tg = np.arange(0.0, t[-1] - t[0], 1.0)
        gidx = np.clip(np.searchsorted(t - t[0], tg, side="right") - 1, 0, len(t) - 1)
        i_u = i[gidx]
        q_ah = np.cumsum(i_u) / 3600.0
        q_cap = float(q_ah[-1])
        return i_u, v[gidx], tp[gidx], np.clip(1.0 - q_ah / q_cap, 0.0, 1.0), q_cap, float(c.ambient_temperature)

    X_np, Y_np, I_np = [], [], []
    for sub, cid, d_idx in [
        ("1. BatteryAgingARC-FY08Q4", "B0005", 0),
        ("1. BatteryAgingARC-FY08Q4", "B0005", 110),
        ("3. BatteryAgingARC_25-44", "B0029", 5),
        ("6. BatteryAgingARC_53_54_55_56", "B0053", 5),
    ]:
        ik, vk, tk, sk, q_c, _ = load_nasa_extra_dis(sub, cid, d_idx)
        dekf_n = DualExtendedKalmanFilter(nasa_nom_cell, p0_theta=np.diag([1e-3, 1e-2]), q_theta=np.diag([1e-7, 1e-6]))
        dekf_n.reset(sk[0])
        for k in range(len(ik)):
            _, xp, r0p, _ = dekf_n.predict_state_prior(ik[k])
            dekf_n.step(ik[k], vk[k])
            v_ecm = nasa_nom_cell.get_ocv(sk[k]) - ik[k] * r0p - xp[1, 0] - xp[2, 0]
            X_np.append(build_feature_vector(ik[k], tk[k], xp[0, 0], xp[1, 0], xp[2, 0], r0p, "NMC"))
            Y_np.append(np.clip(vk[k] - v_ecm, -0.15, 0.15))
            I_np.append(ik[k])

    X_npt = torch.tensor(np.array(X_np, dtype=np.float32))
    Y_npt = torch.tensor(np.array(Y_np, dtype=np.float32))
    I_npt = torch.tensor(np.array(I_np, dtype=np.float32))
    nasa_pids = []
    for s in seeds:
        torch.manual_seed(s)
        pm_n = ResidualCompensatorMLP(kappa_0=0.015, kappa_1=0.025, i_max=6.0)
        opt_n = optim.AdamW(pm_n.parameters(), lr=1e-2, weight_decay=1e-5)
        for _ in range(300):
            opt_n.zero_grad()
            raw = pm_n.raw_output(X_npt)
            vmax = pm_n.kappa_0 + pm_n.kappa_1 * torch.clamp(torch.abs(I_npt), max=pm_n.i_max)
            pred = torch.clamp(raw, -vmax, vmax)
            loss = torch.mean((pred - Y_npt) ** 2) + 0.5 * torch.mean(torch.relu(torch.abs(raw) - vmax) ** 2 + (torch.relu(I_npt * pred) ** 2))
            loss.backward()
            opt_n.step()
        pm_n.eval()
        nasa_pids.append(pm_n)

    nasa_rows = []
    for lbl, sub, cid, d_idx in [
        ("NASA B0007 (24C Fresh 2A)", "1. BatteryAgingARC-FY08Q4", "B0007", 0),
        ("NASA B0007 (24C Aged 81% 2A)", "1. BatteryAgingARC-FY08Q4", "B0007", 112),
        ("NASA B0030 (43C Hot 4A)", "3. BatteryAgingARC_25-44", "B0030", 10),
        ("NASA B0055 (4C Cold 2A)", "6. BatteryAgingARC_53_54_55_56", "B0055", 5),
        ("NASA B0056 (4C Cold 2A)", "6. BatteryAgingARC_53_54_55_56", "B0056", 5),
    ]:
        ik, vk, tk, sk, q_c, amb = load_nasa_extra_dis(sub, cid, d_idx)
        base_r = eval_observers_real(nasa_nom_cell, ik, vk, sk, tk, pidkf_nn=None)
        p_rmses = [eval_observers_real(nasa_nom_cell, ik, vk, sk, tk, pidkf_nn=pm)["PIDKF"][0] for pm in nasa_pids]
        nasa_rows.append({
            "NASA Cell & Condition": lbl,
            "Q_usable (Ah)": round(q_c, 3),
            "Amb (C)": amb,
            "EKF RMSE (%)": round(base_r["EKF"][0], 3),
            "AEKF RMSE (%)": round(base_r["AEKF"][0], 3),
            "DEKF RMSE (%)": round(base_r["DEKF"][0], 3),
            "SMO RMSE (%)": round(base_r["SMO"][0], 3),
            "PIDKF RMSE (%)": f"{np.mean(p_rmses):.3f} +/- {np.std(p_rmses):.3f}",
        })
    df_nasa = pd.DataFrame(nasa_rows)
    df_nasa.to_csv(os.path.join(results_dir, "tier1_nasa_thermal_aging.csv"), index=False)
    print("\n=== TIER 1 REAL CYCLER RESULTS: NASA PCoE Thermal & Aging (Unseen Cells) ===")
    print(df_nasa.to_string(index=False))


if __name__ == "__main__":
    run_all_real_experiments()
