"""
Hybrid Physics-Informed Dual Kalman Filter (PIDKF)
Journal of Energy Storage (Elsevier)

Exact implementation synchronized with Section 3.7, Appendix C.2, and embedded_c/pidkf.c:
- Backbone: 3-state (NMC) or 4-state (LFP with ZOH hysteresis) DEKF initialized STRICTLY
  from identical fresh nominal 100% SOH priors (ZERO oracle aging leakage).
- Causal Compensator: 257-parameter [6 -> 16 -> 8 -> 1] GELU MLP operating on
  u_k = [I_k, T_k, z_k^-, U_{1,k}^-, U_{2,k}^-, R_{0,k}^-]^T, using fast-converging
  online ohmic resistance R_{0,k} to condition polarization compensation in real time.
- Butler-Volmer Saturation + Hard Dissipative Passivity Projection:
  V_pol^max(I_k) = kappa_0 + kappa_1 * min(|I_k|, I_max), I_k * delta_V_comp <= 0.
- Supports both Exact Autograd Jacobian and Exogenous Sensitivity Decoupling.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from battery_model import BatteryCellNMC, BatteryCellLFP
from drive_cycles import DriveCycles
from models_model_based import DualExtendedKalmanFilter


class ResidualCompensatorMLP(nn.Module):
    """
    2-hidden-layer [6 -> 16 -> 8 -> 1] GELU MLP = 257 parameters (1.03 KB Flash, 560 FLOPs/step).
    Explicitly enforces Butler-Volmer symmetric envelope and optional dissipative passivity.
    """

    def __init__(self, kappa_0=0.015, kappa_1=0.022, i_max=6.0, scale_out=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.kappa_0 = float(kappa_0)
        self.kappa_1 = float(kappa_1)
        self.i_max = float(i_max)
        self.scale_out = float(scale_out)

    def raw_output(self, x):
        return self.net(x).squeeze(-1) * self.scale_out

    def forward(self, x, raw_current, hard_passivity=False, rest_th=0.010):
        raw = self.raw_output(x)
        vmax = self.kappa_0 + self.kappa_1 * torch.clamp(torch.abs(raw_current), max=self.i_max)
        clipped = torch.clamp(raw, -vmax, vmax)
        if hard_passivity:
            # Enforce I_k * delta_V <= 0 strictly outside rest gate
            active = torch.abs(raw_current) > rest_th
            violate = active & ((raw_current * clipped) > 0.0)
            clipped = torch.where(violate, torch.zeros_like(clipped), clipped)
        return clipped


def build_feature_vector(ik, temp_c, z_prior, u1_prior, u2_prior, r0_est, chemistry="NMC"):
    r0_nom = 0.022 if chemistry == "LFP" else 0.035
    return [
        float(ik) / 4.0,
        (float(temp_c) - 25.0) / 20.0,
        float(z_prior),
        float(u1_prior) * 15.0,
        float(u2_prior) * 15.0,
        (float(r0_est) - r0_nom) / 0.030,
    ]


def train_residual_compensator(
    chemistry="NMC",
    seed=42,
    epochs=600,
    lr=1.5e-2,
    lambda_phys=0.50,
    gamma_i=1.0,
    lambda_reg=1e-5,
    unconstrained_blackbox=False,
):
    """
    Strictly causal training on Batch A DST across 100%, 90%, 80% SOH and 0, 25, 45 degC.
    Target residual delta_V_target = -( (U1+U2)_plant - (U1+U2)_nom ) excludes R0 & Q_act,
    preventing closed-loop leakage and training the compensator purely on unmodeled RC kinetics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    I_dst = DriveCycles.generate_dst(duration_sec=3600, dt=1.0)
    X_list, Y_list, I_list = [], [], []
    is_lfp = (chemistry == "LFP")

    for soh in (1.00, 0.90, 0.80):
        for temp_c in (0.0, 25.0, 45.0):
            if is_lfp:
                plant = BatteryCellLFP(nominal_capacity_ah=2.30, soh=soh, temperature_c=temp_c, dt=1.0)
                nom = BatteryCellLFP(nominal_capacity_ah=2.30, soh=1.0, temperature_c=25.0, dt=1.0)
                z0 = 0.85
            else:
                plant = BatteryCellNMC(nominal_capacity_ah=2.00, soh=soh, temperature_c=temp_c, dt=1.0)
                nom = BatteryCellNMC(nominal_capacity_ah=2.00, soh=1.0, temperature_c=25.0, dt=1.0)
                z0 = 0.95
            plant.reset_state(initial_soc=z0)
            nom.reset_state(initial_soc=z0)

            for ik in I_dst:
                plant.step(ik)
                nom.step(ik)
                dv_target = -((plant.up1 + plant.up2) - (nom.up1 + nom.up2))
                # Use fixed-RC nominal states paired with physical ohmic impedance plant.r0 (causal training proxy)
                feat = build_feature_vector(ik, temp_c, nom.soc, nom.up1, nom.up2, plant.r0, chemistry=chemistry)
                X_list.append(feat)
                Y_list.append(dv_target)
                I_list.append(ik)

    X_t = torch.tensor(np.array(X_list, dtype=np.float32))
    Y_t = torch.tensor(np.array(Y_list, dtype=np.float32))
    I_t = torch.tensor(np.array(I_list, dtype=np.float32))

    if unconstrained_blackbox:
        model = ResidualCompensatorMLP(kappa_0=10.0, kappa_1=0.0, i_max=100.0)
    else:
        model = ResidualCompensatorMLP(kappa_0=0.015, kappa_1=0.022, i_max=6.0)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=lambda_reg)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        if unconstrained_blackbox:
            pred = model.raw_output(X_t)
            loss = torch.mean((pred - Y_t) ** 2)
        else:
            raw = model.raw_output(X_t)
            vmax = model.kappa_0 + model.kappa_1 * torch.clamp(torch.abs(I_t), max=model.i_max)
            pred = torch.clamp(raw, -vmax, vmax)
            l_res = torch.mean((pred - Y_t) ** 2)
            l_env = torch.relu(torch.abs(raw) - vmax) ** 2
            l_pass = gamma_i * (torch.relu(I_t * pred) ** 2)
            l_phys = torch.mean(l_env + l_pass)
            loss = l_res + lambda_phys * l_phys
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred_eval = model(X_t, I_t, hard_passivity=not unconstrained_blackbox)
        res_err = (pred_eval - Y_t).numpy()
        sigma_nn = float(np.std(res_err))
        eps_emp = float(np.percentile(np.abs(res_err), 99.5))

    stats = {"sigma_nn_mv": sigma_nn * 1000.0, "eps_995_mv": eps_emp * 1000.0}
    return model, stats


class PhysicsInformedDualKalmanFilter:
    """
    Hybrid PIDKF unifying DEKF state/parameter tracking with Butler-Volmer neural overpotential
    compensation across both NMC (3-state) and LFP (4-state with hysteresis).
    Strictly fair 100% SOH nominal initialization (zero oracle prior leakage).
    """

    def __init__(
        self,
        nominal_cell,
        nn_model=None,
        use_hysteresis=True,
        exact_jacobian=False,
        hard_passivity=True,
        sigma_nn=0.008,
        sigma_approx=0.002,
        p0_theta=None,
        q_theta=None,
    ):
        self.cell = nominal_cell
        self.chemistry = getattr(nominal_cell, "chemistry", "NMC")
        self.nn_model = nn_model
        self.exact_jacobian = bool(exact_jacobian)
        self.hard_passivity = bool(hard_passivity)
        self.r_x_eff = 1e-3 + (sigma_nn ** 2) + (sigma_approx ** 2)
        self.r_th_eff = 1e-2 + (sigma_nn ** 2)

        if p0_theta is None:
            p0_theta = np.diag([1e-3, 1e-4])
        if q_theta is None:
            q_theta = np.diag([1e-7, 1e-9])

        self.dekf = DualExtendedKalmanFilter(
            nominal_cell,
            use_hysteresis=use_hysteresis,
            p0_theta=p0_theta,
            q_theta=q_theta,
        )
        self.rel_jac_diffs = []

    def reset(self, initial_soc=1.0, p_diag=None):
        self.dekf.reset(initial_soc=initial_soc, p_diag=p_diag)
        self.rel_jac_diffs = []

    def step(self, current_a, v_measured, temp_c=25.0):
        _, x_pred, r0_curr, _ = self.dekf.predict_state_prior(current_a)
        dv_comp = 0.0
        c_extra = None

        if self.nn_model is not None:
            feat_vals = build_feature_vector(
                current_a, temp_c, x_pred[0, 0], x_pred[1, 0], x_pred[2, 0], r0_curr, chemistry=self.chemistry
            )
            if self.exact_jacobian:
                x_in = torch.tensor([feat_vals], dtype=torch.float32, requires_grad=True)
                i_t = torch.tensor([current_a], dtype=torch.float32)
                dv_t = self.nn_model(x_in, i_t, hard_passivity=self.hard_passivity)
                dv_comp = float(dv_t.item())
                grad_f = torch.autograd.grad(dv_t.sum(), x_in)[0][0].detach().numpy()
                # Chain rule back to state vector [z, U1, U2 (, Vh)]:
                # feat[2] = z, feat[3] = 15*U1, feat[4] = 15*U2
                d_z = float(grad_f[2])
                d_u1 = float(grad_f[3] * 15.0)
                d_u2 = float(grad_f[4] * 15.0)
                c_extra = [d_z, d_u1, d_u2, 0.0] if self.dekf.n == 4 else [d_z, d_u1, d_u2]
                docv = self.cell.get_docv_dsoc(x_pred[0, 0])
                h_base = np.array([docv, -1.0, -1.0, -1.0] if self.dekf.n == 4 else [docv, -1.0, -1.0])
                self.rel_jac_diffs.append(float(np.linalg.norm(c_extra) / max(np.linalg.norm(h_base), 1e-9)))
            else:
                x_in = torch.tensor([feat_vals], dtype=torch.float32)
                i_t = torch.tensor([current_a], dtype=torch.float32)
                with torch.no_grad():
                    dv_comp = float(self.nn_model(x_in, i_t, hard_passivity=self.hard_passivity).item())

        return self.dekf.step(
            current_a,
            v_measured,
            delta_v_comp=dv_comp,
            r_x_eff=self.r_x_eff,
            r_theta_eff=self.r_th_eff,
            c_extra=c_extra,
        )


if __name__ == "__main__":
    nmc_nn, st_n = train_residual_compensator("NMC", seed=42)
    n_p = sum(p.numel() for p in nmc_nn.parameters())
    print(f"PIDKF MLP params: {n_p} ({n_p*4/1024:.2f} KB), NMC train stats: {st_n}")
    lfp_nn, st_l = train_residual_compensator("LFP", seed=42)
    print(f"LFP train stats: {st_l}")
