"""
Physics-Informed Neural Network (PINN-SOC) for Battery State-of-Charge Estimation
Journal of Energy Storage (Elsevier)

Exact architecture synchronized with Section 3.6 & Appendix C.2:
- 4-layer [64, 64, 32] GELU MLP operating strictly causally on shift-invariant telemetry
  u_k = [I_k, V_{t,k}, T_k, dV_{t,k}/dt, SOH_{latched,k}]^T (5 inputs) -> dual heads (SOC_hat, V_pol_hat).
- Enforces:
  1. Supervised data fidelity L_data
  2. Differential Faradaic charge conservation residual L_Faraday
  3. Kirchhoff's Voltage Law residual L_KVL
  4. Thermodynamic monotonicity L_mono via PyTorch autograd (d z_hat / d V_t >= 0)
- Exact parameter count: 6,690 float32 parameters (26.13 KB Flash).
"""

import torch
import torch.nn as nn


class PINNSOCModel(nn.Module):
    """
    Causal shift-invariant 4-layer [64, 64, 32] GELU MLP for SOC and polarization voltage prediction.
    """

    def __init__(self, input_dim=5, h1=64, h2=64, h3=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, h3),
            nn.GELU(),
        )
        self.soc_head = nn.Sequential(nn.Linear(h3, 1), nn.Sigmoid())
        self.v_pol_head = nn.Sequential(nn.Linear(h3, 1), nn.Tanh())

    def forward(self, x):
        feat = self.encoder(x)
        soc_est = self.soc_head(feat).squeeze(-1)
        v_pol = (self.v_pol_head(feat) * 0.25).squeeze(-1)  # Bounded polarization overpotential +/-250 mV
        return soc_est, v_pol


class PhysicsInformedLoss(nn.Module):
    """
    Composite Physics-Informed Objective Eq. (43):
    L_total = L_data + lambda_faraday * L_Faraday + lambda_kvl * L_KVL + lambda_mono * L_mono
    """

    def __init__(
        self,
        q_nominal_ah=2.00,
        r0_nominal=0.035,
        dt=1.0,
        lambda_faraday=0.40,
        lambda_kvl=0.30,
        lambda_mono=0.10,
        chemistry="NMC",
    ):
        super().__init__()
        self.q_nominal = float(q_nominal_ah)
        self.r0 = float(r0_nominal)
        self.dt = float(dt)
        self.lambda_faraday = float(lambda_faraday)
        self.lambda_kvl = float(lambda_kvl)
        self.lambda_mono = float(lambda_mono)
        self.chemistry = chemistry
        self.mse_loss = nn.MSELoss()
        # NMC 5th-order polynomial coefficients (z^0 -> z^5)
        self.register_buffer(
            "ocv_coeffs",
            torch.tensor([3.21181343, 4.67403064, -17.41856030, 33.37645360, -28.84109993, 9.20379162], dtype=torch.float32),
        )

    def evaluate_ocv_torch(self, soc_tensor):
        s = torch.clamp(soc_tensor, 0.0, 1.0)
        if self.chemistry == "LFP":
            sig_low = torch.sigmoid(26.0 * (s - 0.06))
            sig_high = torch.sigmoid(24.0 * (s - 0.89))
            return 1.950 + 1.320 * sig_low + 0.065 * s + 0.250 * sig_high
        ocv = torch.zeros_like(s)
        for i, c in enumerate(self.ocv_coeffs):
            ocv = ocv + c * (s ** i)
        return ocv

    def forward(self, model, x_seq, soc_true, current_seq, v_measured_seq):
        """
        Compute supervised + differential + KVL + autograd monotonicity loss on a trajectory window.
        """
        x_in = x_seq.clone().detach().requires_grad_(True)
        soc_pred, v_pol_pred = model(x_in)

        # 1. Data loss
        loss_data = self.mse_loss(soc_pred, soc_true)

        # 2. Faradaic conservation residual along consecutive discrete samples
        if soc_pred.shape[0] > 1:
            d_soc_dt = (soc_pred[1:] - soc_pred[:-1]) / self.dt
            q_eff = self.q_nominal * torch.clamp(x_in[1:, 4], min=0.5, max=1.2)
            d_soc_faraday = -current_seq[1:] / (3600.0 * q_eff)
            loss_faraday = torch.mean((d_soc_dt - d_soc_faraday) ** 2)
        else:
            loss_faraday = torch.tensor(0.0, device=x_in.device)

        # 3. KVL algebraic consistency
        ocv_pred = self.evaluate_ocv_torch(soc_pred)
        v_model = ocv_pred - current_seq * self.r0 - v_pol_pred
        loss_kvl = self.mse_loss(v_model, v_measured_seq)

        # 4. Monotonicity d(z_hat)/d(V_t) >= 0 via PyTorch autograd w.r.t. normalized voltage feature (index 1)
        grad_x = torch.autograd.grad(
            outputs=soc_pred.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
        )[0]
        dz_dv = grad_x[:, 1]
        loss_mono = torch.mean(torch.relu(-dz_dv) ** 2)

        total_loss = (
            loss_data
            + self.lambda_faraday * loss_faraday
            + self.lambda_kvl * loss_kvl
            + self.lambda_mono * loss_mono
        )
        return total_loss


if __name__ == "__main__":
    m = PINNSOCModel()
    n_p = sum(p.numel() for p in m.parameters())
    print(f"PINN-SOC params: {n_p} ({n_p * 4 / 1024:.2f} KB float32)")
