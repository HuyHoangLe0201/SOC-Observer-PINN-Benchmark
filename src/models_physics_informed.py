"""
Physics-Informed Neural Network (PINN-SOC) for Battery State-of-Charge Estimation
Journal of Energy Storage (Elsevier)

Key Features:
- Multi-objective Loss Function embedding:
  1. Data Supervised Loss (L_data)
  2. Faraday's Conservation Law / Coulomb Counting Gradient (L_faraday)
  3. Kirchhoff's Voltage Law & OCV-Consistency Residual (L_kvl)
- High generalization ability under battery aging (SOH shift) and temperature changes.
"""

import torch
import torch.nn as nn
import numpy as np


class PINNSOCModel(nn.Module):
    """
    Physics-Informed Neural Network for Battery SOC Estimation.
    Outputs:
        soc_pred: Predicted State of Charge [0, 1]
        v_polarization: Learned internal dynamic polarization voltage
    """

    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        # Backbone feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Smooth activation essential for gradient-based physics derivatives
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.SiLU()
        )

        # SOC prediction head (bounded in [0, 1])
        self.soc_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Auxiliary physical polarization voltage head (in Volts)
        self.v_pol_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()  # Allows small positive or negative polarization tracking
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        features = self.encoder(x)
        soc_est = self.soc_head(features)
        v_pol = self.v_pol_head(features) * 0.15  # Scale polarization to physical range +/-150mV
        return soc_est.squeeze(-1), v_pol.squeeze(-1)


class PhysicsInformedLoss(nn.Module):
    """
    Composite Physics-Informed Loss Function:
    L = L_data + lambda_faraday * L_faraday + lambda_kvl * L_kvl
    """

    def __init__(
        self,
        q_nominal_ah=2.0,
        r0_nominal=0.035,
        dt=1.0,
        lambda_faraday=0.15,
        lambda_kvl=0.10,
        ocv_coeffs=None
    ):
        super().__init__()
        self.q_nominal = float(q_nominal_ah)
        self.r0 = float(r0_nominal)
        self.dt = float(dt)
        self.lambda_faraday = float(lambda_faraday)
        self.lambda_kvl = float(lambda_kvl)
        self.mse_loss = nn.MSELoss()

        if ocv_coeffs is None:
            # Calibrated monotonic NMC 5th order polynomial coefficients
            self.ocv_coeffs = torch.tensor([
                3.21181343, 4.67403064, -17.41856030, 33.37645360, -28.84109993, 9.20379162
            ], dtype=torch.float32)
        else:
            self.ocv_coeffs = torch.tensor(ocv_coeffs, dtype=torch.float32)

    def evaluate_ocv_torch(self, soc_tensor):
        """Differentiable OCV evaluation in PyTorch."""
        soc_clamped = torch.clamp(soc_tensor, 0.0, 1.0)
        ocv = torch.zeros_like(soc_clamped)
        device = soc_tensor.device
        coeffs = self.ocv_coeffs.to(device)

        for i, c in enumerate(coeffs):
            ocv = ocv + c * (soc_clamped ** i)
        return ocv

    def forward(self, soc_pred, v_pol_pred, soc_true, current_seq, v_measured_seq):
        """
        Compute total multi-physics loss.
        """
        # 1. Supervised Data Loss
        loss_data = self.mse_loss(soc_pred, soc_true)

        # 2. Faraday's Conservation / Coulomb-Counting Gradient Loss
        # delta_SOC / delta_t must approximate -I / (3600 * Q_nom)
        if soc_pred.shape[0] > 1:
            d_soc_pred = (soc_pred[1:] - soc_pred[:-1]) / self.dt
            current_mid = current_seq[1:]
            d_soc_faraday = -current_mid / (3600.0 * self.q_nominal)
            loss_faraday = torch.mean((d_soc_pred - d_soc_faraday) ** 2)
        else:
            loss_faraday = torch.tensor(0.0, device=soc_pred.device)

        # 3. Kirchhoff's Voltage Law (KVL) Consistency Loss
        # V_terminal = OCV(SOC) - I * R0 - V_polarization
        ocv_pred = self.evaluate_ocv_torch(soc_pred)
        v_model_pred = ocv_pred - current_seq * self.r0 - v_pol_pred
        loss_kvl = self.mse_loss(v_model_pred, v_measured_seq)

        # Total combined loss
        total_loss = loss_data + self.lambda_faraday * loss_faraday + self.lambda_kvl * loss_kvl

        return total_loss, {
            "loss_data": loss_data.item(),
            "loss_faraday": loss_faraday.item() if isinstance(loss_faraday, torch.Tensor) else 0.0,
            "loss_kvl": loss_kvl.item()
        }
