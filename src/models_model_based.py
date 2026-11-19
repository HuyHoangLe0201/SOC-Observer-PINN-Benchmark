"""
Model-Based Observers for Battery State-of-Charge (SOC) Estimation
Journal of Energy Storage (Elsevier)

Exact Python reference implementations synchronized with Sections 3.1-3.4, Appendix A, and Appendix C.2:
1. ExtendedKalmanFilter (EKF) - 3-state NMC / 4-state LFP with Joseph covariance update
2. AdaptiveEKF (AEKF) - Sage-Husa innovation variance estimator with Joseph form
3. UnscentedKalmanFilter (UKF) - Scaled unscented transform (7 / 9 sigma points)
4. DualExtendedKalmanFilter (DEKF) - Joint state & parameter [R0, 1/Q_act] tracking with rest gating
5. SlidingModeObserver (SMO) - Discrete boundary-layer sliding mode observer Eq. (34)
"""

import numpy as np


def _is_lfp(cell):
    return getattr(cell, "chemistry", "NMC") == "LFP"


def _step_hysteresis(cell, v_h, current_a, q_cap):
    """Exact ZOH discrete hysteresis Eq. (18): V_{h,k+1} = a_h V_{h,k} + (1 - a_h) M_0 sgn(I_k)."""
    if abs(current_a) <= 1e-6:
        return float(v_h), 1.0
    eta = cell.get_eta(current_a)
    abs_dz = abs(eta * current_a * cell.dt) / (3600.0 * q_cap)
    a_h = float(np.exp(-cell.beta_hyst * abs_dz))
    sgn_i = 1.0 if current_a > 0.0 else -1.0
    v_h_next = a_h * v_h + (1.0 - a_h) * cell.m0_hyst * sgn_i
    return float(v_h_next), a_h


class ExtendedKalmanFilter:
    """
    Standard Extended Kalman Filter (EKF) for 2-RC (3-state NMC or 4-state LFP) model.
    State vector: [SOC, U1, U2]^T (NMC) or [SOC, U1, U2, V_h]^T (LFP when use_hysteresis=True).
    """

    def __init__(self, battery_cell, q_cov=None, r_cov=1e-3, p0_cov=None, use_hysteresis=True, rest_gate_a=0.010):
        self.cell = battery_cell
        self.dt = battery_cell.dt
        self.use_hyst = bool(use_hysteresis and _is_lfp(battery_cell))
        self.n = 4 if self.use_hyst else 3
        self.rest_gate_a = float(rest_gate_a)

        self.x = np.zeros((self.n, 1))
        self.x[0, 0] = 1.0

        default_p0 = [0.01, 1e-4, 1e-4, 1e-4] if self.n == 4 else [0.01, 1e-4, 1e-4]
        default_q = [1e-6, 1e-5, 1e-5, 1e-6] if self.n == 4 else [1e-6, 1e-5, 1e-5]

        self.P = np.diag(default_p0) if p0_cov is None else np.array(p0_cov, dtype=float)
        self.Q = np.diag(default_q) if q_cov is None else np.array(q_cov, dtype=float)
        self.R = float(r_cov)

    def reset(self, initial_soc=1.0, p_diag=None):
        self.x = np.zeros((self.n, 1))
        self.x[0, 0] = float(initial_soc)
        if p_diag is None:
            p_diag = [0.01, 1e-4, 1e-4, 1e-4] if self.n == 4 else [0.01, 1e-4, 1e-4]
        elif len(p_diag) == 3 and self.n == 4:
            p_diag = list(p_diag) + [1e-4]
        self.P = np.diag(p_diag)

    def step(self, current_a, v_measured):
        # Strictly nominal baseline parameters (no oracle temperature/aging leakage)
        tau1 = self.cell.r1_base * self.cell.c1_base
        tau2 = self.cell.r2_base * self.cell.c2_base
        exp1 = float(np.exp(-self.dt / tau1))
        exp2 = float(np.exp(-self.dt / tau2))
        eta = self.cell.get_eta(current_a)

        if self.use_hyst:
            v_h_next, a_h = _step_hysteresis(self.cell, self.x[3, 0], current_a, self.cell.q_nominal)
            A = np.diag([1.0, exp1, exp2, a_h])
            B = np.array([
                [-eta * self.dt / (3600.0 * self.cell.q_nominal)],
                [self.cell.r1_base * (1.0 - exp1)],
                [self.cell.r2_base * (1.0 - exp2)],
                [0.0],
            ])
            x_pred = A @ self.x + B * current_a
            x_pred[3, 0] = v_h_next
        else:
            A = np.diag([1.0, exp1, exp2])
            B = np.array([
                [-eta * self.dt / (3600.0 * self.cell.q_nominal)],
                [self.cell.r1_base * (1.0 - exp1)],
                [self.cell.r2_base * (1.0 - exp2)],
            ])
            x_pred = A @ self.x + B * current_a

        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)
        P_pred = A @ self.P @ A.T + self.Q

        soc_pred = x_pred[0, 0]
        ocv_pred = self.cell.get_ocv(soc_pred)
        v_pred = ocv_pred - current_a * self.cell.r0_base - x_pred[1, 0] - x_pred[2, 0]
        if self.use_hyst:
            v_pred -= x_pred[3, 0]

        d_ocv = self.cell.get_docv_dsoc(soc_pred)
        if self.use_hyst:
            C = np.array([[d_ocv, -1.0, -1.0, -1.0]])
        else:
            C = np.array([[d_ocv, -1.0, -1.0]])

        y_res = v_measured - v_pred
        if abs(current_a) < self.rest_gate_a:
            K = np.zeros((self.n, 1))
        else:
            S = float((C @ P_pred @ C.T)[0, 0] + self.R)
            K = (P_pred @ C.T) / S

        self.x = x_pred + K * y_res
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        I_mat = np.eye(self.n)
        IKC = I_mat - K @ C
        self.P = IKC @ P_pred @ IKC.T + self.R * (K @ K.T)

        return float(self.x[0, 0]), float(v_pred), float(self.P[0, 0])


class AdaptiveEKF(ExtendedKalmanFilter):
    """
    Sage-Husa Adaptive Extended Kalman Filter (AEKF), Eqs. (28)-(29).
    Uses exponential weighting sequence d_k = (1 - lambda) / (1 - lambda^k),
    lambda_SH = 0.985, R_{v,min} = 1e-4, and Joseph covariance update.
    """

    def __init__(self, battery_cell, forgetting_factor=0.985, r_min=1e-4, r_max=0.05, **kwargs):
        super().__init__(battery_cell, **kwargs)
        self.lambda_sh = float(forgetting_factor)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.r_hat = 0.0
        self.r_val = self.R
        self.k_step = 0

    def reset(self, initial_soc=1.0, p_diag=None):
        super().reset(initial_soc=initial_soc, p_diag=p_diag)
        self.r_hat = 0.0
        self.r_val = self.R
        self.k_step = 0

    def step(self, current_a, v_measured):
        self.k_step += 1
        d_k = (1.0 - self.lambda_sh) / (1.0 - self.lambda_sh ** self.k_step)

        tau1 = self.cell.r1_base * self.cell.c1_base
        tau2 = self.cell.r2_base * self.cell.c2_base
        exp1 = float(np.exp(-self.dt / tau1))
        exp2 = float(np.exp(-self.dt / tau2))
        eta = self.cell.get_eta(current_a)

        if self.use_hyst:
            v_h_next, a_h = _step_hysteresis(self.cell, self.x[3, 0], current_a, self.cell.q_nominal)
            A = np.diag([1.0, exp1, exp2, a_h])
            B = np.array([
                [-eta * self.dt / (3600.0 * self.cell.q_nominal)],
                [self.cell.r1_base * (1.0 - exp1)],
                [self.cell.r2_base * (1.0 - exp2)],
                [0.0],
            ])
            x_pred = A @ self.x + B * current_a
            x_pred[3, 0] = v_h_next
        else:
            A = np.diag([1.0, exp1, exp2])
            B = np.array([
                [-eta * self.dt / (3600.0 * self.cell.q_nominal)],
                [self.cell.r1_base * (1.0 - exp1)],
                [self.cell.r2_base * (1.0 - exp2)],
            ])
            x_pred = A @ self.x + B * current_a

        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)
        P_pred = A @ self.P @ A.T + self.Q

        soc_pred = x_pred[0, 0]
        v_pred = self.cell.get_ocv(soc_pred) - current_a * self.cell.r0_base - x_pred[1, 0] - x_pred[2, 0]
        if self.use_hyst:
            v_pred -= x_pred[3, 0]

        d_ocv = self.cell.get_docv_dsoc(soc_pred)
        C = np.array([[d_ocv, -1.0, -1.0, -1.0]]) if self.use_hyst else np.array([[d_ocv, -1.0, -1.0]])

        e_k = v_measured - v_pred
        self.r_hat = (1.0 - d_k) * self.r_hat + d_k * e_k
        e_centered = e_k - self.r_hat
        r_inst = (e_centered ** 2) - float((C @ P_pred @ C.T)[0, 0])
        self.r_val = float(np.clip((1.0 - d_k) * self.r_val + d_k * r_inst, self.r_min, self.r_max))

        if abs(current_a) < self.rest_gate_a:
            K = np.zeros((self.n, 1))
        else:
            S = float((C @ P_pred @ C.T)[0, 0] + self.r_val)
            K = (P_pred @ C.T) / S

        self.x = x_pred + K * e_k
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        I_mat = np.eye(self.n)
        IKC = I_mat - K @ C
        self.P = IKC @ P_pred @ IKC.T + self.r_val * (K @ K.T)

        return float(self.x[0, 0]), float(v_pred), float(self.P[0, 0])


class UnscentedKalmanFilter:
    """
    Scaled Unscented Kalman Filter (UKF) with alpha_UT = 1e-3, beta_UT = 2, kappa_UT = 0 (Appendix C.2).
    """

    def __init__(self, battery_cell, alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0, q_cov=None, r_cov=1e-3, use_hysteresis=True):
        self.cell = battery_cell
        self.dt = battery_cell.dt
        self.use_hyst = bool(use_hysteresis and _is_lfp(battery_cell))
        self.n = 4 if self.use_hyst else 3

        self.alpha_ut = float(alpha_ut)
        self.beta_ut = float(beta_ut)
        self.kappa_ut = float(kappa_ut)
        lam = (self.alpha_ut ** 2) * (self.n + self.kappa_ut) - self.n
        self.gamma_ut = np.sqrt(self.n + lam)

        self.wm = np.full(2 * self.n + 1, 0.5 / (self.n + lam))
        self.wc = np.full(2 * self.n + 1, 0.5 / (self.n + lam))
        self.wm[0] = lam / (self.n + lam)
        self.wc[0] = self.wm[0] + (1.0 - self.alpha_ut ** 2 + self.beta_ut)

        default_p0 = [0.01, 1e-4, 1e-4, 1e-4] if self.n == 4 else [0.01, 1e-4, 1e-4]
        default_q = [1e-6, 1e-5, 1e-5, 1e-6] if self.n == 4 else [1e-6, 1e-5, 1e-5]
        self.P = np.diag(default_p0)
        self.Q = np.diag(default_q) if q_cov is None else np.array(q_cov, dtype=float)
        self.R = float(r_cov)

        self.x = np.zeros(self.n)
        self.x[0] = 1.0

    def reset(self, initial_soc=1.0, p_diag=None):
        self.x = np.zeros(self.n)
        self.x[0] = float(initial_soc)
        if p_diag is None:
            p_diag = [0.01, 1e-4, 1e-4, 1e-4] if self.n == 4 else [0.01, 1e-4, 1e-4]
        elif len(p_diag) == 3 and self.n == 4:
            p_diag = list(p_diag) + [1e-4]
        self.P = np.diag(p_diag)

    def step(self, current_a, v_measured):
        # Generate 2n + 1 sigma points
        P_sym = 0.5 * (self.P + self.P.T) + 1e-12 * np.eye(self.n)
        L = np.linalg.cholesky(P_sym)
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = self.x
        for i in range(self.n):
            col = self.gamma_ut * L[:, i]
            sigmas[i + 1] = self.x + col
            sigmas[self.n + i + 1] = self.x - col

        tau1 = self.cell.r1_base * self.cell.c1_base
        tau2 = self.cell.r2_base * self.cell.c2_base
        exp1 = float(np.exp(-self.dt / tau1))
        exp2 = float(np.exp(-self.dt / tau2))
        eta = self.cell.get_eta(current_a)
        dsoc_step = (-eta * current_a * self.dt) / (3600.0 * self.cell.q_nominal)

        sig_pred = np.zeros_like(sigmas)
        v_sigma = np.zeros(2 * self.n + 1)
        for m in range(2 * self.n + 1):
            sp = sigmas[m].copy()
            sp[0] = np.clip(sp[0] + dsoc_step, 0.0, 1.0)
            sp[1] = sp[1] * exp1 + current_a * self.cell.r1_base * (1.0 - exp1)
            sp[2] = sp[2] * exp2 + current_a * self.cell.r2_base * (1.0 - exp2)
            if self.use_hyst:
                sp[3], _ = _step_hysteresis(self.cell, sp[3], current_a, self.cell.q_nominal)
            sig_pred[m] = sp
            vp = self.cell.get_ocv(sp[0]) - current_a * self.cell.r0_base - sp[1] - sp[2]
            if self.use_hyst:
                vp -= sp[3]
            v_sigma[m] = vp

        x_pred = np.sum(self.wm[:, None] * sig_pred, axis=0)
        x_pred[0] = np.clip(x_pred[0], 0.0, 1.0)
        v_pred = float(np.sum(self.wm * v_sigma))

        P_pred = self.Q.copy()
        P_xy = np.zeros((self.n, 1))
        S_v = self.R
        for m in range(2 * self.n + 1):
            dx = (sig_pred[m] - x_pred).reshape(-1, 1)
            dv = float(v_sigma[m] - v_pred)
            P_pred += self.wc[m] * (dx @ dx.T)
            S_v += self.wc[m] * (dv ** 2)
            P_xy += self.wc[m] * dx * dv

        K = P_xy / max(S_v, 1e-12)
        self.x = x_pred + (K[:, 0] * (v_measured - v_pred))
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = P_pred - S_v * (K @ K.T)
        self.P = 0.5 * (self.P + self.P.T)
        return float(self.x[0]), float(v_pred), float(self.P[0, 0])


class DualExtendedKalmanFilter:
    """
    Dual Extended Kalman Filter (DEKF) for Joint State [SOC, U1, U2 (, V_h)]^T
    and Parameter [R_0, 1/Q_act]^T tracking (Section 3.3, Appendix C.2).

    Features:
    - Strictly identical fresh/nominal 100% SOH initialization (zero oracle leakage).
    - Stabilized Joseph covariance updates for both state and parameter filters.
    - Hard rest-period innovation gating (|I_k| < 10 mA) freezing K_k and K_{theta,k}.
    - Optional residual voltage compensation delta_v_comp for PIDKF coupling.
    """

    def __init__(
        self,
        battery_cell,
        use_hysteresis=True,
        p0_x=None,
        q_x=None,
        r_x=1e-3,
        p0_theta=None,
        q_theta=None,
        r_theta=1e-2,
        rest_gate_a=0.010,
    ):
        self.cell = battery_cell
        self.dt = battery_cell.dt
        self.use_hyst = bool(use_hysteresis and _is_lfp(battery_cell))
        self.n = 4 if self.use_hyst else 3
        self.rest_gate_a = float(rest_gate_a)

        self.x = np.zeros((self.n, 1))
        self.x[0, 0] = 1.0
        default_p0x = [0.01, 1e-4, 1e-4, 1e-4] if self.n == 4 else [0.01, 1e-4, 1e-4]
        default_qx = [1e-6, 1e-5, 1e-5, 1e-6] if self.n == 4 else [1e-6, 1e-5, 1e-5]
        self.p0_x_default = default_p0x
        self.P_x = np.diag(default_p0x) if p0_x is None else np.array(p0_x, dtype=float)
        self.Q_x = np.diag(default_qx) if q_x is None else np.array(q_x, dtype=float)
        self.R_x = float(r_x)

        # Parameter vector theta = [R_0 (Ohm), 1 / Q_act (Ah^-1)]^T matching paper notation
        self.theta = np.array([[battery_cell.r0_base], [1.0 / battery_cell.q_nominal]])
        self.P_theta_default = np.diag([1e-3, 1e-2]) if p0_theta is None else np.array(p0_theta, dtype=float)
        self.P_theta = self.P_theta_default.copy()
        self.Q_theta = np.diag([1e-7, 1e-6]) if q_theta is None else np.array(q_theta, dtype=float)
        self.R_theta = float(r_theta)
        self.dx_plus_dtheta = np.zeros((self.n, 2))
        self.i_prev = 0.0

    def reset(self, initial_soc=1.0, p_diag=None):
        self.x = np.zeros((self.n, 1))
        self.x[0, 0] = float(initial_soc)
        if p_diag is None:
            self.P_x = np.diag(self.p0_x_default)
        else:
            pd_list = list(p_diag) + [1e-4] if (len(p_diag) == 3 and self.n == 4) else list(p_diag)
            self.P_x = np.diag(pd_list)
        self.theta = np.array([[self.cell.r0_base], [1.0 / self.cell.q_nominal]])
        self.P_theta = self.P_theta_default.copy()
        self.dx_plus_dtheta = np.zeros((self.n, 2))
        self.i_prev = 0.0

    def predict_state_prior(self, current_a):
        """Evaluate a priori state x_k^- and predicted ECM terminal voltage V_{t,ECM,k}^- without updating filter."""
        r0_curr = float(self.theta[0, 0])
        q_act_curr = 1.0 / float(self.theta[1, 0])
        tau1 = self.cell.r1_base * self.cell.c1_base
        tau2 = self.cell.r2_base * self.cell.c2_base
        exp1 = float(np.exp(-self.dt / tau1))
        exp2 = float(np.exp(-self.dt / tau2))
        eta = self.cell.get_eta(current_a)

        if self.use_hyst:
            v_h_next, a_h = _step_hysteresis(self.cell, self.x[3, 0], current_a, q_act_curr)
            A = np.diag([1.0, exp1, exp2, a_h])
            B = np.array([
                [-eta * self.dt / (3600.0 * q_act_curr)],
                [self.cell.r1_base * (1.0 - exp1)],
                [self.cell.r2_base * (1.0 - exp2)],
                [0.0],
            ])
            x_pred = A @ self.x + B * current_a
            x_pred[3, 0] = v_h_next
        else:
            A = np.diag([1.0, exp1, exp2])
            B = np.array([
                [-eta * self.dt / (3600.0 * q_act_curr)],
                [self.cell.r1_base * (1.0 - exp1)],
                [self.cell.r2_base * (1.0 - exp2)],
            ])
            x_pred = A @ self.x + B * current_a

        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)
        return A, x_pred, r0_curr, q_act_curr

    def step(self, current_a, v_measured, delta_v_comp=0.0, r_x_eff=None, r_theta_eff=None, c_extra=None):
        """
        Execute one joint State + Parameter DEKF cycle matching Section 3.3, Section 3.7, and dekf.c.
        Optional delta_v_comp and effective variances (R_eff,k, R_{theta,eff}) enable PIDKF coupling.
        """
        A, x_pred, r0_curr, q_act_curr = self.predict_state_prior(current_a)
        P_x_pred = A @ self.P_x @ A.T + self.Q_x

        soc_pred = x_pred[0, 0]
        v_pred_ecm = self.cell.get_ocv(soc_pred) - current_a * r0_curr - x_pred[1, 0] - x_pred[2, 0]
        if self.use_hyst:
            v_pred_ecm -= x_pred[3, 0]
        v_pred_total = v_pred_ecm + float(delta_v_comp)

        d_ocv_pred = self.cell.get_docv_dsoc(soc_pred)
        C_x = np.array([[d_ocv_pred, -1.0, -1.0, -1.0]]) if self.use_hyst else np.array([[d_ocv_pred, -1.0, -1.0]])
        if c_extra is not None:
            C_x = C_x + np.asarray(c_extra, dtype=float).reshape(1, self.n)

        rx = self.R_x if r_x_eff is None else float(r_x_eff)
        rth = self.R_theta if r_theta_eff is None else float(r_theta_eff)
        e_k = v_measured - v_pred_total
        is_rest = abs(current_a) < self.rest_gate_a

        if is_rest:
            K_x = np.zeros((self.n, 1))
        else:
            S_x = float((C_x @ P_x_pred @ C_x.T)[0, 0] + rx)
            K_x = (P_x_pred @ C_x.T) / S_x

        self.x = x_pred + K_x * e_k
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        I_n = np.eye(self.n)
        IKC_x = I_n - K_x @ C_x
        self.P_x = IKC_x @ P_x_pred @ IKC_x.T + rx * (K_x @ K_x.T)

        # --- Parameter Filter Update (Plett 2004 Recurrent Total Derivative Sensitivity) ---
        P_theta_pred = self.P_theta + self.Q_theta
        eta_curr = self.cell.get_eta(current_a)
        df_dtheta = np.zeros((self.n, 2))
        df_dtheta[0, 1] = -eta_curr * self.dt * current_a / 3600.0
        dx_minus_dtheta = A @ self.dx_plus_dtheta + df_dtheta

        dg_dtheta = np.zeros((1, 2))
        dg_dtheta[0, 0] = -current_a
        C_theta = dg_dtheta + C_x @ dx_minus_dtheta

        self.dx_plus_dtheta = dx_minus_dtheta - K_x @ C_theta

        # Innovation e_k evaluated at a priori predicted terminal voltage matches total derivative linearization
        if is_rest:
            K_theta = np.zeros((2, 1))
        else:
            S_theta = float((C_theta @ P_theta_pred @ C_theta.T)[0, 0] + rth)
            K_theta = (P_theta_pred @ C_theta.T) / S_theta

        self.theta = self.theta + K_theta * e_k
        # Physical boundaries anchored to initial capacity Q_init
        self.theta[0, 0] = float(np.clip(self.theta[0, 0], 0.005, 0.250))
        q_inv_min = 1.0 / (1.5 * self.cell.q_nominal)
        q_inv_max = 1.0 / (0.5 * self.cell.q_nominal)
        self.theta[1, 0] = float(np.clip(self.theta[1, 0], q_inv_min, q_inv_max))

        I_2 = np.eye(2)
        IKC_t = I_2 - K_theta @ C_theta
        self.P_theta = IKC_t @ P_theta_pred @ IKC_t.T + rth * (K_theta @ K_theta.T)

        self.i_prev = float(current_a)
        cap_est = 1.0 / self.theta[1, 0]
        return float(self.x[0, 0]), float(v_pred_total), float(self.theta[0, 0]), float(cap_est)


class SlidingModeObserver:
    """
    Discrete Boundary-Layer Sliding Mode Observer (SMO), Eq. (34).
    Parameters synchronized with Appendix C.2: l_smo = 0.025 V, mu = 0.020 V.
    Supports both 3-state NMC (alpha_min = 0.40 V) and 4-state LFP (alpha_min = 0.065 V).
    """

    def __init__(self, battery_cell, l_gain=0.025, boundary_layer=0.020, use_hysteresis=True):
        self.cell = battery_cell
        self.dt = battery_cell.dt
        self.l_gain = float(l_gain)
        self.mu = float(boundary_layer)
        self.use_hyst = bool(use_hysteresis and _is_lfp(battery_cell))
        self.alpha_min = 0.065 if _is_lfp(battery_cell) else 0.400

        self.soc = 1.0
        self.up1 = 0.0
        self.up2 = 0.0
        self.v_h = 0.0

    def reset(self, initial_soc=1.0):
        self.soc = float(initial_soc)
        self.up1 = 0.0
        self.up2 = 0.0
        self.v_h = 0.0

    def step(self, current_a, v_measured):
        # Current predicted voltage V_hat_{t,k} from current state x_hat_k (consistent with S_k = V_meas,k - V_hat_{t,k})
        v_model = self.cell.get_ocv(self.soc) - current_a * self.cell.r0_base - self.up1 - self.up2
        if self.use_hyst:
            v_model -= self.v_h
        s_k = v_measured - v_model

        # Scheduled SOC-only sliding injection L(z_hat_k) * sat(S_k / mu)
        d_ocv = max(self.alpha_min, float(self.cell.get_docv_dsoc(self.soc)))
        sat_val = float(np.clip(s_k / self.mu, -1.0, 1.0))
        l_inj = (self.l_gain / d_ocv) * sat_val

        # Discrete state update Eq. (34): x_hat_{k+1} = A_k x_hat_k + B_k I_k + g_k + L(z_k) sat(S_k / mu)
        eta = self.cell.get_eta(current_a)
        tau1 = self.cell.r1_base * self.cell.c1_base
        tau2 = self.cell.r2_base * self.cell.c2_base
        exp1 = float(np.exp(-self.dt / tau1))
        exp2 = float(np.exp(-self.dt / tau2))

        self.soc = float(np.clip(self.soc - (eta * current_a * self.dt) / (3600.0 * self.cell.q_nominal) + l_inj, 0.0, 1.0))
        self.up1 = self.up1 * exp1 + current_a * self.cell.r1_base * (1.0 - exp1)
        self.up2 = self.up2 * exp2 + current_a * self.cell.r2_base * (1.0 - exp2)
        if self.use_hyst:
            self.v_h, _ = _step_hysteresis(self.cell, self.v_h, current_a, self.cell.q_nominal)

        return float(self.soc), float(v_model)
