"""
Model-Based Observers for Battery State-of-Charge (SOC) Estimation
Journal of Energy Storage (Elsevier)

Algorithms Implemented:
1. Extended Kalman Filter (EKF) - Baseline observer
2. Adaptive Extended Kalman Filter (AEKF) - Innovation-based covariance adaptation
3. Dual Extended Kalman Filter (DEKF) - Joint State (SOC) and Parameter (R0, Q_act) Tracking
4. Robust Sliding Mode Observer (SMO) - Non-linear boundary layer tracking
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    Standard Extended Kalman Filter (EKF) for 2-RC Battery Equivalent Circuit Model.
    State: x = [SOC, Up1, Up2]^T
    """

    def __init__(self, battery_cell, q_cov=None, r_cov=0.001, p0_cov=None):
        self.cell = battery_cell
        self.dt = battery_cell.dt

        # Initial State: [SOC, Up1, Up2]
        self.x = np.array([[1.0], [0.0], [0.0]])

        # Covariance matrices
        if p0_cov is None:
            self.P = np.diag([0.01, 1e-4, 1e-4])
        else:
            self.P = np.array(p0_cov, dtype=float)

        if q_cov is None:
            self.Q = np.diag([1e-6, 1e-5, 1e-5])
        else:
            self.Q = np.array(q_cov, dtype=float)

        self.R = float(r_cov)  # Measurement noise covariance (voltage sensor variance)

    def reset(self, initial_soc=1.0, p_diag=(0.25, 1e-4, 1e-4)):
        self.x = np.array([[float(initial_soc)], [0.0], [0.0]])
        self.P = np.diag(p_diag)

    def step(self, current_a, v_measured):
        """
        Execute 1 EKF estimation step.
        Returns:
            soc_est: Estimated SOC (0.0 to 1.0)
            v_est: Predicted terminal voltage
            p_soc: Estimation variance of SOC
        """
        # 1. State Space System Matrices
        exp1 = np.exp(-self.dt / self.cell.tau1)
        exp2 = np.exp(-self.dt / self.cell.tau2)

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, exp1, 0.0],
            [0.0, 0.0, exp2]
        ])

        B = np.array([
            [-self.dt / (3600.0 * self.cell.q_nominal)],
            [self.cell.r1 * (1.0 - exp1)],
            [self.cell.r2 * (1.0 - exp2)]
        ])

        # 2. Time Update (Prediction)
        x_pred = A @ self.x + B * current_a
        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)
        P_pred = A @ self.P @ A.T + self.Q

        # 3. Measurement Prediction & Observation Matrix C
        soc_pred = x_pred[0, 0]
        up1_pred = x_pred[1, 0]
        up2_pred = x_pred[2, 0]

        ocv_pred = self.cell.get_ocv(soc_pred)
        v_pred = ocv_pred - current_a * self.cell.r0 - up1_pred - up2_pred

        # Jacobian C = [d(Vt)/d(SOC), d(Vt)/d(Up1), d(Vt)/d(Up2)]
        d_ocv = self.cell.get_docv_dsoc(soc_pred)
        C = np.array([[d_ocv, -1.0, -1.0]])

        # 4. Measurement Update (Correction)
        y_residual = v_measured - v_pred
        S = C @ P_pred @ C.T + self.R
        K = (P_pred @ C.T) / S[0, 0]

        self.x = x_pred + K * y_residual
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        I_mat = np.eye(3)
        self.P = (I_mat - K @ C) @ P_pred @ (I_mat - K @ C).T + K * self.R @ K.T

        return float(self.x[0, 0]), float(v_pred), float(self.P[0, 0])


class AdaptiveEKF(ExtendedKalmanFilter):
    """
    Adaptive Extended Kalman Filter (AEKF) with Sage-Husa Noise Estimator.
    Dynamically adjusts measurement covariance R and process covariance Q
    to handle sudden current spikes, temperature fluctuations, and noise drift.
    """

    def __init__(self, battery_cell, forgetting_factor=0.98, **kwargs):
        super().__init__(battery_cell, **kwargs)
        self.lambda_factor = float(forgetting_factor)
        self.r_val = self.R

    def reset(self, initial_soc=1.0, p_diag=(0.25, 1e-4, 1e-4)):
        super().reset(initial_soc=initial_soc, p_diag=p_diag)
        self.r_val = self.R

    def step(self, current_a, v_measured):
        exp1 = np.exp(-self.dt / self.cell.tau1)
        exp2 = np.exp(-self.dt / self.cell.tau2)

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, exp1, 0.0],
            [0.0, 0.0, exp2]
        ])
        B = np.array([
            [-self.dt / (3600.0 * self.cell.q_nominal)],
            [self.cell.r1 * (1.0 - exp1)],
            [self.cell.r2 * (1.0 - exp2)]
        ])

        # Prediction
        x_pred = A @ self.x + B * current_a
        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)
        P_pred = A @ self.P @ A.T + self.Q

        soc_pred = x_pred[0, 0]
        v_pred = self.cell.get_ocv(soc_pred) - current_a * self.cell.r0 - x_pred[1, 0] - x_pred[2, 0]
        d_ocv = self.cell.get_docv_dsoc(soc_pred)
        C = np.array([[d_ocv, -1.0, -1.0]])

        # Innovation residual
        innovation = v_measured - v_pred

        # Adaptive R update (Sage-Husa formulation with bounds)
        r_instant = innovation**2 - (C @ P_pred @ C.T)[0, 0]
        r_instant = max(1e-5, min(0.05, r_instant))
        self.r_val = self.lambda_factor * self.r_val + (1.0 - self.lambda_factor) * r_instant

        # Kalman gain with adapted R
        S = (C @ P_pred @ C.T)[0, 0] + self.r_val
        K = (P_pred @ C.T) / S

        self.x = x_pred + K * innovation
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        I_mat = np.eye(3)
        self.P = (I_mat - K @ C) @ P_pred

        return float(self.x[0, 0]), float(v_pred), float(self.P[0, 0])


class DualExtendedKalmanFilter:
    """
    Dual Extended Kalman Filter (DEKF) for Joint State (SOC) and Parameter (R0, Q_act) Tracking.
    Critical for battery degradation (SOH):
    - Filter 1 (State): Tracks x = [SOC, Up1, Up2]^T
    - Filter 2 (Parameter): Tracks theta = [R0, 1 / Q_act]^T
    Allows the model to self-tune as the battery ages.
    """

    def __init__(self, battery_cell):
        self.cell = battery_cell
        self.dt = battery_cell.dt

        # State filter: x = [SOC, Up1, Up2]
        self.x = np.array([[1.0], [0.0], [0.0]])
        self.P_x = np.diag([0.02, 1e-4, 1e-4])
        self.Q_x = np.diag([1e-6, 1e-5, 1e-5])
        self.R_x = 0.001

        # Parameter filter: theta = [R0, beta], where beta = 1 / (3600 * Q_act)
        nominal_beta = 1.0 / (3600.0 * battery_cell.q_nominal)
        self.theta = np.array([[battery_cell.r0_base], [nominal_beta]])
        self.P_theta = np.diag([1e-3, 1e-10])
        self.Q_theta = np.diag([1e-7, 1e-14])
        self.R_theta = 0.001

    def reset(self, initial_soc=1.0):
        self.x = np.array([[float(initial_soc)], [0.0], [0.0]])
        self.P_x = np.diag([0.25, 1e-4, 1e-4])
        nominal_beta = 1.0 / (3600.0 * self.cell.q_nominal)
        self.theta = np.array([[self.cell.r0_base], [nominal_beta]])
        self.P_theta = np.diag([1e-3, 1e-10])

    def step(self, current_a, v_measured):
        r0_est = max(0.005, min(0.3, self.theta[0, 0]))
        beta_est = max(1e-6, min(1e-3, self.theta[1, 0]))

        # --- 1. Parameter Filter Update ---
        # Prior prediction for parameter (random walk assumption)
        theta_pred = self.theta.copy()
        P_theta_pred = self.P_theta + self.Q_theta

        # Measurement Jacobian w.r.t parameters: C_theta = d(Vt)/d(theta)
        # Vt = OCV(SOC) - I * R0 - Up1 - Up2
        # d(Vt)/d(R0) = -I
        # d(Vt)/d(beta) ~ d(OCV)/d(SOC) * d(SOC)/d(beta) = d(OCV)/d(SOC) * (-I * dt)
        d_ocv = self.cell.get_docv_dsoc(self.x[0, 0])
        C_theta = np.array([[-current_a, -d_ocv * current_a * self.dt]])

        # Voltage prediction using prior parameter
        v_pred_param = self.cell.get_ocv(self.x[0, 0]) - current_a * theta_pred[0, 0] - self.x[1, 0] - self.x[2, 0]
        y_param_res = v_measured - v_pred_param

        S_theta = (C_theta @ P_theta_pred @ C_theta.T)[0, 0] + self.R_theta
        K_theta = (P_theta_pred @ C_theta.T) / S_theta
        self.theta = theta_pred + K_theta * y_param_res
        self.P_theta = P_theta_pred - K_theta @ C_theta @ P_theta_pred

        # Constrain parameters to physical boundaries
        self.theta[0, 0] = max(0.01, min(0.25, self.theta[0, 0]))
        self.theta[1, 0] = max(1.0 / (3600.0 * self.cell.q_nominal * 1.5), min(1.0 / (3600.0 * self.cell.q_nominal * 0.5), self.theta[1, 0]))

        # Updated parameters for state filter
        r0_curr = self.theta[0, 0]
        beta_curr = self.theta[1, 0]

        # --- 2. State Filter Update ---
        exp1 = np.exp(-self.dt / self.cell.tau1)
        exp2 = np.exp(-self.dt / self.cell.tau2)

        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, exp1, 0.0],
            [0.0, 0.0, exp2]
        ])
        B = np.array([
            [-beta_curr * self.dt],
            [self.cell.r1 * (1.0 - exp1)],
            [self.cell.r2 * (1.0 - exp2)]
        ])

        x_pred = A @ self.x + B * current_a
        x_pred[0, 0] = np.clip(x_pred[0, 0], 0.0, 1.0)
        P_x_pred = A @ self.P_x @ A.T + self.Q_x

        soc_pred = x_pred[0, 0]
        v_pred_state = self.cell.get_ocv(soc_pred) - current_a * r0_curr - x_pred[1, 0] - x_pred[2, 0]
        d_ocv_state = self.cell.get_docv_dsoc(soc_pred)
        C_x = np.array([[d_ocv_state, -1.0, -1.0]])

        y_state_res = v_measured - v_pred_state
        S_x = (C_x @ P_x_pred @ C_x.T)[0, 0] + self.R_x
        K_x = (P_x_pred @ C_x.T) / S_x

        self.x = x_pred + K_x * y_state_res
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        self.P_x = (np.eye(3) - K_x @ C_x) @ P_x_pred

        # Extracted actual capacity estimate
        cap_est = 1.0 / (3600.0 * self.theta[1, 0])

        return float(self.x[0, 0]), float(v_pred_state), float(r0_curr), float(cap_est)


class SlidingModeObserver:
    """
    Super-Twisting / Boundary-Layer Sliding Mode Observer (SMO).
    Known for robust disturbance rejection and immunity against model parameter uncertainty.
    """

    def __init__(self, battery_cell, l_gain=0.03, boundary_layer=0.03):
        self.cell = battery_cell
        self.dt = battery_cell.dt
        self.l_gain = float(l_gain)
        self.epsilon = float(boundary_layer)
        self.soc = 1.0
        self.up1 = 0.0
        self.up2 = 0.0

    def reset(self, initial_soc=1.0):
        self.soc = float(initial_soc)
        self.up1 = 0.0
        self.up2 = 0.0

    def step(self, current_a, v_measured):
        # ECM model prediction
        exp1 = np.exp(-self.dt / self.cell.tau1)
        exp2 = np.exp(-self.dt / self.cell.tau2)

        self.up1 = self.up1 * exp1 + current_a * self.cell.r1 * (1.0 - exp1)
        self.up2 = self.up2 * exp2 + current_a * self.cell.r2 * (1.0 - exp2)

        v_model = self.cell.get_ocv(self.soc) - current_a * self.cell.r0 - self.up1 - self.up2
        error_v = v_measured - v_model

        # Smooth boundary layer injection scaled by local electro-thermal slope dOCV/dSOC
        d_ocv = max(0.4, float(self.cell.get_docv_dsoc(self.soc)))
        s_val = np.clip(error_v / self.epsilon, -1.0, 1.0)
        sliding_correction = (self.l_gain * s_val) / d_ocv

        # Update SOC with Coulomb counting + sliding mode injection
        d_soc = (-current_a * self.dt) / (3600.0 * self.cell.q_nominal) + sliding_correction * self.dt
        self.soc = np.clip(self.soc + d_soc, 0.0, 1.0)

        return float(self.soc), float(v_model)
