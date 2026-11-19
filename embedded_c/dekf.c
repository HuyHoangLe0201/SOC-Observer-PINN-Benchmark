/**
 * @file dekf.c
 * @brief Discrete-time Dual Extended Kalman Filter implementation (ANSI-C99)
 */

#include "dekf.h"
#include <math.h>
#include <string.h>

void DEKF_Init(DEKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2)
{
    memset(h, 0, sizeof(DEKF_Handle_t));
    EKF_Init(&h->state_filter, soc_init, Q_nom, R0, R1, C1, R2, C2);

    h->Q_init = Q_nom;
    h->theta[0] = R0;
    h->theta[1] = 1.0f / Q_nom;

    h->P_theta[0] = 1e-3f;
    h->P_theta[3] = 1e-2f;

    h->Q_theta[0] = 1e-7f;
    h->Q_theta[3] = 1e-6f;

    h->R_theta = 1e-2f;
    h->I_prev = 0.0f;
}

void DEKF_PredictPrior(DEKF_Handle_t *h, float I, float *out_z, float *out_u1, float *out_u2, float *out_r0)
{
    EKF_Handle_t *sf = &h->state_filter;
    float q_act = 1.0f / h->theta[1];
    float a1 = expf(-sf->dt / (sf->R1 * sf->C1));
    float b1 = sf->R1 * (1.0f - a1);
    float a2 = expf(-sf->dt / (sf->R2 * sf->C2));
    float b2 = sf->R2 * (1.0f - a2);
    float eta = (I >= 0.0f) ? 1.0f : 0.995f;

    float z_m = sf->x[0] - (eta * sf->dt / (3600.0f * q_act)) * I;
    if (z_m < 0.0f) z_m = 0.0f;
    if (z_m > 1.0f) z_m = 1.0f;

    if (out_z)  *out_z  = z_m;
    if (out_u1) *out_u1 = a1 * sf->x[1] + b1 * I;
    if (out_u2) *out_u2 = a2 * sf->x[2] + b2 * I;
    if (out_r0) *out_r0 = h->theta[0];
}

float DEKF_StepEx(DEKF_Handle_t *h, float I, float V_meas, float delta_V_comp,
                  float R_x_eff, float R_theta_eff, float *out_R0, float *out_Qact)
{
    EKF_Handle_t *sf = &h->state_filter;
    float r0_curr = h->theta[0];
    float q_act = 1.0f / h->theta[1];

    float a1 = expf(-sf->dt / (sf->R1 * sf->C1));
    float b1 = sf->R1 * (1.0f - a1);
    float a2 = expf(-sf->dt / (sf->R2 * sf->C2));
    float b2 = sf->R2 * (1.0f - a2);
    float eta = (I >= 0.0f) ? 1.0f : 0.995f;

    /* 1. State Time Update (a priori) */
    float x_pred[3];
    x_pred[0] = sf->x[0] - (eta * sf->dt / (3600.0f * q_act)) * I;
    if (x_pred[0] < 0.0f) x_pred[0] = 0.0f;
    if (x_pred[0] > 1.0f) x_pred[0] = 1.0f;
    x_pred[1] = a1 * sf->x[1] + b1 * I;
    x_pred[2] = a2 * sf->x[2] + b2 * I;

    /* P_x^- = A P_x^+ A^T + Q_x with A = diag(1, a1, a2) */
    float Px_pred[9];
    Px_pred[0] = sf->P[0] + sf->Q[0];
    Px_pred[1] = a1 * sf->P[1];
    Px_pred[2] = a2 * sf->P[2];
    Px_pred[3] = a1 * sf->P[3];
    Px_pred[4] = a1 * a1 * sf->P[4] + sf->Q[4];
    Px_pred[5] = a1 * a2 * sf->P[5];
    Px_pred[6] = a2 * sf->P[6];
    Px_pred[7] = a1 * a2 * sf->P[7];
    Px_pred[8] = a2 * a2 * sf->P[8] + sf->Q[8];

    /* 2. State Measurement Update */
    float Vocv, dVocv;
    EKF_OCV_NMC(x_pred[0], &Vocv, &dVocv);
    float V_pred = Vocv - x_pred[1] - x_pred[2] - I * r0_curr + delta_V_comp;
    float e_k = V_meas - V_pred;
    float H[3] = { dVocv, -1.0f, -1.0f };

    int is_rest = (fabsf(I) < 0.010f) ? 1 : 0;
    float K_x[3] = { 0.0f, 0.0f, 0.0f };
    if (!is_rest) {
        float PH_T[3] = {
            Px_pred[0]*H[0] + Px_pred[1]*H[1] + Px_pred[2]*H[2],
            Px_pred[3]*H[0] + Px_pred[4]*H[1] + Px_pred[5]*H[2],
            Px_pred[6]*H[0] + Px_pred[7]*H[1] + Px_pred[8]*H[2]
        };
        float S_x = H[0]*PH_T[0] + H[1]*PH_T[1] + H[2]*PH_T[2] + R_x_eff;
        float invSx = 1.0f / S_x;
        K_x[0] = PH_T[0] * invSx;
        K_x[1] = PH_T[1] * invSx;
        K_x[2] = PH_T[2] * invSx;
    }

    sf->x[0] = x_pred[0] + K_x[0] * e_k;
    sf->x[1] = x_pred[1] + K_x[1] * e_k;
    sf->x[2] = x_pred[2] + K_x[2] * e_k;
    if (sf->x[0] < 0.0f) sf->x[0] = 0.0f;
    if (sf->x[0] > 1.0f) sf->x[0] = 1.0f;

    /* Joseph stabilized state covariance */
    float IKH[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            IKH[i*3 + j] = ((i == j) ? 1.0f : 0.0f) - K_x[i] * H[j];
        }
    }
    float tmp3[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            tmp3[i*3 + j] = IKH[i*3 + 0]*Px_pred[0*3 + j]
                          + IKH[i*3 + 1]*Px_pred[1*3 + j]
                          + IKH[i*3 + 2]*Px_pred[2*3 + j];
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sf->P[i*3 + j] = tmp3[i*3 + 0]*IKH[j*3 + 0]
                           + tmp3[i*3 + 1]*IKH[j*3 + 1]
                           + tmp3[i*3 + 2]*IKH[j*3 + 2]
                           + K_x[i] * R_x_eff * K_x[j];
        }
    }

    /* 3. Parameter Filter (Plett 2004 Recurrent Total Derivative Sensitivity) */
    float Pth_pred[4] = {
        h->P_theta[0] + h->Q_theta[0], h->P_theta[1] + h->Q_theta[1],
        h->P_theta[2] + h->Q_theta[2], h->P_theta[3] + h->Q_theta[3]
    };

    float df_dth01 = -(eta * sf->dt * I) / 3600.0f;
    float dx_m[6];
    dx_m[0] = h->dx_dtheta[0];
    dx_m[1] = h->dx_dtheta[1] + df_dth01;
    dx_m[2] = a1 * h->dx_dtheta[2];
    dx_m[3] = a1 * h->dx_dtheta[3];
    dx_m[4] = a2 * h->dx_dtheta[4];
    dx_m[5] = a2 * h->dx_dtheta[5];

    float C_theta[2];
    C_theta[0] = -I + (H[0]*dx_m[0] + H[1]*dx_m[2] + H[2]*dx_m[4]);
    C_theta[1] =       (H[0]*dx_m[1] + H[1]*dx_m[3] + H[2]*dx_m[5]);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 2; c++) {
            h->dx_dtheta[r*2 + c] = dx_m[r*2 + c] - K_x[r] * C_theta[c];
        }
    }

    float K_theta[2] = { 0.0f, 0.0f };
    if (!is_rest) {
        float PC_T[2] = {
            Pth_pred[0]*C_theta[0] + Pth_pred[1]*C_theta[1],
            Pth_pred[2]*C_theta[0] + Pth_pred[3]*C_theta[1]
        };
        float S_theta = C_theta[0]*PC_T[0] + C_theta[1]*PC_T[1] + R_theta_eff;
        float invS = 1.0f / S_theta;
        K_theta[0] = PC_T[0] * invS;
        K_theta[1] = PC_T[1] * invS;
    }

    h->theta[0] += K_theta[0] * e_k;
    h->theta[1] += K_theta[1] * e_k;

    /* Physical bounds strictly anchored to immutable Q_init */
    if (h->theta[0] < 0.005f) h->theta[0] = 0.005f;
    if (h->theta[0] > 0.250f) h->theta[0] = 0.250f;
    float q_inv_min = 1.0f / (1.5f * h->Q_init);
    float q_inv_max = 1.0f / (0.5f * h->Q_init);
    if (h->theta[1] < q_inv_min) h->theta[1] = q_inv_min;
    if (h->theta[1] > q_inv_max) h->theta[1] = q_inv_max;

    /* Joseph stabilized parameter covariance */
    float IKC[4] = {
        1.0f - K_theta[0]*C_theta[0], -K_theta[0]*C_theta[1],
        -K_theta[1]*C_theta[0],        1.0f - K_theta[1]*C_theta[1]
    };
    float tmp2[4] = {
        IKC[0]*Pth_pred[0] + IKC[1]*Pth_pred[2], IKC[0]*Pth_pred[1] + IKC[1]*Pth_pred[3],
        IKC[2]*Pth_pred[0] + IKC[3]*Pth_pred[2], IKC[2]*Pth_pred[1] + IKC[3]*Pth_pred[3]
    };
    h->P_theta[0] = tmp2[0]*IKC[0] + tmp2[1]*IKC[1] + K_theta[0]*R_theta_eff*K_theta[0];
    h->P_theta[1] = tmp2[0]*IKC[2] + tmp2[1]*IKC[3] + K_theta[0]*R_theta_eff*K_theta[1];
    h->P_theta[2] = tmp2[2]*IKC[0] + tmp2[3]*IKC[1] + K_theta[1]*R_theta_eff*K_theta[0];
    h->P_theta[3] = tmp2[2]*IKC[2] + tmp2[3]*IKC[3] + K_theta[1]*R_theta_eff*K_theta[1];

    h->I_prev = I;
    if (out_R0) *out_R0 = h->theta[0];
    if (out_Qact) *out_Qact = 1.0f / h->theta[1];

    return sf->x[0];
}

float DEKF_Step(DEKF_Handle_t *h, float I, float V_meas, float *out_R0, float *out_Qact)
{
    return DEKF_StepEx(h, I, V_meas, 0.0f, h->state_filter.R, h->R_theta, out_R0, out_Qact);
}
