/**
 * @file dekf.c
 * @brief Discrete-time Dual Extended Kalman Filter implementation for ARM Cortex-M
 */

#include "dekf.h"
#include <string.h>

void DEKF_Init(DEKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2)
{
    memset(h, 0, sizeof(DEKF_Handle_t));
    EKF_Init(&h->state_filter, soc_init, Q_nom, R0, R1, C1, R2, C2);

    /* Parameters: theta[0] = R0, theta[1] = 1.0 / Q_act */
    h->theta[0] = R0;
    h->theta[1] = 1.0f / Q_nom;

    /* Parameter covariance */
    h->P_theta[0] = 1e-4f;
    h->P_theta[3] = 1e-4f;

    /* Parameter process noise */
    h->Q_theta[0] = 1e-8f;
    h->Q_theta[3] = 1e-9f;

    h->R_theta = 1e-2f; /* 0.01 V^2 */
    h->I_prev = 0.0f;
}

float DEKF_Step(DEKF_Handle_t *h, float I, float V_meas, float *out_R0, float *out_Qact)
{
    /* 1. Update State Filter using current parameters */
    h->state_filter.R0 = h->theta[0];
    h->state_filter.Q_nom = 1.0f / h->theta[1];

    float soc_est = EKF_Step(&h->state_filter, I, V_meas);

    /* 2. Parameter Filter Step */
    /* Time update: P_theta = P_theta + Q_theta */
    h->P_theta[0] += h->Q_theta[0];
    h->P_theta[3] += h->Q_theta[3];

    /* Parameter sensitivity matrix C_theta = [ -I, -eta*dt*I_prev/3600 * dVocv ] */
    float Vocv, dVocv;
    EKF_OCV_NMC(soc_est, &Vocv, &dVocv);

    float C_theta[2];
    C_theta[0] = -I;
    C_theta[1] = -(h->state_filter.eta * h->state_filter.dt * h->I_prev / 3600.0f) * dVocv;

    /* Predicted terminal voltage with current parameter prior */
    float V_pred = Vocv - h->state_filter.x[1] - h->state_filter.x[2] - I * h->theta[0];
    float e_theta = V_meas - V_pred;

    /* S_theta = C_theta P_theta C_theta^T + R_theta */
    float PC_T[2];
    PC_T[0] = h->P_theta[0]*C_theta[0] + h->P_theta[1]*C_theta[1];
    PC_T[1] = h->P_theta[2]*C_theta[0] + h->P_theta[3]*C_theta[1];

    float S_theta = C_theta[0]*PC_T[0] + C_theta[1]*PC_T[1] + h->R_theta;
    float invS = 1.0f / S_theta;

    /* Kalman Gain K_theta */
    float K_theta[2] = { PC_T[0] * invS, PC_T[1] * invS };

    /* Innovation gating at rest */
    if (I > -0.01f && I < 0.01f) {
        K_theta[0] = 0.0f;
        K_theta[1] = 0.0f;
    }

    /* Update parameters */
    h->theta[0] += K_theta[0] * e_theta;
    h->theta[1] += K_theta[1] * e_theta;

    /* Physical bounds */
    if (h->theta[0] < 0.01f) h->theta[0] = 0.01f;
    if (h->theta[0] > 0.20f) h->theta[0] = 0.20f;
    if (h->theta[1] < 1.0f / (1.5f * h->state_filter.Q_nom)) h->theta[1] = 1.0f / (1.5f * h->state_filter.Q_nom);
    if (h->theta[1] > 1.0f / (0.5f * h->state_filter.Q_nom)) h->theta[1] = 1.0f / (0.5f * h->state_filter.Q_nom);

    /* Stabilized Joseph Covariance for Parameter Filter */
    float IKC[4] = {
        1.0f - K_theta[0]*C_theta[0], -K_theta[0]*C_theta[1],
        -K_theta[1]*C_theta[0],        1.0f - K_theta[1]*C_theta[1]
    };

    float temp[4];
    temp[0] = IKC[0]*h->P_theta[0] + IKC[1]*h->P_theta[2];
    temp[1] = IKC[0]*h->P_theta[1] + IKC[1]*h->P_theta[3];
    temp[2] = IKC[2]*h->P_theta[0] + IKC[3]*h->P_theta[2];
    temp[3] = IKC[2]*h->P_theta[1] + IKC[3]*h->P_theta[3];

    h->P_theta[0] = temp[0]*IKC[0] + temp[1]*IKC[1] + K_theta[0]*h->R_theta*K_theta[0];
    h->P_theta[1] = temp[0]*IKC[2] + temp[1]*IKC[3] + K_theta[0]*h->R_theta*K_theta[1];
    h->P_theta[2] = temp[2]*IKC[0] + temp[3]*IKC[1] + K_theta[1]*h->R_theta*K_theta[0];
    h->P_theta[3] = temp[2]*IKC[2] + temp[3]*IKC[3] + K_theta[1]*h->R_theta*K_theta[1];

    h->I_prev = I;

    if (out_R0) *out_R0 = h->theta[0];
    if (out_Qact) *out_Qact = 1.0f / h->theta[1];

    return soc_est;
}
