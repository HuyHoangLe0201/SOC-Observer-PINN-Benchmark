/**
 * @file ekf.c
 * @brief Discrete-time Extended Kalman Filter implementation for ARM Cortex-M
 */

#include "ekf.h"
#include <math.h>
#include <string.h>

void EKF_OCV_NMC(float soc, float *Vocv, float *dVocv)
{
    /* Strictly monotonic 5th-order polynomial calibrated for Samsung INR18650-20R (Eq. 11) */
    if (soc < 0.0f) soc = 0.0f;
    if (soc > 1.0f) soc = 1.0f;

    const float p5 =   9.204f;
    const float p4 = -28.841f;
    const float p3 =  33.376f;
    const float p2 = -17.419f;
    const float p1 =   4.674f;
    const float p0 =   3.212f;

    *Vocv = p0 + soc*(p1 + soc*(p2 + soc*(p3 + soc*(p4 + soc*p5))));
    *dVocv = p1 + soc*(2.0f*p2 + soc*(3.0f*p3 + soc*(4.0f*p4 + soc*5.0f*p5)));
    if (*dVocv < 0.40f) *dVocv = 0.40f;
}

void EKF_Init(EKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2)
{
    memset(h, 0, sizeof(EKF_Handle_t));
    h->x[0] = soc_init;
    h->x[1] = 0.0f;
    h->x[2] = 0.0f;

    /* Covariance P0 */
    h->P[0] = 1e-2f; /* P_soc */
    h->P[4] = 1e-4f; /* P_u1 */
    h->P[8] = 1e-4f; /* P_u2 */

    /* Process noise Q */
    h->Q[0] = 1e-6f;
    h->Q[4] = 1e-5f;
    h->Q[8] = 1e-5f;

    h->R = 1e-3f; /* 1000 uV^2 */
    h->Q_nom = Q_nom;
    h->R0 = R0;
    h->R1 = R1;
    h->C1 = C1;
    h->R2 = R2;
    h->C2 = C2;
    h->eta = 0.995f;
    h->dt = 1.0f;
}

float EKF_Step(EKF_Handle_t *h, float I, float V_meas)
{
    /* 1. Time Update */
    float a1 = expf(-h->dt / (h->R1 * h->C1));
    float b1 = h->R1 * (1.0f - a1);
    float a2 = expf(-h->dt / (h->R2 * h->C2));
    float b2 = h->R2 * (1.0f - a2);

    h->x[0] = h->x[0] - (h->eta * h->dt / (3600.0f * h->Q_nom)) * I;
    h->x[1] = a1 * h->x[1] + b1 * I;
    h->x[2] = a2 * h->x[2] + b2 * I;

    /* Propagate covariance: P = A P A^T + Q */
    h->P[0] = h->P[0] + h->Q[0];
    h->P[1] = a1 * h->P[1];
    h->P[2] = a2 * h->P[2];
    h->P[3] = a1 * h->P[3];
    h->P[4] = a1 * a1 * h->P[4] + h->Q[4];
    h->P[5] = a1 * a2 * h->P[5];
    h->P[6] = a2 * h->P[6];
    h->P[7] = a1 * a2 * h->P[7];
    h->P[8] = a2 * a2 * h->P[8] + h->Q[8];

    /* 2. Measurement Update */
    float Vocv, dVocv;
    EKF_OCV_NMC(h->x[0], &Vocv, &dVocv);
    float V_pred = Vocv - h->x[1] - h->x[2] - I * h->R0;
    float innovation = V_meas - V_pred;

    /* Observation Jacobian H = [dVocv, -1, -1] */
    float H[3] = { dVocv, -1.0f, -1.0f };

    /* S = H P H^T + R */
    float PH_T[3];
    PH_T[0] = h->P[0]*H[0] + h->P[1]*H[1] + h->P[2]*H[2];
    PH_T[1] = h->P[3]*H[0] + h->P[4]*H[1] + h->P[5]*H[2];
    PH_T[2] = h->P[6]*H[0] + h->P[7]*H[1] + h->P[8]*H[2];

    float S = H[0]*PH_T[0] + H[1]*PH_T[1] + H[2]*PH_T[2] + h->R;
    float invS = 1.0f / S;

    /* Kalman Gain K = P H^T / S */
    float K[3] = { PH_T[0] * invS, PH_T[1] * invS, PH_T[2] * invS };

    /* State correction */
    h->x[0] += K[0] * innovation;
    h->x[1] += K[1] * innovation;
    h->x[2] += K[2] * innovation;

    /* Bounded SOC */
    if (h->x[0] < 0.0f) h->x[0] = 0.0f;
    if (h->x[0] > 1.0f) h->x[0] = 1.0f;

    /* Stabilized Joseph Covariance Update: P = (I - K H) P (I - K H)^T + K R K^T */
    float IKH[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float delta = (i == j) ? 1.0f : 0.0f;
            IKH[i*3 + j] = delta - K[i] * H[j];
        }
    }

    float temp[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i*3 + j] = IKH[i*3 + 0]*h->P[0*3 + j] +
                            IKH[i*3 + 1]*h->P[1*3 + j] +
                            IKH[i*3 + 2]*h->P[2*3 + j];
        }
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            h->P[i*3 + j] = temp[i*3 + 0]*IKH[j*3 + 0] +
                            temp[i*3 + 1]*IKH[j*3 + 1] +
                            temp[i*3 + 2]*IKH[j*3 + 2] +
                            K[i] * h->R * K[j];
        }
    }

    return h->x[0];
}
