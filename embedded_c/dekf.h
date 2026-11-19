/**
 * @file dekf.h
 * @brief Discrete-time Dual Extended Kalman Filter for Li-ion SOC & SOH Estimation
 * @details Uses Plett (2004) recurrent total derivative sensitivity and stabilized
 *          Joseph covariance forms on both state and parameter filters, with separate
 *          immutable nominal capacity anchor Q_init for safe capacity clamping.
 */

#ifndef DEKF_H
#define DEKF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ekf.h"

#define DEKF_PARAM_DIM 2 /* theta = [R0 (Ohm), 1/Q_act (Ah^-1)]^T */

typedef struct {
    EKF_Handle_t state_filter;

    float Q_init;                   /* Immutable nominal fresh capacity anchor (Ah) */
    float theta[DEKF_PARAM_DIM];    /* [R0, 1/Q_act] */
    float P_theta[DEKF_PARAM_DIM * DEKF_PARAM_DIM];
    float Q_theta[DEKF_PARAM_DIM * DEKF_PARAM_DIM];
    float R_theta;

    /* 3x2 row-major Plett (2004) recurrent total derivative dx^+/dtheta */
    float dx_dtheta[EKF_STATE_DIM * DEKF_PARAM_DIM];
    float I_prev;
} DEKF_Handle_t;

void DEKF_Init(DEKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2);

void DEKF_PredictPrior(DEKF_Handle_t *h, float I, float *out_z, float *out_u1, float *out_u2, float *out_r0);

float DEKF_StepEx(DEKF_Handle_t *h, float I, float V_meas, float delta_V_comp,
                  float R_x_eff, float R_theta_eff, float *out_R0, float *out_Qact);

float DEKF_Step(DEKF_Handle_t *h, float I, float V_meas, float *out_R0, float *out_Qact);

#ifdef __cplusplus
}
#endif

#endif /* DEKF_H */
