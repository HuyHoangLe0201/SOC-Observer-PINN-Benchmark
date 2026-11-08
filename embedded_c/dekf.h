/**
 * @file dekf.h
 * @brief Discrete-time Dual Extended Kalman Filter for Li-ion SOC & SOH Estimation
 */

#ifndef DEKF_H
#define DEKF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ekf.h"

#define DEKF_PARAM_DIM 2 /* theta = [R0, 1/Q_act]^T */

typedef struct {
    EKF_Handle_t state_filter;
    
    /* Parameter Filter */
    float theta[DEKF_PARAM_DIM];    /* [R0, invQ] */
    float P_theta[DEKF_PARAM_DIM * DEKF_PARAM_DIM];
    float Q_theta[DEKF_PARAM_DIM * DEKF_PARAM_DIM];
    float R_theta;

    float I_prev;
} DEKF_Handle_t;

void DEKF_Init(DEKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2);
float DEKF_Step(DEKF_Handle_t *h, float I, float V_meas, float *out_R0, float *out_Qact);

#ifdef __cplusplus
}
#endif

#endif /* DEKF_H */
