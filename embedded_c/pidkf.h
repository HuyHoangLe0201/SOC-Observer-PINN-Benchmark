/**
 * @file pidkf.h
 * @brief Physics-Informed Dual Kalman Filter (PIDKF) for Embedded BMS
 */

#ifndef PIDKF_H
#define PIDKF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "dekf.h"

typedef struct {
    DEKF_Handle_t dekf;
    float kappa_0;              /* Zero-current equilibrium residual floor (V) */
    float kappa_1;              /* Ohmic passivity slope coefficient (Ohm) */
    float i_max;                /* Saturation current clamp (A) */
    float sigma_nn;             /* Neural compensator uncertainty std (V) */
    float sigma_approx;         /* Exogenous Jacobian omission std (V) */
    float temp_core;            /* Battery core temperature (deg C) */
} PIDKF_Handle_t;

void PIDKF_Init(PIDKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2, float temp_init);
float PIDKF_Step(PIDKF_Handle_t *h, float current, float voltage, float temperature, float *out_R0, float *out_Qact);

/* Causal neural forward inference with Butler-Volmer passivity projection */
float PIDKF_NeuralCompensator_Forward(float I, float T, float soc_prior, float U1_prior, float U2_prior, float R0_est);

#ifdef __cplusplus
}
#endif

#endif /* PIDKF_H */
