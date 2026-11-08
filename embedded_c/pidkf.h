/**
 * @file pidkf.h
 * @brief Physics-Informed Dual Kalman Filter (PIDKF) for Embedded Battery Management Systems
 */

#ifndef PIDKF_H
#define PIDKF_H

#ifdef __cplusplus
extern "C" {
#endif

#include "dekf.h"

typedef struct {
    DEKF_Handle_t dekf;
    float Vpol_max;             /* Hard overpotential bound (V) */
    float temp_core;            /* Battery core temperature (deg C) */
} PIDKF_Handle_t;

void PIDKF_Init(PIDKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2, float temp_init);
float PIDKF_Step(PIDKF_Handle_t *h, float current, float voltage, float temperature, float *out_R0, float *out_Qact);

/* Neural forward inference with Butler-Volmer passivity clipping */
float PIDKF_NeuralCompensator_Forward(float I, float V, float T, float soc, float R0);

#ifdef __cplusplus
}
#endif

#endif /* PIDKF_H */
