/**
 * @file ekf.h
 * @brief Discrete-time Extended Kalman Filter for Li-ion SOC Estimation
 * @details Implemented with stabilized Joseph covariance form for ARM Cortex-M microcontrollers.
 */

#ifndef EKF_H
#define EKF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#define EKF_STATE_DIM 3  /* x = [SOC, U1, U2]^T */

typedef struct {
    float x[EKF_STATE_DIM];     /* State vector: [SOC, U1, U2] */
    float P[EKF_STATE_DIM * EKF_STATE_DIM]; /* State error covariance matrix (3x3) */
    float Q[EKF_STATE_DIM * EKF_STATE_DIM]; /* Process noise covariance (3x3) */
    float R;                    /* Voltage measurement noise variance */
    
    /* Battery parameters */
    float Q_nom;                /* Usable capacity (A*h) */
    float R0;                   /* Ohmic + SEI resistance (Ohm) */
    float R1;                   /* Polarization resistance 1 (Ohm) */
    float C1;                   /* Polarization capacitance 1 (F) */
    float R2;                   /* Polarization resistance 2 (Ohm) */
    float C2;                   /* Polarization capacitance 2 (F) */
    float eta;                  /* Coulombic efficiency */
    float dt;                   /* Sample time (s) */
} EKF_Handle_t;

/**
 * @brief Initialize EKF handle with nominal priors.
 */
void EKF_Init(EKF_Handle_t *handle, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2);

/**
 * @brief Execute one EKF time and measurement update step.
 * @param handle EKF handle
 * @param current Measured current (A, positive = discharge)
 * @param voltage Measured terminal voltage (V)
 * @return Estimated SOC in [0, 1]
 */
float EKF_Step(EKF_Handle_t *handle, float current, float voltage);

/**
 * @brief Evaluates polynomial OCV curve and derivative for NMC chemistry.
 */
void EKF_OCV_NMC(float soc, float *Vocv, float *dVocv);

#ifdef __cplusplus
}
#endif

#endif /* EKF_H */
