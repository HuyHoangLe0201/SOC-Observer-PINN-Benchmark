/**
 * @file pidkf.c
 * @brief Physics-Informed Dual Kalman Filter reference implementation (ANSI-C99)
 * @details Uses exact exported Seed-42 float32 weights (pidkf_weights.h), a priori state +
 *          fast-converging R0_hat conditioning, Butler-Volmer saturation kappa_0 + kappa_1*min(|I|, I_max),
 *          and hard dissipative passivity projection I_k * delta_V_comp <= 0.
 */

#include "pidkf.h"
#include "pidkf_weights.h"
#include <math.h>

/* Exact GELU tanh-approximation matching PyTorch nn.GELU(approximate='tanh') */
static inline float gelu_f32(float x)
{
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

float PIDKF_NeuralCompensator_Forward(float I, float T, float soc_prior, float U1_prior, float U2_prior, float R0_est)
{
    /* Input normalization matching src/pidkf_hybrid.py build_feature_vector() for NMC:
     * u_k = [I/4.0, (T-25)/20.0, z_prior, 15.0*U1_prior, 15.0*U2_prior, (R0_est - 0.035)/0.030]^T
     */
    float in[6] = {
        I / 4.0f,
        (T - 25.0f) / 20.0f,
        soc_prior,
        U1_prior * 15.0f,
        U2_prior * 15.0f,
        (R0_est - 0.035f) / 0.030f
    };

    /* Hidden Layer 1: 6 -> 16 with GELU */
    float h1[16];
    for (int j = 0; j < 16; j++) {
        float sum = PIDKF_B1[j];
        for (int i = 0; i < 6; i++) {
            sum += PIDKF_W1[j][i] * in[i];
        }
        h1[j] = gelu_f32(sum);
    }

    /* Hidden Layer 2: 16 -> 8 with GELU */
    float h2[8];
    for (int j = 0; j < 8; j++) {
        float sum = PIDKF_B2[j];
        for (int i = 0; i < 16; i++) {
            sum += PIDKF_W2[j][i] * h1[i];
        }
        h2[j] = gelu_f32(sum);
    }

    /* Linear Output Layer: 8 -> 1 scaled by scale_out = 0.15f */
    float acc = PIDKF_B3;
    for (int j = 0; j < 8; j++) {
        acc += PIDKF_W3[j] * h2[j];
    }
    float delta_V = 0.15f * acc;

    /* Current-dependent Butler-Volmer passivity bound: kappa_0 = 15 mV, kappa_1 = 22 mOhm, I_max = 6.0 A */
    float absI = fabsf(I);
    float I_clamped = (absI < 6.0f) ? absI : 6.0f;
    float Vpol_max = 0.015f + 0.022f * I_clamped;
    if (delta_V > Vpol_max)  delta_V = Vpol_max;
    if (delta_V < -Vpol_max) delta_V = -Vpol_max;

    /* Hard dissipative passivity projection outside 10 mA rest threshold: I_k * delta_V <= 0 */
    if (I > 0.010f && delta_V > 0.0f)  delta_V = 0.0f;
    if (I < -0.010f && delta_V < 0.0f) delta_V = 0.0f;

    return delta_V;
}

void PIDKF_Init(PIDKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2, float temp_init)
{
    DEKF_Init(&h->dekf, soc_init, Q_nom, R0, R1, C1, R2, C2);
    /* Match Python PIDKF parameter covariance defaults: P0_theta = diag(1e-3, 1e-4), Q_theta = diag(1e-7, 1e-9) */
    h->dekf.P_theta[0] = 1e-3f;
    h->dekf.P_theta[3] = 1e-4f;
    h->dekf.Q_theta[0] = 1e-7f;
    h->dekf.Q_theta[3] = 1e-9f;

    h->kappa_0 = 0.015f;
    h->kappa_1 = 0.022f;
    h->i_max = 6.0f;
    h->sigma_nn = 0.008f;
    h->sigma_approx = 0.002f;
    h->temp_core = temp_init;
}

float PIDKF_Step(PIDKF_Handle_t *h, float I, float V_meas, float T, float *out_R0, float *out_Qact)
{
    h->temp_core = T;

    /* Strictly a priori state prediction & fast R0_hat before measurement update */
    float z_prior, U1_prior, U2_prior, R0_est;
    DEKF_PredictPrior(&h->dekf, I, &z_prior, &U1_prior, &U2_prior, &R0_est);

    float delta_Vcomp = PIDKF_NeuralCompensator_Forward(I, T, z_prior, U1_prior, U2_prior, R0_est);

    float r_x_eff = h->dekf.state_filter.R + (h->sigma_nn * h->sigma_nn) + (h->sigma_approx * h->sigma_approx);
    float r_th_eff = h->dekf.R_theta + (h->sigma_nn * h->sigma_nn);

    return DEKF_StepEx(&h->dekf, I, V_meas, delta_Vcomp, r_x_eff, r_th_eff, out_R0, out_Qact);
}
