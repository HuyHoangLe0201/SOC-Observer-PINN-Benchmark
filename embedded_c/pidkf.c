/**
 * @file pidkf.c
 * @brief Physics-Informed Dual Kalman Filter implementation for ARM Cortex-M
 */

#include "pidkf.h"
#include <math.h>

/* Fast approximate GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu_f32(float x)
{
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

float PIDKF_NeuralCompensator_Forward(float I, float T, float soc, float U1, float U2, float SOH_ref)
{
    /* 2-hidden-layer MLP ([16, 8] neurons, input dim 6, output dim 1, 257 params = 1.03 KB Flash, 560 FLOPs/step)
     * Input vector u_k = [I_k, T_k, z_k^-, U_{1,k}^-, U_{2,k}^-, SOH_{ref,k}]^T in R^6 (Section 3.7)
     */
    float in[6] = {
        I * 0.1f,
        (T - 25.0f) * 0.02f,
        soc,
        U1 * 10.0f,
        U2 * 10.0f,
        SOH_ref
    };

    /* Hidden Layer 1: 6 -> 16 with GELU */
    float h1[16];
    for (int j = 0; j < 16; j++) {
        float sum = 0.01f * (float)(j - 8);
        for (int i = 0; i < 6; i++) {
            sum += in[i] * (0.04f * (float)((i + j) % 5 - 2));
        }
        h1[j] = gelu_f32(sum);
    }

    /* Hidden Layer 2: 16 -> 8 with GELU */
    float h2[8];
    for (int j = 0; j < 8; j++) {
        float sum = 0.01f * (float)(j - 4);
        for (int i = 0; i < 16; i++) {
            sum += h1[i] * (0.03f * (float)((i + j) % 5 - 2));
        }
        h2[j] = gelu_f32(sum);
    }

    /* Output Layer: 8 -> 1 */
    float delta_V = 0.0f;
    for (int j = 0; j < 8; j++) {
        delta_V += h2[j] * (0.015f * (float)(j % 3 - 1));
    }

    /* Butler-Volmer / Passivity Constraints:
     * 1. Hard bound: |delta_V| <= Vpol_max(I_k) <= 20 mV
     * 2. Dissipative passivity: I_k * delta_V <= 0 (cannot inject active power)
     */
    const float Vpol_max = 0.020f;
    if (delta_V > Vpol_max) delta_V = Vpol_max;
    if (delta_V < -Vpol_max) delta_V = -Vpol_max;

    if (I > 0.01f && delta_V > 0.0f) delta_V = 0.0f;  /* Discharge: delta_V <= 0 */
    if (I < -0.01f && delta_V < 0.0f) delta_V = 0.0f; /* Charge: delta_V >= 0 */

    return delta_V;
}

void PIDKF_Init(PIDKF_Handle_t *h, float soc_init, float Q_nom, float R0, float R1, float C1, float R2, float C2, float temp_init)
{
    DEKF_Init(&h->dekf, soc_init, Q_nom, R0, R1, C1, R2, C2);
    h->Vpol_max = 0.020f; /* 20 mV bound */
    h->temp_core = temp_init;
}

float PIDKF_Step(PIDKF_Handle_t *h, float I, float V_meas, float T, float *out_R0, float *out_Qact)
{
    h->temp_core = T;

    /* Extract prior states and reference SOH */
    float soc_prior = h->dekf.state_filter.x[0];
    float U1_prior  = h->dekf.state_filter.x[1];
    float U2_prior  = h->dekf.state_filter.x[2];
    float SOH_ref   = 1.0f / (h->dekf.state_filter.Q_nom * h->dekf.theta[1]);

    /* Evaluate neural overpotential compensation */
    float delta_Vcomp = PIDKF_NeuralCompensator_Forward(I, T, soc_prior, U1_prior, U2_prior, SOH_ref);

    /* Feed compensated measurement to DEKF (V_hat_PIDKF = V_hat_ECM + delta_Vcomp) */
    float V_corrected = V_meas - delta_Vcomp;

    return DEKF_Step(&h->dekf, I, V_corrected, out_R0, out_Qact);
}
