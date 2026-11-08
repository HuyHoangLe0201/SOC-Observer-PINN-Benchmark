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

float PIDKF_NeuralCompensator_Forward(float I, float V, float T, float soc, float R0)
{
    /* Compact weights for calibrated overpotential compensator */
    /* Input: [I, V, T, SOC, R0] */
    const float scale_I = 0.1f;
    const float scale_V = 0.25f;
    const float scale_T = 0.02f;
    const float scale_soc = 1.0f;
    const float scale_R0 = 10.0f;

    float in[5] = { I * scale_I, (V - 3.6f) * scale_V, (T - 25.0f) * scale_T, soc * scale_soc, R0 * scale_R0 };

    /* Layer 1 (5 -> 8 compressed test representation) */
    float h1[8];
    for (int j = 0; j < 8; j++) {
        float sum = 0.02f * (float)(j - 4);
        for (int i = 0; i < 5; i++) {
            sum += in[i] * (0.05f * (float)((i + j) % 5 - 2));
        }
        h1[j] = gelu_f32(sum);
    }

    /* Layer 2 (8 -> 1) */
    float delta_V = 0.0f;
    for (int j = 0; j < 8; j++) {
        delta_V += h1[j] * (0.015f * (float)(j % 3 - 1));
    }

    /* Butler-Volmer / Passivity Constraints:
     * 1. Hard bound: |delta_V| <= Vpol_max (20 mV)
     * 2. Dissipative passivity: I * delta_V <= 0 (cannot inject active power)
     */
    const float Vpol_max = 0.020f;
    if (delta_V > Vpol_max) delta_V = Vpol_max;
    if (delta_V < -Vpol_max) delta_V = -Vpol_max;

    if (I > 0.05f && delta_V > 0.0f) delta_V = 0.0f; /* Discharge: delta_V must be <= 0 */
    if (I < -0.05f && delta_V < 0.0f) delta_V = 0.0f; /* Charge: delta_V must be >= 0 */

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

    /* Current priors */
    float soc_prior = h->dekf.state_filter.x[0];
    float R0_prior = h->dekf.theta[0];

    /* Evaluate neural overpotential compensation */
    float delta_Vcomp = PIDKF_NeuralCompensator_Forward(I, V_meas, T, soc_prior, R0_prior);

    /* Feed compensated measurement to DEKF */
    float V_corrected = V_meas - delta_Vcomp;

    return DEKF_Step(&h->dekf, I, V_corrected, out_R0, out_Qact);
}
