import matplotlib.pyplot as plt
import numpy as np
import os

# Set clean matplotlib params
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

algs = ['SMO', 'EKF', 'AEKF', 'DEKF', 'PIDKF (Proposed)', 'PINN-SOC', 'BiLSTM-Attn']
latencies = [1.67, 3.87, 5.24, 8.63, 14.20, 127.98, 2440.48]
aged_rmse = [4.254, 1.985, 3.834, 0.452, 0.414, 4.177, 4.202]
fresh_rmse = [0.941, 0.065, 0.759, 0.191, 0.063, 2.781, 2.813]

colors = {
    'SMO': '#2CA02C',
    'EKF': '#1F77B4',
    'AEKF': '#9467BD',
    'DEKF': '#FF7F0E',
    'PIDKF (Proposed)': '#D62728',
    'PINN-SOC': '#8C564B',
    'BiLSTM-Attn': '#E377C2'
}

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)

# Plot degradation arrows from fresh to aged
for i, a in enumerate(algs):
    ax.annotate('', xy=(latencies[i], aged_rmse[i]), xytext=(latencies[i], fresh_rmse[i]),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1.1, ls=':'))

# Plot fresh points as hollow circles
ax.scatter(latencies, fresh_rmse, s=65, facecolors='white', edgecolors='#777777', linewidth=1.2,
           label='Fresh Baseline (100% SOH)', zorder=4)

# Plot aged points as solid filled markers
for i, a in enumerate(algs):
    m = '*' if 'PIDKF' in a else 'o'
    s = 230 if 'PIDKF' in a else 120
    ax.scatter(latencies[i], aged_rmse[i], s=s, marker=m, color=colors[a],
               edgecolors='black', linewidth=1.2, zorder=6)

# Annotations for Aged points
annot_pos = {
    'SMO': (1.1, 4.45),
    'EKF': (2.4, 1.55),
    'AEKF': (5.7, 3.55),
    'DEKF': (9.7, 0.55),
    'PIDKF (Proposed)': (16.2, 0.32),
    'PINN-SOC': (60.0, 4.35),
    'BiLSTM-Attn': (650.0, 4.40)
}

for a, pos in annot_pos.items():
    ax.annotate(a, pos, fontsize=8.5, fontweight='bold',
                color=colors[a] if 'PIDKF' in a else '#222222')

# Draw Pareto Frontier for Degraded Cells (85% SOH)
pareto_lat = [1.67, 3.87, 8.63, 14.20]
pareto_rmse = [4.254, 1.985, 0.452, 0.414]
ax.plot(pareto_lat, pareto_rmse, color='#D62728', linestyle='-', linewidth=2.0, alpha=0.9,
        label='Aged Pareto Frontier (85% SOH)', zorder=5)

# Fresh Pareto curve
ax.plot([1.67, 3.87], [0.941, 0.065], color='#777777', linestyle='--', linewidth=1.2, alpha=0.8,
        label='Fresh Pareto Frontier (100% SOH)')

ax.set_xscale('log')
ax.set_xlabel(r'ARM Cortex-M4 Execution Latency ($\mu$s/step) [Log Scale]', fontsize=9.5, fontweight='bold')
ax.set_ylabel('SOC Estimation RMSE (%) [Lower is Better]', fontsize=9.5, fontweight='bold')
ax.set_title('Pareto Frontier: Estimation Accuracy vs. Cortex-M4 Latency on Degraded Cells (85% SOH)', fontsize=10, fontweight='bold')
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.set_xlim(0.8, 6500.0)
ax.set_ylim(-0.35, 5.2)

# BMS Real-time region
ax.axvspan(0.8, 20.0, alpha=0.08, color='#2CA02C')
ax.axvline(20.0, color='#2CA02C', linestyle='--', linewidth=1.0, alpha=0.8)
ax.text(1.15, -0.15, r'Ideal Real-Time BMS Zone (<20 $\mu$s)', color='#1E7E34', fontsize=8.0, fontweight='bold')

ax.legend(loc='upper right', framealpha=0.94, fontsize=8.0)
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/fig10_embedded_resource_tradeoff.png', bbox_inches='tight')
plt.savefig('results/fig8_embedded_resource_tradeoff.png', bbox_inches='tight')
plt.close()
print('Successfully generated publication-grade Fig 10!')
