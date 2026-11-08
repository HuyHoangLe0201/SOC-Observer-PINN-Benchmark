# -*- coding: utf-8 -*-
tex_path = 'manuscript/manuscript_jeecs.tex'
with open(tex_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Adjust figure scaling to reduce float congestion on Pages 9 and 10
replacements = [
    (r"\includegraphics[width=\linewidth]{../results/fig6_lfp_plateau_tracking.png}",
     r"\includegraphics[width=0.85\linewidth]{../results/fig6_lfp_plateau_tracking.png}"),
    (r"\includegraphics[width=\linewidth]{../results/fig7_sensor_bias_drift_stress.png}",
     r"\includegraphics[width=0.85\linewidth]{../results/fig7_sensor_bias_drift_stress.png}"),
    (r"\includegraphics[width=\linewidth]{../results/fig8_eis_nyquist_aging.png}",
     r"\includegraphics[width=0.80\linewidth]{../results/fig8_eis_nyquist_aging.png}"),
    (r"\includegraphics[width=\linewidth]{../results/fig9_pack_cell_inconsistency.png}",
     r"\includegraphics[width=0.75\linewidth]{../results/fig9_pack_cell_inconsistency.png}"),
    (r"\includegraphics[width=0.60\linewidth]{../results/fig10_embedded_resource_tradeoff.png}",
     r"\includegraphics[width=0.52\linewidth]{../results/fig10_embedded_resource_tradeoff.png}"),
    (r"\includegraphics[width=0.60\linewidth]{../results/fig11_pidkf_hybrid_benchmark.png}",
     r"\includegraphics[width=0.52\linewidth]{../results/fig11_pidkf_hybrid_benchmark.png}")
]

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"Replaced {old[:30]}...")
    else:
        print(f"NOT found: {old[:30]}...")

with open(tex_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(text)
print("Figure scaling updated successfully!")
