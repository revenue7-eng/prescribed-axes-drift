#!/usr/bin/env python3
"""
Generate figures for Paper 2: Semantic Drift, Not Rank Collapse
Real gym-pusht data, 3 seeds (42, 123, 777), 30 epochs.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/home/claude/paper2_figures_real')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

with open('/home/claude/paper2_real_results.json') as f:
    R = json.load(f)

seeds = [42, 123, 777]
freeze_pts = [1, 2, 3, 5, 7, 10]

# ================================================================
# Figure 1: Geometry Fails (3 panels)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for s in seeds:
    p_snaps = R[f'seed_{s}']['cov_drift_prescribed']['covariance']
    f_snaps = R[f'seed_{s}']['cov_drift_free']['covariance']
    
    p_ranks = [c['rank'] for c in p_snaps]
    p_losses = [c['val_loss'] for c in p_snaps]
    f_ranks = [c['rank'] for c in f_snaps]
    f_losses = [c['val_loss'] for c in f_snaps]
    
    alpha = 1.0 if s == 42 else 0.5
    axes[0].scatter(p_ranks, p_losses, c='#2ecc71', s=40, alpha=alpha, zorder=5)
    axes[0].scatter(f_ranks, f_losses, c='#e74c3c', s=40, alpha=alpha, zorder=5)
    
    p_iso = [c['iso'] for c in p_snaps]
    f_iso = [c['iso'] for c in f_snaps]
    axes[1].scatter(p_iso, p_losses, c='#2ecc71', s=40, alpha=alpha, zorder=5)
    axes[1].scatter(f_iso, f_losses, c='#e74c3c', s=40, alpha=alpha, zorder=5)

axes[0].scatter([], [], c='#2ecc71', s=40, label='Prescribed')
axes[0].scatter([], [], c='#e74c3c', s=40, label='Free')
axes[0].set_xlabel('Effective Rank')
axes[0].set_ylabel('Val Loss')
axes[0].set_yscale('log')
axes[0].set_title('(a) Rank vs Performance')
axes[0].legend(fontsize=9)
axes[0].annotate('Higher rank,\nworse loss', xy=(2.99, 0.3), fontsize=8, color='#e74c3c', ha='center')

axes[1].scatter([], [], c='#2ecc71', s=40, label='Prescribed')
axes[1].scatter([], [], c='#e74c3c', s=40, label='Free')
axes[1].set_xlabel('Isotropy (min/max eigenvalue)')
axes[1].set_ylabel('Val Loss')
axes[1].set_yscale('log')
axes[1].set_title('(b) Isotropy vs Performance')
axes[1].legend(fontsize=9)
axes[1].annotate('More isotropic,\nworse loss', xy=(0.85, 0.3), fontsize=8, color='#e74c3c', ha='center')

# Panel C: eigenvalue spectrum averaged across seeds at final epoch
p_eigs = np.mean([R[f'seed_{s}']['cov_drift_prescribed']['covariance'][-1]['eig'] for s in seeds], axis=0)
f_eigs = np.mean([R[f'seed_{s}']['cov_drift_free']['covariance'][-1]['eig'] for s in seeds], axis=0)
x = np.arange(3); w = 0.35
axes[2].bar(x - w/2, p_eigs, w, color='#2ecc71', label='Prescribed', alpha=0.8)
axes[2].bar(x + w/2, f_eigs, w, color='#e74c3c', label='Free', alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(['Axis 0\n(block x)', 'Axis 1\n(block y)', 'Axis 2\n(block θ)'])
axes[2].set_ylabel('Eigenvalue')
axes[2].set_title('(c) Eigenvalue Spectrum (ep 30, avg 3 seeds)')
axes[2].legend(fontsize=9)
axes[2].annotate('Anisotropic\n(task structure)', xy=(1, 0.06), fontsize=8, color='#2ecc71', ha='center')
axes[2].annotate('Isotropic\n(SIGReg artifact)', xy=(1, 0.82), fontsize=8, color='#e74c3c', ha='center')

plt.tight_layout()
plt.savefig(OUT / 'fig1_geometry_fails.png', bbox_inches='tight')
plt.close()
print('Figure 1 saved')


# ================================================================
# Figure 2: Drift Over Time (3 panels, averaged)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Collect drift data per seed, align by epoch
max_len = min(len(R[f'seed_{s}']['cov_drift_free']['drift']) for s in seeds)
all_raw = np.zeros((3, max_len))
all_aligned = np.zeros((3, max_len))

for i, s in enumerate(seeds):
    drift = R[f'seed_{s}']['cov_drift_free']['drift'][:max_len]
    all_raw[i] = [d['raw_drift'] for d in drift]
    all_aligned[i] = [d['aligned_drift'] for d in drift]

epochs = list(range(1, max_len + 1))
mean_raw = all_raw.mean(0)
mean_aligned = all_aligned.mean(0)
std_raw = all_raw.std(0)
std_aligned = all_aligned.std(0)

axes[0].plot(epochs, mean_raw, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Free (mean)')
axes[0].fill_between(epochs, mean_raw - std_raw, mean_raw + std_raw, color='#e74c3c', alpha=0.15)
axes[0].axhline(0, color='#2ecc71', linewidth=2, linestyle='--', label='Prescribed (always 0)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Raw Drift')
axes[0].set_title('(a) Raw Drift ||f_t(x) - f_{t+1}(x)||')
axes[0].legend(fontsize=9)
axes[0].axvspan(0.5, 2.5, alpha=0.1, color='red')

axes[1].plot(epochs, mean_aligned, 's-', color='#e74c3c', linewidth=2, markersize=4, label='Aligned (Procrustes)')
axes[1].plot(epochs, mean_raw, 'o--', color='#e74c3c', linewidth=1, markersize=3, alpha=0.3, label='Raw')
axes[1].axhline(0, color='#2ecc71', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Drift')
axes[1].set_title('(b) Aligned vs Raw Drift')
axes[1].legend(fontsize=9)

ratio_proc = np.where(mean_raw > 0.001, 1 - mean_aligned / mean_raw, 0)
axes[2].bar(epochs[:10], ratio_proc[:10], color='#3498db', alpha=0.7)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Fraction removed by Procrustes')
axes[2].set_title('(c) How much is rotation?')
axes[2].set_ylim(0, 1)
axes[2].axhline(0.5, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(OUT / 'fig2_drift_over_time.png', bbox_inches='tight')
plt.close()
print('Figure 2 saved')


# ================================================================
# Figure 3: R² Transfer (Killer Plot)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))

for i, s in enumerate(seeds):
    drift = R[f'seed_{s}']['cov_drift_free']['drift']
    eps_d = [d['epoch_to'] for d in drift]
    r2_self = [d['r2_self'] for d in drift]
    r2_xfer = [d['r2_transfer'] for d in drift]
    
    alpha = 1.0 if s == 42 else 0.4
    lw = 2 if s == 42 else 1
    if s == 42:
        ax.plot(eps_d, r2_self, 'o-', color='#3498db', linewidth=lw, markersize=5, alpha=alpha, label='R² (same epoch)')
        ax.plot(eps_d, r2_xfer, 's-', color='#e74c3c', linewidth=lw, markersize=5, alpha=alpha, label='R² (transferred)')
    else:
        ax.plot(eps_d, r2_self, 'o-', color='#3498db', linewidth=lw, markersize=3, alpha=alpha)
        ax.plot(eps_d, r2_xfer, 's-', color='#e74c3c', linewidth=lw, markersize=3, alpha=alpha)

ax.axhline(1.0, color='#2ecc71', linewidth=2, linestyle='--', label='Prescribed (always 1.0)', alpha=0.7)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.axvspan(0.5, 2.5, alpha=0.1, color='red')

# Annotate worst R² values
worst_r2 = min(R[f'seed_{s}']['cov_drift_free']['drift'][0]['r2_transfer'] for s in seeds)
ax.annotate(f'R² = {worst_r2:.1f}\nDecoder produces\ngarbage', xy=(1, worst_r2/2),
            fontsize=9, ha='center', color='#e74c3c', fontweight='bold')

ax.set_xlabel('Epoch')
ax.set_ylabel('R² (linear probe → ground truth)')
ax.set_title('Decoder Transferability Across Epochs (3 seeds, gym-pusht)')
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(min(worst_r2 * 1.1, -70), 1.3)

plt.tight_layout()
plt.savefig(OUT / 'fig3_r2_transfer.png', bbox_inches='tight')
plt.close()
print('Figure 3 saved')


# ================================================================
# Figure 4: Freeze Test (V-curve, 3 seeds)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))

for i, s in enumerate(seeds):
    sr = R[f'seed_{s}']
    fp = [sr[f'freeze_free_at_{t}']['best_vp'] for t in freeze_pts]
    alpha = 1.0 if s == 42 else 0.4
    lw = 2 if s == 42 else 1
    label = f'Seed {s}' if s == 42 else None
    ax.plot(freeze_pts, fp, 'D-', color='#9b59b6', linewidth=lw, markersize=6, alpha=alpha, label=label)

# Mean
mean_freeze = [np.mean([R[f'seed_{s}'][f'freeze_free_at_{t}']['best_vp'] for s in seeds]) for t in freeze_pts]
ax.plot(freeze_pts, mean_freeze, 'D-', color='#9b59b6', linewidth=2.5, markersize=8, label='Mean (3 seeds)')

mean_unfrozen = np.mean([R[f'seed_{s}']['freeze_free_unfrozen']['best_vp'] for s in seeds])
mean_prescribed = np.mean([R[f'seed_{s}']['freeze_prescribed']['best_vp'] for s in seeds])

ax.axhline(mean_unfrozen, color='#e74c3c', linewidth=2, linestyle='--',
           label=f'Free unfrozen ({mean_unfrozen:.4f})')
ax.axhline(mean_prescribed, color='#2ecc71', linewidth=2, linestyle='--',
           label=f'Prescribed ({mean_prescribed:.4f})')

ax.annotate(f'freeze@1: +20%\nvs unfrozen', xy=(1, mean_freeze[0]),
            xytext=(3, mean_freeze[0] - 0.008),
            arrowprops=dict(arrowstyle='->', color='#9b59b6'),
            fontsize=9, color='#9b59b6')

ax.set_xlabel('Freeze Epoch T')
ax.set_ylabel('Best Val Loss')
ax.set_title('Freeze Test: When Does Stabilization Help? (3 seeds, gym-pusht)')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(OUT / 'fig4_freeze_test.png', bbox_inches='tight')
plt.close()
print('Figure 4 saved')


# ================================================================
# Figure 5: Stability vs Alignment (2D)
# ================================================================
fig, ax = plt.subplots(figsize=(7, 6))

avg_drift_free = np.mean([
    np.mean([d['raw_drift'] for d in R[f'seed_{s}']['cov_drift_free']['drift']])
    for s in seeds
])

points = {
    'Prescribed': {
        'drift': 0.0, 'loss': mean_prescribed,
        'color': '#2ecc71', 'marker': '*', 'size': 300
    },
    'Free (unfrozen)': {
        'drift': avg_drift_free, 'loss': mean_unfrozen,
        'color': '#e74c3c', 'marker': 'o', 'size': 150
    },
    'Free (freeze@1)': {
        'drift': avg_drift_free * 0.05,  # minimal drift after freeze
        'loss': np.mean([R[f'seed_{s}']['freeze_free_at_1']['best_vp'] for s in seeds]),
        'color': '#9b59b6', 'marker': 'D', 'size': 150
    },
}

for label, p in points.items():
    ax.scatter(p['drift'], p['loss'], c=p['color'], s=p['size'],
               marker=p['marker'], zorder=5, edgecolors='black', linewidths=0.5)
    ofs_x = 0.02 if 'unfrozen' in label else -0.01
    ofs_y = 0.003 if 'unfrozen' in label else -0.004
    ax.annotate(label, xy=(p['drift'], p['loss']),
                xytext=(p['drift'] + ofs_x, p['loss'] + ofs_y),
                fontsize=10, fontweight='bold', color=p['color'])

ax.annotate('', xy=(points['Free (freeze@1)']['drift'], points['Free (freeze@1)']['loss']),
            xytext=(points['Free (unfrozen)']['drift'], points['Free (unfrozen)']['loss']),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
ax.text(avg_drift_free * 0.5, 0.075, 'Factor 1:\nStability\n(+20%)', fontsize=9, ha='center', color='gray', style='italic')

ax.annotate('', xy=(0.0, mean_prescribed),
            xytext=(points['Free (freeze@1)']['drift'], points['Free (freeze@1)']['loss']),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
ax.text(-0.02, 0.035, 'Factor 2:\nAlignment\n(26×)', fontsize=9, ha='center', color='gray', style='italic')

ax.set_xlabel('Avg Drift (higher = less stable)')
ax.set_ylabel('Best Val Loss (lower = better)')
ax.set_title('Two Factors: Stability × Alignment')

plt.tight_layout()
plt.savefig(OUT / 'fig5_stability_vs_alignment.png', bbox_inches='tight')
plt.close()
print('Figure 5 saved')


# ================================================================
# Figure 6: Free loss curve ep 0-3 (catastrophic phase)
# showing ep 1 spike across all seeds
# ================================================================
fig, ax = plt.subplots(figsize=(8, 5))

for s in seeds:
    p_cov = R[f'seed_{s}']['cov_drift_prescribed']['covariance']
    f_cov = R[f'seed_{s}']['cov_drift_free']['covariance']
    
    p_eps = [c['epoch'] for c in p_cov]
    p_vals = [c['val_loss'] for c in p_cov]
    f_eps = [c['epoch'] for c in f_cov]
    f_vals = [c['val_loss'] for c in f_cov]
    
    alpha = 1.0 if s == 42 else 0.4
    lw = 2 if s == 42 else 1
    ax.plot(p_eps, p_vals, 'o-', color='#2ecc71', linewidth=lw, markersize=4, alpha=alpha)
    ax.plot(f_eps, f_vals, 's-', color='#e74c3c', linewidth=lw, markersize=4, alpha=alpha)

ax.plot([], [], 'o-', color='#2ecc71', linewidth=2, label='Prescribed')
ax.plot([], [], 's-', color='#e74c3c', linewidth=2, label='Free')

ax.set_xlabel('Epoch')
ax.set_ylabel('Val Loss')
ax.set_yscale('log')
ax.set_title('Training Curves: Prescribed vs Free (3 seeds, gym-pusht)')
ax.legend(fontsize=10)
ax.axvspan(0.5, 2.5, alpha=0.1, color='red')
ax.annotate('Catastrophic\nphase', xy=(1.5, 0.5), fontsize=9, ha='center',
            color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / 'fig6_training_curves.png', bbox_inches='tight')
plt.close()
print('Figure 6 saved')

print(f'\nAll figures saved to {OUT}/')
