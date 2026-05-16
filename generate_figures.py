#!/usr/bin/env python3
"""Final production figures — Type 42 fonts, no overlap, clean legends."""
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["figure.dpi"] = 350
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.08
import matplotlib.pyplot as plt

OUT = '/home/claude/arxiv_pkg'
os.makedirs(OUT, exist_ok=True)

def fig1():
    df = pd.read_csv('/home/claude/output_v3/sparsity_results.csv')
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    cfg = {
        'MCAR':            ('#2563eb','o','-'),
        'Cancer-biased':   ('#dc2626','s','--'),
        'Drug-biased':     ('#16a34a','^','-.'),
        'Block(pathway)':  ('#7c3aed','D',':'),
    }
    labels = {'Block(pathway)': 'Pathway-block'}
    for regime in cfg:
        c, m, ls = cfg[regime]
        sub = df[df['regime']==regime].sort_values('sparsity')
        sp = np.concatenate([[0], sub['sparsity'].values])
        delta = np.concatenate([[0], sub['delta'].values])
        sd = np.concatenate([[0], sub['irt_sd'].values])
        lbl = labels.get(regime, regime)
        ax.errorbar(sp*100, delta, yerr=sd, marker=m, color=c, label=lbl,
                    linewidth=1.4, markersize=4.5, capsize=2, capthick=0.7, linestyle=ls)
    ax.set_xlabel('Missing data (%)')
    ax.set_ylabel('$\\Delta\\rho$ (IRT $-$ Averaging)')
    ax.legend(frameon=True, fontsize=7.5, loc='upper left', edgecolor='#d1d5db')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlim(-2, 65); ax.set_ylim(-0.01, 0.13)
    ax.grid(True, alpha=0.15)
    fig.savefig(f'{OUT}/fig1_sparsity.pdf')
    plt.close()

def fig2():
    df = pd.read_csv('/home/claude/output_v3/bootstrap_cis.csv').sort_values('theta', ascending=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    y = np.arange(len(df))
    colors = ['#dc2626' if t>0.5 else '#f97316' if t>0 else
              '#60a5fa' if t>-0.5 else '#1d4ed8' for t in df['theta']]
    ax.barh(y, df['theta'], color=colors, height=0.7, alpha=0.85, edgecolor='none')
    ax.errorbar(df['theta'], y,
                xerr=[df['theta']-df['ci_lo'], df['ci_hi']-df['theta']],
                fmt='none', ecolor='#374151', elinewidth=0.5, capsize=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df['cancer'], fontsize=7)
    ax.set_xlabel('$\\theta$ (In-Vitro Resistance)')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.15)
    ax.set_xlim(-2.8, 1.8)
    fig.savefig(f'{OUT}/fig2_leaderboard.pdf')
    plt.close()

def fig3():
    df = pd.read_csv('/home/claude/output_v3/heldout_prediction.csv')
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = ['#94a3b8','#94a3b8','#475569','#475569','#dc2626']
    ax.barh(np.arange(len(df)), df['brier'], color=colors, height=0.55,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['method'], fontsize=8)
    ax.set_xlabel('Brier Score (lower = better)')
    ax.invert_yaxis()
    for i, v in enumerate(df['brier']):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=7)
    ax.set_xlim(0, 0.16); ax.grid(True, axis='x', alpha=0.15)
    fig.savefig(f'{OUT}/fig3_heldout.pdf')
    plt.close()

def fig4():
    df = pd.read_csv('/home/claude/output_v3/prism_replication.csv')
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    agree = (df['gdsc2_theta']>0)==(df['prism_theta']>0)
    ax.scatter(df.loc[agree,'gdsc2_theta'], df.loc[agree,'prism_theta'],
              c='#2563eb', s=35, zorder=3, edgecolors='white', linewidth=0.5,
              label='Direction agrees')
    ax.scatter(df.loc[~agree,'gdsc2_theta'], df.loc[~agree,'prism_theta'],
              c='#dc2626', s=40, zorder=3, marker='x', linewidths=1.5,
              label='Direction disagrees')
    # Label only extremes + disagreements with manual offsets
    offsets = {
        'PAAD':  (6, -8),  'MESO':  (-35, 5),  'UCEC':  (5, -10),
        'LUAD':  (-30, -10), 'NB': (5, 5), 'HNSC': (5, -10),
        'STAD':  (5, 5),   'GBM':   (5, 5),    'KIRC':  (-32, -8),
    }
    for _, r in df.iterrows():
        if r['cancer'] in offsets:
            ox, oy = offsets[r['cancer']]
            ax.annotate(r['cancer'], (r['gdsc2_theta'], r['prism_theta']),
                       fontsize=6.5, textcoords='offset points', xytext=(ox, oy),
                       arrowprops=dict(arrowstyle='-', color='#9ca3af', lw=0.4))
    lims = [min(df['gdsc2_theta'].min(), df['prism_theta'].min())-0.3,
            max(df['gdsc2_theta'].max(), df['prism_theta'].max())+0.3]
    ax.plot(lims, lims, '--', color='gray', linewidth=0.5, alpha=0.4)
    ax.axhline(0, color='gray', linewidth=0.4, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.4, alpha=0.3)
    from scipy import stats
    rho, p = stats.spearmanr(df['gdsc2_theta'], df['prism_theta'])
    ax.text(0.05, 0.95, f'$\\rho$ = {rho:.3f}, p = {p:.2f}\n{agree.sum()}/{len(df)} direction agree',
           transform=ax.transAxes, fontsize=7.5, va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef3c7', alpha=0.7, edgecolor='#d4a373'))
    ax.set_xlabel('GDSC2 $\\theta$')
    ax.set_ylabel('PRISM $\\theta$')
    ax.legend(fontsize=7, loc='lower right', framealpha=0.8, edgecolor='#d1d5db')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.15)
    fig.savefig(f'{OUT}/fig4_prism.pdf')
    plt.close()

if __name__ == '__main__':
    print("Generating final figures (Type 42, no overlap)...")
    fig1(); print("  Fig 1 ✓")
    fig2(); print("  Fig 2 ✓")
    fig3(); print("  Fig 3 ✓")
    fig4(); print("  Fig 4 ✓")
    # Verify Type 42
    import subprocess
    r = subprocess.run(['pdffonts', f'{OUT}/fig1_sparsity.pdf'],
                       capture_output=True, text=True)
    if 'Type 3' in r.stdout:
        print("  ⚠ Type 3 fonts still present")
    else:
        print("  ✓ No Type 3 fonts detected")
