"""Publication-style figures for a run, written to RUN_DIR/figures/.

(a) decode-error curve, (b) top-32 grid units (ratemap|SAC, jet, as in
paper Fig 1d), (c) grid-scale histogram + GMM overlay (Fig 1e analog),
(d) per-measure score distributions with threshold lines.

Run:  .venv/bin/python -m modern.analysis.figures runs/seed1 --step 300000
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from modern.analysis import scoring

BLUE, ORANGE = '#2563EB', '#E8590C'
INK, MUTED = '#1f2937', '#6b7280'
MEASURE_LABELS = {'grid_60': 'grid score (60)', 'border': 'border score',
                  'hd_rv': 'HD resultant vector length'}


def _style(ax):
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  ax.grid(True, color='#e5e7eb', linewidth=0.6)
  ax.set_axisbelow(True)


def fig_decode_error(run_dir, out):
  with open(os.path.join(run_dir, 'metrics.json')) as f:
    m = json.load(f)
  steps = [r['step'] for r in m]
  fig, ax = plt.subplots(figsize=(6, 4))
  _style(ax)
  for key, color, label in (
      ('decode_err_final_m', BLUE, 'final timestep'),
      ('decode_err_mean_m', ORANGE, 'mean over timesteps')):
    ax.plot(steps, [r[key] * 100 for r in m], color=color, lw=2, marker='o',
            ms=4, label=label)
  for y, txt in ((91, 'paper untrained (91 cm)'), (16, 'paper trained (16 cm)')):
    ax.axhline(y, color=MUTED, lw=1, ls='--')
    ax.annotate(txt, (0.99, y), xycoords=('axes fraction', 'data'),
                ha='right', va='bottom', fontsize=8, color=MUTED)
  ax.set_xlabel('training step')
  ax.set_ylabel('position decode error (cm)')
  ax.set_title('Position decoding error', color=INK)
  ax.legend(frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(out, dpi=150)
  plt.close(fig)


def fig_grid_cells(run_dir, step, out, top_n=32):
  ratemaps = np.load(
      os.path.join(run_dir, f'ratemaps_{step:06d}.npz'))['ratemaps']
  g60 = np.load(
      os.path.join(run_dir, f'unit_scores_{step:06d}.npz'))['grid_60']
  order = np.argsort(np.nan_to_num(g60, nan=-np.inf))[::-1][:top_n]
  scorer = scoring.default_scorer()
  rows = (top_n + 3) // 4
  fig, axes = plt.subplots(rows, 8, figsize=(14, 1.9 * rows))
  for ax in axes.flat:
    ax.axis('off')
  for i, u in enumerate(order):
    r, c = i // 4, (i % 4) * 2
    sac = scorer.calculate_sac(ratemaps[u])
    axes[r, c].imshow(ratemaps[u], cmap='jet', interpolation='none')
    axes[r, c].set_title(f'u{u}  g={g60[u]:.2f}', fontsize=7)
    axes[r, c + 1].imshow(sac * scorer.plotting_sac_mask, cmap='jet',
                          interpolation='none')
  fig.suptitle(f'Top {top_n} grid-scoring units (ratemap | SAC), '
               f'step {step}', color=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out, dpi=150)
  plt.close(fig)


def fig_scales(run_dir, out):
  with open(os.path.join(run_dir, 'scale_analysis.json')) as f:
    sa = json.load(f)
  scales = np.array(list(sa.get('scales_cm', {}).values()), dtype=np.float64)
  if scales.size == 0:
    return False
  fig, ax = plt.subplots(figsize=(6, 4))
  _style(ax)
  bins = np.linspace(scales.min() - 5, scales.max() + 5, 24)
  ax.hist(scales, bins=bins, color=BLUE, alpha=0.75, edgecolor='white')
  if 'cluster_means_cm' in sa:
    means = np.array(sa['cluster_means_cm'])
    stds = np.array(sa['cluster_stds_cm'])
    weights = np.array(sa['cluster_weights'])
    xs = np.linspace(bins[0], bins[-1], 400)
    pdf = np.sum(weights[:, None] / (stds[:, None] * np.sqrt(2 * np.pi)) *
                 np.exp(-0.5 * ((xs - means[:, None]) / stds[:, None])**2), 0)
    ax.plot(xs, pdf * scales.size * (bins[1] - bins[0]), color=ORANGE, lw=2,
            label=f'GMM (k={sa["best_k"]}, BIC)')
    for mu in means:
      ax.axvline(mu, color=MUTED, lw=1, ls='--')
      ax.annotate(f'{mu:.0f}', (mu, 0.97), xycoords=('data', 'axes fraction'),
                  ha='center', fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=9)
  ax.set_xlabel('grid scale (cm)')
  ax.set_ylabel('units')
  ax.set_title(f'Grid scales, {sa["n_grid_units"]} grid units', color=INK)
  fig.tight_layout()
  fig.savefig(out, dpi=150)
  plt.close(fig)
  return True


def fig_score_distributions(run_dir, step, out):
  scores = np.load(os.path.join(run_dir, f'unit_scores_{step:06d}.npz'))
  with open(os.path.join(run_dir, 'shuffle_thresholds.json')) as f:
    shuffle_th = json.load(f)
  fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
  for ax, key in zip(axes, ('grid_60', 'border', 'hd_rv')):
    _style(ax)
    v = scores[key]
    v = v[np.isfinite(v)]
    ax.hist(v, bins=40, color=BLUE, alpha=0.75, edgecolor='white')
    ax.axvline(shuffle_th[key], color=INK, lw=1.5,
               label=f'shuffle 95th ({shuffle_th[key]:.2f})')
    ax.axvline(scoring.PAPER_THRESHOLDS[key], color=MUTED, lw=1.5, ls='--',
               label=f'paper ({scoring.PAPER_THRESHOLDS[key]:.2f})')
    ax.set_xlabel(MEASURE_LABELS[key])
    ax.set_ylabel('units')
    ax.legend(frameon=False, fontsize=8)
  fig.suptitle(f'Score distributions across 512 units, step {step}',
               color=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(out, dpi=150)
  plt.close(fig)


def make_all(run_dir, step):
  """Writes all four figures; returns list of paths written."""
  fdir = os.path.join(run_dir, 'figures')
  os.makedirs(fdir, exist_ok=True)
  written = []
  p = os.path.join(fdir, 'decode_error.png')
  fig_decode_error(run_dir, p)
  written.append(p)
  p = os.path.join(fdir, f'grid_cells_{step:06d}.png')
  fig_grid_cells(run_dir, step, p)
  written.append(p)
  p = os.path.join(fdir, f'grid_scales_{step:06d}.png')
  if fig_scales(run_dir, p):
    written.append(p)
  p = os.path.join(fdir, f'score_distributions_{step:06d}.png')
  fig_score_distributions(run_dir, step, p)
  written.append(p)
  return written


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--step', type=int, required=True)
  args = ap.parse_args()
  for p in make_all(args.run_dir, args.step):
    print(p)


if __name__ == '__main__':
  main()
