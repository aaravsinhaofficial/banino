"""Learning curves for the async-A3C cells, against the measured chance line.

Two panels per task: raw episode return vs frames (with the untrained-network
score drawn as a horizontal band) and policy entropy vs frames, which is the
cleanest early indicator that a policy is moving off uniform at all.

Usage: .venv/bin/python -m rl.plot_a3c --rl rl_runs --out report/fig_a3c.png
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

TASKS = {
    'goal': ('a3c_goal_*', 'Goal maze (paper Fig. 3)', 'untrained_control'),
    'arena': ('a3c_arena_*', 'Square arena (paper Fig. 2)', 'untrained_arena'),
    'seekavoid': ('a3c_seekavoid*', 'seekavoid (dense control)',
                  'untrained_seekavoid'),
}
COLORS = {'grid': '#1f77b4', 'placecell': '#d62728', 'a3c': '#2ca02c'}


def load(run_dir):
  mp = os.path.join(run_dir, 'metrics.jsonl')
  if not os.path.exists(mp):
    return None
  rows = []
  for line in open(mp):
    line = line.strip()
    if line:
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        pass
  if not rows:
    return None
  cfg = {}
  cp = os.path.join(run_dir, 'config.json')
  if os.path.exists(cp):
    cfg = json.load(open(cp))
  return rows, cfg


def smooth(y, k=9):
  if len(y) < k:
    return y
  return np.convolve(y, np.ones(k) / k, mode='valid')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--rl', default='rl_runs')
  ap.add_argument('--out', default='report/fig_a3c.png')
  args = ap.parse_args()
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

  tasks = [(k, v) for k, v in TASKS.items()
           if glob.glob(os.path.join(args.rl, v[0]))]
  if not tasks:
    raise SystemExit('no cells found')
  fig, axes = plt.subplots(2, len(tasks), figsize=(5.2 * len(tasks), 7),
                           squeeze=False)

  for col, (_, (pattern, title, chance_dir)) in enumerate(tasks):
    ax_r, ax_e = axes[0][col], axes[1][col]
    chance = None
    cp = os.path.join(args.rl, chance_dir, 'eval_scores.json')
    if os.path.exists(cp):
      chance = json.load(open(cp))
    for run_dir in sorted(glob.glob(os.path.join(args.rl, pattern))):
      if not os.path.isdir(run_dir):
        continue
      got = load(run_dir)
      if not got:
        continue
      rows, cfg = got
      name = os.path.basename(run_dir)
      agent = cfg.get('agent', '?')
      f = np.array([r['frames'] for r in rows]) / 1e6
      ret = np.array([r.get('avg_return_50') or np.nan for r in rows])
      ent = np.array([r.get('ent') or np.nan for r in rows])
      c = COLORS.get(agent, '#7f7f7f')
      ls = '--' if 'hi' in name else '-'
      k = 9
      ax_r.plot(f[k - 1:] if len(f) >= k else f, smooth(ret, k),
                color=c, ls=ls, lw=1.6, label=f'{agent}'
                + (' (lr 2e-4)' if 'hi' in name else ''))
      ax_e.plot(f, ent, color=c, ls=ls, lw=1.4)
    if chance:
      lo = chance['mean_score'] - chance['sem']
      hi = chance['mean_score'] + chance['sem']
      ax_r.axhspan(lo, hi, color='0.6', alpha=0.35, zorder=0)
      ax_r.axhline(chance['mean_score'], color='0.35', lw=1.2, ls=':',
                   label=f"untrained ({chance['mean_score']:.1f})")
    ax_r.set_title(title)
    ax_r.set_ylabel('episode return (raw)')
    ax_r.legend(fontsize=8, frameon=False)
    ax_e.set_ylabel('policy entropy (nats)')
    ax_e.set_xlabel('env frames (millions)')
    for ax in (ax_r, ax_e):
      ax.spines[['top', 'right']].set_visible(False)

  fig.tight_layout()
  fig.savefig(args.out, dpi=140)
  print('wrote', args.out)


if __name__ == '__main__':
  main()
