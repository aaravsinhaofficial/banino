"""Aggregate all reproduction results into one paper-vs-ours table.

Collects: per-seed supervised summaries (runs/seedN/summary.json), the
legacy TF1 run's logged grid counts (results/, docker log text passed in),
and RL run outputs (rl_runs/*/grid_scores.json, goal_decode_*.json,
metrics.jsonl). Writes REPORT.json and a markdown table to stdout/file.

Usage: .venv/bin/python -m modern.analysis.aggregate --runs runs --rl rl_runs \
    --out REPORT.json --md REPORT.md
"""

import argparse
import glob
import json
import os

import numpy as np

PAPER = {
    'decode_err_trained_cm': 16.0,
    'decode_err_untrained_cm': 91.0,
    'pct_grid': 25.2,
    'pct_border': 10.2,
    'pct_hd': 8.7,
    'grid_threshold': 0.37,
    'scale_ratio': 1.5,
    'scale_means_cm': [47.0, 70.0, 106.0],
    'rl_agent_grid_pct': 21.4,
}


def collect_supervised(runs_dir):
  out = []
  for sj in sorted(glob.glob(os.path.join(runs_dir, 'seed*/summary.json'))):
    with open(sj) as f:
      s = json.load(f)
    s['run'] = os.path.basename(os.path.dirname(sj))
    out.append(s)
  return out


def collect_rl(rl_dir):
  out = {}
  for run in sorted(glob.glob(os.path.join(rl_dir, '*'))):
    if not os.path.isdir(run):
      continue
    r = {}
    for name, fn in [('grid_scores', 'grid_scores.json'),
                     ('goal_decode', 'goal_decode_grid.json')]:
      p = os.path.join(run, fn)
      if os.path.exists(p):
        with open(p) as f:
          r[name] = json.load(f)
    mp = os.path.join(run, 'metrics.jsonl')
    if os.path.exists(mp):
      lines = [json.loads(l) for l in open(mp) if l.strip()]
      if lines:
        r['final'] = lines[-1]
        r['returns_tail'] = [l.get('avg_return_50') for l in lines[-20:]]
    if r:
      out[os.path.basename(run)] = r
  return out


def fmt_mean_sd(vals, scale=1.0):
  vals = [v * scale for v in vals if v is not None]
  if not vals:
    return 'n/a'
  if len(vals) == 1:
    return f'{vals[0]:.1f}'
  return f'{np.mean(vals):.1f} ± {np.std(vals):.1f} (n={len(vals)})'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--runs', default='runs')
  ap.add_argument('--rl', default='rl_runs')
  ap.add_argument('--out', default='REPORT.json')
  ap.add_argument('--md', default=None)
  args = ap.parse_args()

  sup = collect_supervised(args.runs)
  rl = collect_rl(args.rl)
  report = {'paper': PAPER, 'supervised_seeds': sup, 'rl_runs': rl}
  with open(args.out, 'w') as f:
    json.dump(report, f, indent=1)

  def g(*path):
    vals = []
    for s in sup:
      v = s
      for k in path:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
          break
      vals.append(v)
    return vals

  ratios = [np.mean(r) if r else None
            for r in g('scale_adjacent_ratios')]
  lines = [
      '| Result | Paper | This reproduction |',
      '|---|---|---|',
      f'| Decode error, trained (cm) | 16 | '
      f'{fmt_mean_sd(g("decode_err_at_step", "final_m"), 100)} |',
      f'| Decode error, untrained (cm) | 91 | '
      f'{fmt_mean_sd(g("decode_err_untrained", "final_m"), 100)} |',
      f'| % grid units (paper threshold 0.37) | 25.2 | '
      f'{fmt_mean_sd(g("pct_paper", "grid"))} |',
      f'| % grid units (own shuffle threshold) | — | '
      f'{fmt_mean_sd(g("pct_shuffle", "grid"))} |',
      f'| % border units (0.50) | 10.2 | '
      f'{fmt_mean_sd(g("pct_paper", "border"))} |',
      f'| % HD units (0.47) | 8.7 | {fmt_mean_sd(g("pct_paper", "hd"))} |',
      f'| Top grid score | — | {fmt_mean_sd(g("top_grid_score"))} |',
      f'| Grid scale ratio | ~1.5 | {fmt_mean_sd(ratios)} |',
  ]
  for name, r in rl.items():
    gs = r.get('grid_scores', {})
    if gs:
      newest = list(gs.values())[-1]
      lines.append(f'| RL {name}: % grid in grid module | 21.4 | '
                   f'{newest.get("pct_above_037", float("nan")):.1f} |')
    gd = r.get('goal_decode', {})
    if 'r2_dist' in gd:
      lines.append(f'| RL {name}: goal-dist decode R² (shuffled ctrl) | '
                   f'sig. | {gd["r2_dist"]:.2f} ({gd["r2_dist_shuffled"]:.2f}) |')
  md = '\n'.join(lines)
  print(md)
  if args.md:
    with open(args.md, 'w') as f:
      f.write(md + '\n')


if __name__ == '__main__':
  main()
