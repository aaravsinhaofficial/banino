"""Per-unit grid/border/HD scores and classification at a training step.

Grid-60 and border scores come from ratemaps_STEP.npz; the HD resultant
vector from the raw activation/head-direction series. Units are classified
against the run's shuffle thresholds and the paper's published thresholds
(0.37 grid, 0.50 border, 0.47 HD RV).

Run:  .venv/bin/python -m modern.analysis.classify runs/seed1 --step 300000
"""

import argparse
import json
import os

import numpy as np

from modern.analysis import scoring, shuffle

MEASURES = (('grid_60', 'grid'), ('border', 'border'), ('hd_rv', 'hd'))


def hd_rv_all(acts, hds, block=64):
  """Vectorized hd_resultant over all units; acts [T,units] float16."""
  hds = hds.astype(np.float64)
  c, s = np.cos(hds), np.sin(hds)
  out = np.empty(acts.shape[1])
  for i in range(0, acts.shape[1], block):
    w = np.clip(acts[:, i:i + block].astype(np.float64), 0, None)
    den = w.sum(0)
    num = np.hypot(c @ w, s @ w)
    out[i:i + block] = np.where(den > 0, num / np.maximum(den, 1e-300),
                                np.nan)
  return out


def unit_scores(run_dir, step):
  """Score arrays [units] for grid_60, border, hd_rv."""
  ratemaps = np.load(
      os.path.join(run_dir, f'ratemaps_{step:06d}.npz'))['ratemaps']
  scorer = scoring.default_scorer()
  grid_60 = np.array([scorer.get_scores(rm)[0] for rm in ratemaps])
  border = np.array([scoring.border_score(rm) for rm in ratemaps])
  acts, _, hds = shuffle.load_series(run_dir, step)
  return {'grid_60': grid_60, 'border': border,
          'hd_rv': hd_rv_all(acts, hds)}


def classify(scores, thresholds):
  """Counts/percentages of units above threshold per measure (NaN fails)."""
  n_units = len(scores['grid_60'])
  counts, pct = {}, {}
  for key, name in MEASURES:
    with np.errstate(invalid='ignore'):
      c = int(np.sum(scores[key] > thresholds[key]))
    counts[name] = c
    pct[name] = round(100.0 * c / n_units, 2)
  return counts, pct


def run(run_dir, step):
  """Writes unit_scores_STEP.npz + unit_classification.json; returns dict."""
  scores = unit_scores(run_dir, step)
  np.savez(os.path.join(run_dir, f'unit_scores_{step:06d}.npz'), **scores)
  with open(os.path.join(run_dir, 'shuffle_thresholds.json')) as f:
    shuffle_th = json.load(f)
  counts_s, pct_s = classify(scores, shuffle_th)
  counts_p, pct_p = classify(scores, scoring.PAPER_THRESHOLDS)
  result = {
      'step': step,
      'n_units': len(scores['grid_60']),
      'shuffle_thresholds': {k: shuffle_th[k] for k, _ in MEASURES},
      'paper_thresholds': scoring.PAPER_THRESHOLDS,
      'counts_shuffle': counts_s, 'pct_shuffle': pct_s,
      'counts_paper': counts_p, 'pct_paper': pct_p,
      'top_grid_score': float(np.nanmax(scores['grid_60'])),
  }
  with open(os.path.join(run_dir, 'unit_classification.json'), 'w') as f:
    json.dump(result, f, indent=1)
  return result


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--step', type=int, required=True)
  args = ap.parse_args()
  print(json.dumps(run(args.run_dir, args.step), indent=1))


if __name__ == '__main__':
  main()
