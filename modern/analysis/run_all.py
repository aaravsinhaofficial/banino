"""Full analysis pipeline: shuffle -> classify -> scales -> figures.

Writes RUN_DIR/summary.json with every headline number (decode errors,
percentages under shuffle + paper thresholds, shuffle thresholds, grid
scale clusters/ratios, top grid score).

Run:  .venv/bin/python -m modern.analysis.run_all runs/seed1 --step 300000
"""

import argparse
import json
import os
import time

from modern.analysis import classify, figures, scales, shuffle


def _decode_errors(run_dir, step):
  """(untrained entry, entry at `step` or latest) from metrics.json."""
  with open(os.path.join(run_dir, 'metrics.json')) as f:
    m = json.load(f)
  fmt = lambda r: {'step': r['step'], 'final_m': r['decode_err_final_m'],
                   'mean_m': r['decode_err_mean_m']}
  untrained = next((r for r in m if r['step'] == 0), None)
  at_step = next((r for r in m if r['step'] == step), m[-1])
  return (fmt(untrained) if untrained else None, fmt(at_step))


def run(run_dir, step, n_shuffles=1000, workers=None, force_shuffle=False):
  timings = {}
  th_path = os.path.join(run_dir, 'shuffle_thresholds.json')
  if force_shuffle or not os.path.exists(th_path):
    t0 = time.time()
    th = shuffle.run(run_dir, step, n_shuffles=n_shuffles, workers=workers)
    with open(th_path, 'w') as f:
      json.dump(th, f, indent=1)
    timings['shuffle'] = round(time.time() - t0, 1)
  with open(th_path) as f:
    th = json.load(f)

  t0 = time.time()
  cls = classify.run(run_dir, step)
  timings['classify'] = round(time.time() - t0, 1)

  t0 = time.time()
  sc = scales.run(run_dir, step)
  timings['scales'] = round(time.time() - t0, 1)

  t0 = time.time()
  figures.make_all(run_dir, step)
  timings['figures'] = round(time.time() - t0, 1)

  untrained, trained = _decode_errors(run_dir, step)
  summary = {
      'run_dir': run_dir,
      'step': step,
      'decode_err_untrained': untrained,
      'decode_err_at_step': trained,
      'shuffle_thresholds': th,
      'pct_shuffle': cls['pct_shuffle'],
      'counts_shuffle': cls['counts_shuffle'],
      'pct_paper': cls['pct_paper'],
      'counts_paper': cls['counts_paper'],
      'top_grid_score': cls['top_grid_score'],
      'n_grid_units_for_scales': sc['n_grid_units'],
      'scale_best_k': sc.get('best_k'),
      'scale_cluster_means_cm': sc.get('cluster_means_cm'),
      'scale_adjacent_ratios': sc.get('adjacent_ratios'),
      'runtimes_s': timings,
  }
  with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
  return summary


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--step', type=int, required=True)
  ap.add_argument('--n_shuffles', type=int, default=1000)
  ap.add_argument('--workers', type=int, default=None)
  ap.add_argument('--force_shuffle', action='store_true')
  args = ap.parse_args()
  summary = run(args.run_dir, args.step, args.n_shuffles, args.workers,
                args.force_shuffle)
  print(json.dumps(summary, indent=1))


if __name__ == '__main__':
  main()
