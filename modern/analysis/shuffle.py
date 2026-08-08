"""Null score distributions via circular time-shifts of the activations.

Trajectories are flattened to one long series per unit; each shuffle rolls
a random unit's activations by >= MIN_SHIFT samples relative to the (fixed)
positions/head directions and rescores. Thresholds = 95th percentile of
each null.

Run:  .venv/bin/python -m modern.analysis.shuffle runs/seed1 --step 300000
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from modern.analysis import scoring

MIN_SHIFT = 1000
_G = {}  # per-process context, inherited via fork


def load_series(run_dir, step):
  """Flattened eval series: acts [T,units] float16, poss [T,2], hds [T]."""
  d = np.load(os.path.join(run_dir, f'acts_{step:06d}.npz'))
  acts = d['acts'].reshape(-1, d['acts'].shape[-1])
  poss = d['poss'].reshape(-1, 2).astype(np.float32)
  hds = d['hds'].reshape(-1).astype(np.float32)
  return acts, poss, hds


def _one_shuffle(task):
  unit, shift = task
  a = np.roll(_G['acts'][:, unit].astype(np.float32), shift)
  rm = scoring.ratemap_from_binned(a, _G['flat'], _G['counts'])
  g60 = _G['scorer'].get_scores(rm)[0]
  return (float(g60), scoring.border_score(rm),
          scoring.hd_resultant(a, _G['hds']))


def _pct95(x):
  x = np.asarray(x, dtype=np.float64)
  return float(np.percentile(x[np.isfinite(x)], 95))


def run(run_dir, step, n_shuffles=1000, workers=None, seed=0):
  """Returns thresholds dict; pools one (unit, shift) sample per shuffle."""
  acts, poss, hds = load_series(run_dir, step)
  flat = scoring.bin_index(poss)
  _G.update(acts=acts, hds=hds, flat=flat,
            counts=np.bincount(flat, minlength=scoring.NBINS**2) + 1e-9,
            scorer=scoring.default_scorer())
  t = acts.shape[0]
  rng = np.random.RandomState(seed)
  tasks = list(zip(rng.randint(0, acts.shape[1], n_shuffles).tolist(),
                   rng.randint(MIN_SHIFT, t - MIN_SHIFT + 1,
                               n_shuffles).tolist()))
  workers = workers or min(40, os.cpu_count())
  with mp.get_context('fork').Pool(workers) as pool:
    out = pool.map(_one_shuffle, tasks,
                   chunksize=max(1, n_shuffles // (workers * 4)))
  g60, border, rv = map(np.asarray, zip(*out))
  return {'grid_60': _pct95(g60), 'border': _pct95(border),
          'hd_rv': _pct95(rv), 'n_shuffles': n_shuffles, 'step': step}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--step', type=int, required=True)
  ap.add_argument('--n_shuffles', type=int, default=1000)
  ap.add_argument('--workers', type=int, default=None)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  t0 = time.time()
  th = run(args.run_dir, args.step, args.n_shuffles, args.workers, args.seed)
  path = os.path.join(args.run_dir, 'shuffle_thresholds.json')
  with open(path, 'w') as f:
    json.dump(th, f, indent=1)
  print(f'wrote {path} in {time.time() - t0:.1f}s: {th}')


if __name__ == '__main__':
  main()
