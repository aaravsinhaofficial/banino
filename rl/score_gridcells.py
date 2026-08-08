"""Grid-ness of the RL agent's grid-module bottleneck (Fig 2g analog).

Scores the gridcodes_*.npz dumps written during RL training (rolling
buffers of (g, pos, hd) collected while acting) with the same scorer and
paper threshold used for the supervised experiments.

Usage: .venv/bin/python -m rl.score_gridcells rl_runs/grid1 [--arena_half 1.375]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modern.analysis import scoring  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--arena_half', type=float, default=1.375)
  ap.add_argument('--nbins', type=int, default=20)
  args = ap.parse_args()

  paths = sorted(glob.glob(os.path.join(args.run_dir, 'gridcodes_*.npz')))
  if not paths:
    raise SystemExit('no gridcodes dumps in ' + args.run_dir)
  scorer = scoring.GridScorer(
      args.nbins, ((-args.arena_half, args.arena_half),) * 2,
      scoring.default_mask_parameters())
  out = {}
  for path in paths[-1:]:  # newest dump only
    d = np.load(path)
    g = d['g'].astype(np.float32)          # [N, 512]
    pos = d['pos'].astype(np.float32)      # [N, 2]
    flat = scoring.bin_index(pos, nbins=args.nbins, half=args.arena_half)
    counts = np.bincount(flat, minlength=args.nbins ** 2) + 1e-9
    scores60 = np.array([
        scorer.get_scores(scoring.ratemap_from_binned(
            g[:, u], flat, counts, nbins=args.nbins))[0]
        for u in range(g.shape[1])])
    out[os.path.basename(path)] = dict(
        top_grid_60=float(np.nanmax(scores60)),
        mean_grid_60=float(np.nanmean(scores60)),
        pct_above_037=float(100 * np.mean(scores60 > 0.37)),
        n_samples=int(g.shape[0]))
    print(os.path.basename(path), out[os.path.basename(path)])
  with open(os.path.join(args.run_dir, 'grid_scores.json'), 'w') as f:
    json.dump(out, f, indent=1)


if __name__ == '__main__':
  main()
