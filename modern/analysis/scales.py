"""Grid scales from SAC peak spacing + GMM clustering into modules.

Scale = mean distance (bins -> cm, 1 bin = 11 cm) from the SAC centre to
the 6 nearest local maxima. Scales of grid-classified units are clustered
with GaussianMixture (1..4 components, BIC selection); the paper finds
modules at ~47/70/106 cm (adjacent ratio ~1.5).

Run:  .venv/bin/python -m modern.analysis.scales runs/seed1 --step 300000
"""

import argparse
import json
import os

import numpy as np
import scipy.ndimage
from sklearn.mixture import GaussianMixture

from modern.analysis import scoring


def sac_grid_scale_cm(sac, nbins=scoring.NBINS):
  """Mean distance (cm) from SAC centre to its 6 nearest local maxima.

  Peaks: 3x3 maximum_filter, positive correlation, within the analysed
  disc of radius nbins; the centre peak is excluded. NaN if no peaks.
  """
  peaks = (sac == scipy.ndimage.maximum_filter(sac, size=3)) & (sac > 0)
  c = nbins - 1
  ii, jj = np.indices(sac.shape)
  d = np.hypot(ii - c, jj - c)
  peaks &= (d <= nbins) & (d > 0)
  if not peaks.any():
    return np.nan
  return float(np.mean(np.sort(d[peaks])[:6]) * scoring.BIN_CM)


def fit_gmm(scales_cm, max_k=4, seed=0):
  """BIC-selected GaussianMixture over the scale distribution.

  reg_covar floors component std at 1 cm so near-duplicate scales cannot
  collapse a component to zero width.
  """
  x = np.asarray(scales_cm, dtype=np.float64)[:, None]
  bic, models = {}, {}
  for k in range(1, min(max_k, len(x)) + 1):
    gm = GaussianMixture(n_components=k, n_init=10, reg_covar=1.0,
                         random_state=seed).fit(x)
    bic[k] = float(gm.bic(x))
    models[k] = gm
  best_k = min(bic, key=bic.get)
  gm = models[best_k]
  order = np.argsort(gm.means_[:, 0])
  means = gm.means_[order, 0]
  return {
      'bic': {str(k): round(v, 2) for k, v in bic.items()},
      'best_k': best_k,
      'cluster_means_cm': [round(float(m), 2) for m in means],
      'cluster_stds_cm': [round(float(np.sqrt(v)), 2) for v in
                          gm.covariances_[order, 0, 0]],
      'cluster_weights': [round(float(w), 3) for w in gm.weights_[order]],
      'adjacent_ratios': [round(float(means[i + 1] / means[i]), 3)
                          for i in range(len(means) - 1)],
  }


def run(run_dir, step, threshold_set='shuffle'):
  """Writes scale_analysis.json for grid-classified units; returns dict."""
  scores = np.load(os.path.join(run_dir, f'unit_scores_{step:06d}.npz'))
  if threshold_set == 'paper':
    thresh = scoring.PAPER_THRESHOLDS['grid_60']
  else:
    with open(os.path.join(run_dir, 'shuffle_thresholds.json')) as f:
      thresh = json.load(f)['grid_60']
  g60 = scores['grid_60']
  with np.errstate(invalid='ignore'):
    units = np.flatnonzero(g60 > thresh)
  ratemaps = np.load(
      os.path.join(run_dir, f'ratemaps_{step:06d}.npz'))['ratemaps']
  scorer = scoring.default_scorer()
  scales = {int(u): sac_grid_scale_cm(scorer.calculate_sac(ratemaps[u]))
            for u in units}
  valid = np.array([s for s in scales.values() if np.isfinite(s)])
  result = {
      'step': step,
      'threshold_set': threshold_set,
      'grid_threshold': float(thresh),
      'n_grid_units': int(len(units)),
      'unit_ids': [int(u) for u in units],
      'scales_cm': {u: round(s, 2) for u, s in scales.items()
                    if np.isfinite(s)},
  }
  if len(valid) >= 2:
    result.update(fit_gmm(valid))
  with open(os.path.join(run_dir, 'scale_analysis.json'), 'w') as f:
    json.dump(result, f, indent=1)
  return result


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run_dir')
  ap.add_argument('--step', type=int, required=True)
  ap.add_argument('--threshold_set', choices=['shuffle', 'paper'],
                  default='shuffle')
  args = ap.parse_args()
  print(json.dumps(run(args.run_dir, args.step, args.threshold_set),
                   indent=1))


if __name__ == '__main__':
  main()
