"""Spatial-tuning scores: grid (SAC), border (Solstad 2008), head direction.

GridScorer is a faithful numpy port of the TF1-era scores.py (no TF, no
matplotlib import needed). Border and HD scores follow Banino et al. 2018
Methods.
"""

import math

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats

NBINS = 20
COORDS_RANGE = ((-1.1, 1.1), (-1.1, 1.1))
BIN_CM = 11.0  # 2.2 m arena / 20 bins
PAPER_THRESHOLDS = {'grid_60': 0.37, 'border': 0.50, 'hd_rv': 0.47}


def default_mask_parameters():
  """Ring masks of the original repo: starts 0.2 (x10), ends 0.4..1.0."""
  starts = [0.2] * 10
  ends = np.linspace(0.4, 1.0, num=10)
  return list(zip(starts, ends.tolist()))


def default_scorer():
  return GridScorer(NBINS, COORDS_RANGE, default_mask_parameters())


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
  """Circular mask; same math as scores.py."""
  sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
  x = np.linspace(-sz[0], sz[1], size[1])
  x = np.expand_dims(x, 0).repeat(size[0], 0)
  y = np.linspace(-sz[0], sz[1], size[1])
  y = np.expand_dims(y, 1).repeat(size[1], 1)
  z = np.sqrt(x**2 + y**2)
  return np.where(np.less_equal(z, radius), in_val, out_val)


class GridScorer(object):
  """Grid scores from ratemaps; port of scores.py GridScorer."""

  def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
    self._nbins = nbins
    self._min_max = min_max
    self._coords_range = coords_range
    self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
    self._masks = [(self._get_ring_mask(mask_min, mask_max),
                    (mask_min, mask_max))
                   for mask_min, mask_max in mask_parameters]
    # Mask hiding the parts of the SAC that are never used (for plotting).
    self.plotting_sac_mask = circle_mask(
        [self._nbins * 2 - 1, self._nbins * 2 - 1], self._nbins,
        in_val=1.0, out_val=np.nan)

  def calculate_ratemap(self, xs, ys, activations, statistic='mean'):
    return scipy.stats.binned_statistic_2d(
        xs, ys, activations, bins=self._nbins, statistic=statistic,
        range=self._coords_range)[0]

  def _get_ring_mask(self, mask_min, mask_max):
    n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
    return (circle_mask(n_points, mask_max * self._nbins) *
            (1 - circle_mask(n_points, mask_min * self._nbins)))

  def grid_score_60(self, corr):
    if self._min_max:
      return np.minimum(corr[60], corr[120]) - np.maximum(
          corr[30], np.maximum(corr[90], corr[150]))
    return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

  def grid_score_90(self, corr):
    return corr[90] - (corr[45] + corr[135]) / 2

  def calculate_sac(self, seq1):
    """Spatial autocorrelogram (normalized 2D autocorrelation)."""
    seq2 = seq1

    def filter2(b, x):
      stencil = np.rot90(b, 2)
      return scipy.signal.convolve2d(x, stencil, mode='full')

    seq1 = np.nan_to_num(seq1)
    seq2 = np.nan_to_num(seq2)

    ones_seq1 = np.ones(seq1.shape)
    ones_seq2 = np.ones(seq2.shape)

    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)

    seq1_x_seq2 = filter2(seq1, seq2)
    sum_seq1 = filter2(seq1, ones_seq2)
    sum_seq2 = filter2(ones_seq1, seq2)
    sum_seq1_sq = filter2(seq1_sq, ones_seq2)
    sum_seq2_sq = filter2(ones_seq1, seq2_sq)
    n_bins = filter2(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)

    with np.errstate(invalid='ignore', divide='ignore'):
      std_seq1 = np.power(
          np.subtract(np.divide(sum_seq1_sq, n_bins),
                      np.divide(np.square(sum_seq1), n_bins_sq)), 0.5)
      std_seq2 = np.power(
          np.subtract(np.divide(sum_seq2_sq, n_bins),
                      np.divide(np.square(sum_seq2), n_bins_sq)), 0.5)
      covar = np.subtract(
          np.divide(seq1_x_seq2, n_bins),
          np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
      x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
    x_coef = np.real(x_coef)
    x_coef = np.nan_to_num(x_coef)
    return x_coef

  def rotated_sacs(self, sac, angles):
    return [scipy.ndimage.rotate(sac, angle, reshape=False)
            for angle in angles]

  def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
    """Pearson correlations of area inside mask at corr_angles."""
    masked_sac = sac * mask
    ring_area = np.sum(mask)
    masked_sac_mean = np.sum(masked_sac) / ring_area
    masked_sac_centered = (masked_sac - masked_sac_mean) * mask
    variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
    corrs = dict()
    for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
      masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
      cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
      corrs[angle] = cross_prod / variance
    return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

  def get_scores(self, rate_map):
    """Returns (score_60, score_90, mask_params_60, mask_params_90, sac)."""
    sac = self.calculate_sac(rate_map)
    rotated_sacs = self.rotated_sacs(sac, self._corr_angles)
    scores = [self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
              for mask, _ in self._masks]
    scores_60, scores_90, _ = map(np.asarray, zip(*scores))
    max_60_ind = np.argmax(scores_60)
    max_90_ind = np.argmax(scores_90)
    return (scores_60[max_60_ind], scores_90[max_90_ind],
            self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac)


def bin_index(poss, nbins=NBINS, half=1.1):
  """Flat spatial bin index per sample; matches train_supervised binning."""
  ix = np.clip(((poss[:, 0] + half) / (2 * half) * nbins).astype(int),
               0, nbins - 1)
  iy = np.clip(((poss[:, 1] + half) / (2 * half) * nbins).astype(int),
               0, nbins - 1)
  return ix * nbins + iy


def ratemap_from_binned(acts, flat, counts=None, nbins=NBINS):
  """Mean activation per bin given precomputed flat bin indices."""
  if counts is None:
    counts = np.bincount(flat, minlength=nbins * nbins) + 1e-9
  sums = np.bincount(flat, weights=acts, minlength=nbins * nbins)
  return (sums / counts).reshape(nbins, nbins)


def border_score(ratemap, field_thresh=0.3, min_field_bins=6):
  """Solstad et al. 2008 border score: (cM - dm) / (cM + dm).

  Fields = connected components of bins > field_thresh * max, discarding
  fields under min_field_bins bins. cM = max over the 4 walls of the
  fraction of that wall's bins covered by a single field; dm = activation-
  weighted mean distance of field bins to the nearest wall, normalized by
  half the arena. NaN if no fields (or non-positive max).
  """
  rm = np.nan_to_num(np.asarray(ratemap, dtype=np.float64))
  mx = rm.max()
  if mx <= 0:
    return np.nan
  labels, n = scipy.ndimage.label(rm > field_thresh * mx)
  fields = [labels == i for i in range(1, n + 1)]
  fields = [f for f in fields if f.sum() >= min_field_bins]
  if not fields:
    return np.nan
  nb = rm.shape[0]
  ii, jj = np.indices(rm.shape)
  walls = [ii == 0, ii == nb - 1, jj == 0, jj == nb - 1]
  cm = max(np.sum(f & w) / nb for f in fields for w in walls)
  dist = np.minimum(np.minimum(ii, nb - 1 - ii),
                    np.minimum(jj, nb - 1 - jj)) / (nb / 2)
  allf = np.logical_or.reduce(fields)
  w = rm[allf]
  dm = float(np.sum(w * dist[allf]) / np.sum(w))
  return float((cm - dm) / (cm + dm))


def hd_resultant(acts_unit, hds):
  """Resultant vector length of activation-weighted head direction.

  r = |sum_t a_t exp(i hd_t)| / sum_t a_t. Bottleneck activations are
  linear so negatives are clipped to 0 to keep r in [0, 1]; NaN if the
  unit never activates.
  """
  w = np.clip(np.asarray(acts_unit, dtype=np.float64), 0, None)
  s = w.sum()
  if s <= 0:
    return np.nan
  z = np.sum(w * np.exp(1j * np.asarray(hds, dtype=np.float64)))
  return float(np.abs(z) / s)
