"""Place-cell / head-direction-cell target ensembles (PyTorch port).

Faithful reimplementation of ensembles.py + utils.py from the TF1 code:
targets are the softmax posterior over a mixture of isotropic Gaussians
(place) / von Mises components (head direction), with centres drawn from
numpy RandomState(seed) exactly as the original so targets are directly
comparable across the two stacks.
"""

import numpy as np
import torch

DEFAULT_SEED = 8341   # task_neurons_seed flag default (= paper configuration)


class PlaceCellEnsemble:
  """N Gaussian place cells; targets are the softmax posterior over cells."""

  def __init__(self, n_cells=256, stdev=0.01, pos_min=-1.1, pos_max=1.1,
               seed=DEFAULT_SEED, device='cpu'):
    rs = np.random.RandomState(seed)
    self.n_cells = n_cells
    self.means = torch.tensor(
        rs.uniform(pos_min, pos_max, size=(n_cells, 2)),
        dtype=torch.float32, device=device)
    self.variance = stdev ** 2

  def unnor_logpdf(self, pos):
    # pos: [..., 2] -> [..., n_cells]
    diff = pos.unsqueeze(-2) - self.means
    return -0.5 * (diff ** 2).sum(-1) / self.variance

  def get_targets(self, pos):
    return torch.softmax(self.unnor_logpdf(pos), dim=-1)


class HeadDirectionCellEnsemble:
  """M von Mises HD cells; targets are the softmax posterior over cells."""

  def __init__(self, n_cells=12, concentration=20.0, seed=DEFAULT_SEED,
               device='cpu'):
    rs = np.random.RandomState(seed)
    self.n_cells = n_cells
    self.means = torch.tensor(rs.uniform(-np.pi, np.pi, n_cells),
                              dtype=torch.float32, device=device)
    self.kappa = concentration

  def unnor_logpdf(self, hd):
    # hd: [..., 1] -> [..., n_cells]
    return self.kappa * torch.cos(hd - self.means)

  def get_targets(self, hd):
    return torch.softmax(self.unnor_logpdf(hd), dim=-1)
