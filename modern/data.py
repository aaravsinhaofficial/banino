"""On-the-fly trajectory batches for the PyTorch port.

Reuses the pure-numpy Raudies–Hasselmo simulator from
generate_trajectories.py (the same one that regenerates the TFRecord
dataset), sampling fresh trajectories per batch — statistically equivalent
to sampling from the original 1M-trajectory dataset.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_trajectories as gt  # noqa: E402


class TrajectorySampler:

  def __init__(self, seed=0, device='cpu'):
    self.rng = np.random.RandomState(seed)
    self.device = device

  def batch(self, batch_size):
    d = gt._simulate(batch_size, self.rng)
    t = {k: torch.from_numpy(v).to(self.device) for k, v in d.items()}
    return t['init_pos'], t['init_hd'], t['ego_vel'], t['target_pos'], t['target_hd']
