"""Dense-reward probe environment: a unit test for the RL update path.

Not part of the reproduction. One action (default 2, 'forward') pays +1
every step and the rest pay 0, with observations and episode structure
shaped like LabEnv's. Any working policy-gradient learner must drive the
action distribution onto that action within a few thousand steps, so this
separates "the learner is broken" from "the task is hard".
"""

import numpy as np

_GOOD_ACTION = 2


class BanditEnv:
  """LabEnv-compatible interface with a trivially learnable reward."""

  def __init__(self, level='', seed=0, size=84, cell_m=0.25, arena_cells=11,
               episode_len=200):
    del level
    self._rng = np.random.default_rng(int(seed))
    self._size = int(size)
    self._episode_len = int(episode_len)
    self._t = 0
    self._running = False

  def _observe(self):
    img = self._rng.normal(128.0, 8.0, (self._size, self._size, 3))
    return {
        'rgb': np.clip(img, 0, 255).astype(np.uint8),
        'vel': np.zeros(3, dtype=np.float32),
        'pos': np.zeros(2, dtype=np.float32),
        'hd': np.float32(0.0),
    }

  def reset(self):
    self._t = 0
    self._running = True
    return self._observe()

  def step(self, action_idx, repeat=4):
    if not self._running:
      raise RuntimeError('step() after episode end; call reset()')
    del repeat
    reward = 1.0 if int(action_idx) == _GOOD_ACTION else 0.0
    self._t += 1
    done = self._t >= self._episode_len
    if done:
      self._running = False
    return self._observe(), reward, done

  def is_running(self):
    return self._running

  def close(self):
    self._running = False
