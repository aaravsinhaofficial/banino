"""LabEnv-contract fake: kinematic agent in a square arena, numpy only."""
import numpy as np

try:
  from .env import ACTIONS, NUM_ACTIONS, _wrap_pi
except ImportError:
  from env import ACTIONS, NUM_ACTIONS, _wrap_pi

_FPS = 60.0
_TAU = 0.25        # velocity smoothing time constant (s)
_V_MAX = 0.35      # target move/strafe speed (m/s), slow-rat-like
_ROT_MAX = 2.2     # target turn speed (rad/s), ~20 px/frame in DM-Lab
_NOISE_V = 0.03    # motor noise on linear targets (m/s)
_NOISE_R = 0.15    # motor noise on turn target (rad/s)
_WALL_MARGIN = 0.08
_GOAL_RADIUS = 0.3
_GOAL_REWARD = 10.0

# action_idx -> (forward, sideways-left, rot) target multipliers.
_TARGETS = {
    0: (0.0, 0.0, 1.0),    # turn_left  (yaw increases CCW, as in DM-Lab)
    1: (0.0, 0.0, -1.0),   # turn_right
    2: (1.0, 0.0, 0.0),    # forward
    3: (-1.0, 0.0, 0.0),   # backward
    4: (0.0, 1.0, 0.0),    # strafe_left
    5: (0.0, -1.0, 0.0),   # strafe_right
    6: (1.0, 0.0, 1.0),    # forward_turn_left
    7: (1.0, 0.0, -1.0),   # forward_turn_right
}
# Follow rl.env when BANINO_ACTION_SET truncates the action set.
_TARGETS = {k: v for k, v in _TARGETS.items() if k < NUM_ACTIONS}
assert len(_TARGETS) == NUM_ACTIONS == len(ACTIONS)


class FakeEnv:
  """Drop-in LabEnv replacement; no deepmind_lab required."""

  def __init__(self, level='', seed=0, size=84, cell_m=0.25, arena_cells=11,
               episode_len=1800, blind='none'):
    del level  # unused; signature parity with LabEnv
    if blind not in ('none', 'pos', 'all'):
      raise ValueError(f'blind must be none/pos/all, got {blind!r}')
    self._blind = blind
    self._rng = np.random.default_rng(int(seed))
    self._size = int(size)
    self._half = 0.5 * cell_m * arena_cells  # 1.375 m for defaults
    self.arena_extent_m = cell_m * arena_cells
    self._episode_len = int(episode_len)
    xs = np.linspace(0.0, 1.0, self._size, dtype=np.float32)
    self._gu, self._gv = np.meshgrid(xs, xs)
    self._t = 0
    self._running = False

  def _random_pos(self):
    lim = self._half - _WALL_MARGIN
    return self._rng.uniform(-lim, lim, size=2)

  def _respawn_agent(self):
    """Random pose away from the goal; zero velocities."""
    while True:
      p = self._random_pos()
      if np.linalg.norm(p - self._goal) > _GOAL_RADIUS + 0.1:
        break
    self._pos = p
    self._hd = self._rng.uniform(-np.pi, np.pi)
    self._vf = self._vs = self._vr = 0.0

  def _observe(self):
    """Synthetic obs: position/heading-keyed gradient image + noise.

    blind='pos' fixes the position keys at mid-range so vision carries
    heading but NOT location (localization must come from elsewhere, e.g.
    a path-integration code); blind='all' fixes the heading key too.
    """
    u = 0.5 * (self._pos[0] / self._half + 1.0)  # in [0, 1]
    v = 0.5 * (self._pos[1] / self._half + 1.0)
    h = 0.5 * (self._hd / np.pi + 1.0)
    if self._blind in ('pos', 'all'):
      u = v = 0.5
    if self._blind == 'all':
      h = 0.5
    img = np.empty((self._size, self._size, 3), dtype=np.float32)
    img[..., 0] = 255.0 * (0.5 * self._gu + 0.5 * u)
    img[..., 1] = 255.0 * (0.5 * self._gv + 0.5 * v)
    img[..., 2] = 255.0 * (0.5 * h + 0.5 * self._gu * self._gv)
    img += self._rng.normal(0.0, 8.0, img.shape)
    return {
        'rgb': np.clip(img, 0, 255).astype(np.uint8),
        'vel': np.array([self._vf, self._vs, self._vr], dtype=np.float32),
        'pos': self._pos.astype(np.float32),
        'hd': np.float32(_wrap_pi(self._hd)),
    }

  def reset(self):
    """New episode: new goal, random agent pose; returns first obs."""
    self._goal = self._random_pos()
    self._respawn_agent()
    self._t = 0
    self._running = True
    return self._observe()

  def step(self, action_idx, repeat=4):
    """Apply action for repeat/60 s -> (obs, reward, done). No auto-reset."""
    if not self._running:
      raise RuntimeError('step() after episode end; call reset()')
    dt = repeat / _FPS
    tf, ts, tr = _TARGETS[int(action_idx)]
    tf = tf * _V_MAX + self._rng.normal(0.0, _NOISE_V)
    ts = ts * _V_MAX + self._rng.normal(0.0, _NOISE_V)
    tr = tr * _ROT_MAX + self._rng.normal(0.0, _NOISE_R)
    k = 1.0 - np.exp(-dt / _TAU)
    self._vf += (tf - self._vf) * k
    self._vs += (ts - self._vs) * k
    self._vr += (tr - self._vr) * k
    self._hd = _wrap_pi(self._hd + self._vr * dt)
    c, s = np.cos(self._hd), np.sin(self._hd)
    step_xy = np.array([self._vf * c - self._vs * s,
                        self._vf * s + self._vs * c]) * dt
    lim = self._half - _WALL_MARGIN
    new = np.clip(self._pos + step_xy, -lim, lim)
    hit = new != self._pos + step_xy
    if hit[0] or hit[1]:  # wall: kill velocity like a collision
      self._vf *= 0.2
      self._vs *= 0.2
    self._pos = new
    reward = 0.0
    if np.linalg.norm(self._pos - self._goal) < _GOAL_RADIUS:
      reward = _GOAL_REWARD
      self._respawn_agent()  # DM-Lab-style: agent respawns, goal stays
    self._t += 1
    done = self._t >= self._episode_len
    if done:
      self._running = False
    return self._observe(), float(reward), done

  def is_running(self):
    """True while the current episode is in progress."""
    return self._running

  def close(self):
    """No-op (contract parity)."""
