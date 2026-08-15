"""DeepMind Lab wrapper with metric observations (Banino et al. 2018 repro).

Coordinates: DM-Lab uses 100 world units per maze cell (kGridWidth in
deepmind/engine/lua_maze_generation.cc); cell (row, col) of an
arena_cells x arena_cells maze has its centre at (100*k + 50) per axis,
so the arena spans [0, 100*arena_cells] world units. Positions are
re-centred by subtracting 50*arena_cells (550 for 11 cells) and scaled
by cell_m/100 metres per unit (0.0025 for cell_m=0.25 -> 2.75 m arena).
Calibration (measured, explore_goal_locations_small, 1000 random steps
over several mazes): raw DEBUG.POS.TRANS spans [116, 984] per axis,
symmetric about 550 (border wall 100 units + player hull ~16), so
centred positions stay within +/-1.09 m of the +/-1.375 m half-extent.

Angular velocity: the engine zeroes the VEL.ROT observation whenever an
RGB observation is rendered (anglesVel delta logic in
ContextGame::SetPlayerState), so obs['vel'][2] is instead derived from
consecutive DEBUG.POS.ROT yaw deltas over the step's frames; this
matches the commanded look rate exactly (20 px/frame -> 2.21 rad/s).
"""
import os

import numpy as np

# DM-Lab discrete action order, as returned by Lab.action_spec():
# [LOOK_LEFT_RIGHT_PIXELS_PER_FRAME, LOOK_DOWN_UP_PIXELS_PER_FRAME,
#  STRAFE_LEFT_RIGHT, MOVE_BACK_FORWARD, FIRE, JUMP, CROUCH]
_ACTION_SPEC_NAMES = [
    'LOOK_LEFT_RIGHT_PIXELS_PER_FRAME', 'LOOK_DOWN_UP_PIXELS_PER_FRAME',
    'STRAFE_LEFT_RIGHT', 'MOVE_BACK_FORWARD', 'FIRE', 'JUMP', 'CROUCH']

_ROT_PIX = 20  # look pixels/frame; 57 px/frame ~= 360 deg/s, so 20 ~= 126 deg/s


def _act(look_lr=0, look_ud=0, strafe=0, move=0, fire=0, jump=0, crouch=0):
  """Build one np.intc[7] DM-Lab action vector."""
  return np.array([look_lr, look_ud, strafe, move, fire, jump, crouch],
                  dtype=np.intc)


# DeepMind Lab's standard navigation action set (Mnih et al. 2016 A3C and
# the DMLab-30 baselines the paper's agents were built on), minus FIRE,
# which does nothing in the goal-navigation levels. The combined
# forward+look actions matter: without them an agent must stop to turn, so
# a random policy barely displaces and almost never encounters a goal —
# the only learning signal these sparse tasks provide.
ACTIONS = [
    ('turn_left', _act(look_lr=-_ROT_PIX)),
    ('turn_right', _act(look_lr=_ROT_PIX)),
    ('forward', _act(move=1)),
    ('backward', _act(move=-1)),
    ('strafe_left', _act(strafe=-1)),
    ('strafe_right', _act(strafe=1)),
    ('forward_turn_left', _act(look_lr=-_ROT_PIX, move=1)),
    ('forward_turn_right', _act(look_lr=_ROT_PIX, move=1)),
]

# The paper's actor is "a linear layer with six units" (Methods, agent
# architecture), i.e. six discrete actions; it does not say which six.
# BANINO_ACTION_SET=6 truncates to the first six — the set this repo used
# before 2026-08-15 — so results from the two action sets can be compared
# against matched baselines.
if os.environ.get('BANINO_ACTION_SET') == '6':
  ACTIONS = ACTIONS[:6]
NUM_ACTIONS = len(ACTIONS)

# RGB observation fallbacks: (name, planar layout?).
_RGB_PREFS = [('RGB_INTERLEAVED', False), ('RGBD_INTERLEAVED', False),
              ('RGB', True), ('RGBD', True)]
_SCALAR_OBS = ['VEL.TRANS', 'VEL.ROT', 'DEBUG.POS.TRANS', 'DEBUG.POS.ROT']


def _wrap_pi(a):
  """Wrap angle(s) to [-pi, pi]."""
  return (a + np.pi) % (2.0 * np.pi) - np.pi


class LabEnv:
  """One DM-Lab instance yielding {'rgb','vel','pos','hd'} observations."""

  def __init__(self, level, seed, size=84, cell_m=0.25, arena_cells=11,
               config=None):
    import deepmind_lab  # lazy so this module imports without DM-Lab
    self._seed = int(seed)
    self._episode = -1
    self._size = int(size)
    self._m_per_unit = cell_m / 100.0
    self._origin_units = 50.0 * arena_cells  # arena centre, world units
    self.arena_extent_m = cell_m * arena_cells
    self._last_obs = None
    self._prev_yaw = None
    self.last_raw_pos = None  # raw DEBUG.POS.TRANS[:2], for calibration

    cfg = {'width': str(self._size), 'height': str(self._size), 'fps': '60'}
    for k, v in (config or {}).items():
      cfg[str(k)] = str(v)

    # Probe available observation names (spec is queryable pre-reset).
    probe = deepmind_lab.Lab(level, [], config=cfg, renderer='software')
    avail = {s['name'] for s in probe.observation_spec()}
    probe.close()
    self.spec_names = sorted(avail)

    self._rgb_name = None
    for name, planar in _RGB_PREFS:
      if name in avail:
        self._rgb_name, self._rgb_planar = name, planar
        break
    missing = [n for n in _SCALAR_OBS if n not in avail]
    if self._rgb_name is None or missing:
      raise RuntimeError(
          'missing observations rgb=%s missing=%s; available: %s'
          % (self._rgb_name, missing, self.spec_names))

    self._lab = deepmind_lab.Lab(level, [self._rgb_name] + _SCALAR_OBS,
                                 config=cfg, renderer='software')
    got = [a['name'] for a in self._lab.action_spec()]
    if got != _ACTION_SPEC_NAMES:
      raise RuntimeError('unexpected action_spec order: %s' % got)

  def _observe(self, dt=None):
    """Read raw observations, convert to metres/radians.

    dt: seconds since the previous observation; None (at reset) yields
    zero angular velocity. Yaw rate comes from DEBUG.POS.ROT deltas
    because VEL.ROT reads zero while RGB observations are rendered.
    """
    o = self._lab.observations()
    rgb = o[self._rgb_name]
    if self._rgb_planar:
      rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    raw = np.asarray(o['DEBUG.POS.TRANS'], dtype=np.float64)  # x, y, z units
    yaw = np.deg2rad(o['DEBUG.POS.ROT'][1])
    rot = 0.0
    if dt is not None and self._prev_yaw is not None:
      teleported = (  # goal respawn: don't turn the pose jump into velocity
          np.linalg.norm(raw[:2] - self.last_raw_pos) > 50.0)
      if not teleported:
        rot = _wrap_pi(yaw - self._prev_yaw) / dt
    self._prev_yaw = yaw
    vt = o['VEL.TRANS']  # units/s, egocentric (forward, leftward, up)
    vel = np.array([vt[0] * self._m_per_unit,   # forward m/s
                    vt[1] * self._m_per_unit,   # sideways (leftward+) m/s
                    rot],                       # yaw rad/s
                   dtype=np.float32)
    self.last_raw_pos = raw[:2].copy()
    pos = ((raw[:2] - self._origin_units) * self._m_per_unit).astype(np.float32)
    hd = np.float32(_wrap_pi(yaw))
    self._last_obs = {'rgb': rgb, 'vel': vel, 'pos': pos, 'hd': hd}
    return self._last_obs

  def reset(self):
    """Start a new episode; returns first obs."""
    self._episode += 1
    self._lab.reset(seed=self._seed + self._episode)
    self._prev_yaw = None
    return self._observe()

  def step(self, action_idx, repeat=4):
    """Apply ACTIONS[action_idx] for `repeat` frames -> (obs, reward, done).

    No auto-reset: after done the caller must reset(). The obs returned
    with done=True is the last valid frame of the finished episode.
    """
    if not self._lab.is_running():
      raise RuntimeError('step() before reset() or after episode end')
    reward = self._lab.step(ACTIONS[action_idx][1], num_steps=int(repeat))
    done = not self._lab.is_running()
    obs = self._last_obs if done else self._observe(dt=repeat / 60.0)
    return obs, float(reward), done

  def is_running(self):
    """True while the current episode is in progress."""
    return self._lab.is_running()

  def close(self):
    """Release the underlying DM-Lab instance."""
    self._lab.close()
