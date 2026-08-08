"""Env-layer tests: fake always; real DM-Lab with --real (inside container)."""
import argparse
import sys
import time

import numpy as np

try:
  from rl.env import ACTIONS, NUM_ACTIONS
  from rl.fake_env import FakeEnv
  from rl.vec_env import SubprocVecEnv
except ImportError:
  from env import ACTIONS, NUM_ACTIONS
  from fake_env import FakeEnv
  from vec_env import SubprocVecEnv

LEVEL = 'contributed/dmlab30/explore_goal_locations_small'
EXTENT = 2.75  # metres, 11 cells x 0.25 m


def check_obs(obs, n=None):
  """Assert batched (n) or single (n=None) obs shapes/dtypes/ranges."""
  lead = () if n is None else (n,)
  assert set(obs) == {'rgb', 'vel', 'pos', 'hd'}, sorted(obs)
  assert obs['rgb'].shape == lead + (84, 84, 3), obs['rgb'].shape
  assert obs['rgb'].dtype == np.uint8, obs['rgb'].dtype
  assert obs['vel'].shape == lead + (3,), obs['vel'].shape
  assert obs['vel'].dtype == np.float32, obs['vel'].dtype
  assert obs['pos'].shape == lead + (2,), obs['pos'].shape
  assert obs['pos'].dtype == np.float32, obs['pos'].dtype
  hd = np.asarray(obs['hd'])
  assert hd.shape == lead and hd.dtype == np.float32, (hd.shape, hd.dtype)
  assert np.all(np.abs(hd) <= np.pi + 1e-5), hd
  assert np.all(np.abs(obs['pos']) <= EXTENT / 2 + 0.1), obs['pos']
  assert np.all(np.isfinite(obs['vel'])), obs['vel']


def test_fake_single():
  """FakeEnv: contract, default 1800-step episodes, goal rewards."""
  env = FakeEnv('', seed=1)
  obs = env.reset()
  check_obs(obs)
  rng = np.random.default_rng(0)
  rewards, dones = 0.0, 0
  t0 = time.time()
  for t in range(1800):
    obs, r, d = env.step(int(rng.integers(NUM_ACTIONS)))
    rewards += r
    if d:
      dones += 1
      assert t == 1799, 'episode ended early at %d' % t
      obs = env.reset()
  dt = time.time() - t0
  check_obs(obs)
  assert dones == 1, dones
  print('[fake single] 1800 steps ok, %.0f steps/s, reward sum %.1f'
        % (1800 / dt, rewards))
  print('[fake single] rgb mean %.1f std %.1f | pos %s | hd %.2f'
        % (obs['rgb'].mean(), obs['rgb'].std(), obs['pos'], obs['hd']))
  env.close()


def test_fake_vec(n=4, steps=500, episode_len=120):
  """SubprocVecEnv(fake): shapes, deterministic auto-reset boundaries."""
  vec = SubprocVecEnv(n, '', base_seed=7, fake=True,
                      env_kwargs={'episode_len': episode_len})
  obs = vec.reset_all()
  check_obs(obs, n)
  rng = np.random.default_rng(1)
  reward_events, done_count = 0, 0
  pos_min = np.full(2, np.inf)
  pos_max = np.full(2, -np.inf)
  t0 = time.time()
  for t in range(steps):
    a = rng.integers(NUM_ACTIONS, size=n)
    obs, rewards, dones = vec.step(a)
    check_obs(obs, n)
    assert rewards.shape == (n,) and rewards.dtype == np.float32
    assert dones.shape == (n,) and dones.dtype == bool
    reward_events += int((rewards > 0).sum())
    done_count += int(dones.sum())
    # episode_len is fixed -> every env ends exactly each episode_len steps.
    want = np.full(n, (t + 1) % episode_len == 0)
    assert np.array_equal(dones, want), (t, dones)
    if dones.any():  # auto-reset: obs is a fresh episode's first obs
      assert np.allclose(obs['vel'][dones], 0.0), obs['vel']
    pos_min = np.minimum(pos_min, obs['pos'].min(0))
    pos_max = np.maximum(pos_max, obs['pos'].max(0))
  dt = time.time() - t0
  expect_dones = n * (steps // episode_len)
  assert done_count == expect_dones, (done_count, expect_dones)
  vec.close()
  print('[fake vec n=%d] %d steps ok, %.0f env-steps/s, %d dones, '
        '%d reward events' % (n, steps, steps * n / dt, done_count,
                              reward_events))
  print('[fake vec n=%d] pos range x[%.2f, %.2f] y[%.2f, %.2f] m'
        % (n, pos_min[0], pos_max[0], pos_min[1], pos_max[1]))


def test_real(steps=1000, vec_n=8, vec_steps=100):
  """Real DM-Lab: obs names, raw pos ranges, rewards, FPS (1 and vec_n)."""
  try:
    from rl.env import LabEnv
  except ImportError:
    from env import LabEnv

  # Short episodes so 1000 steps cover several mazes for calibration.
  env = LabEnv(LEVEL, seed=123, config={'episodeLengthSeconds': '20'})
  print('[real] observation_spec names:')
  print('  ' + '\n  '.join(env.spec_names))
  print('[real] using rgb obs: %s' % env._rgb_name)

  obs = env.reset()
  check_obs(obs)

  # Action sanity: forward -> +vel[0]; turn_left -> +vel[2] at ~2.21 rad/s
  # (20 px/frame; yaw CCW-positive). Turn rate is derived from
  # DEBUG.POS.ROT deltas since VEL.ROT reads 0 while RGB is rendered.
  for name, idx, comp, lo, hi in (('forward', 2, 0, 0.05, 1.0),
                                  ('turn_left', 0, 2, 2.0, 2.4),
                                  ('turn_right', 1, 2, -2.4, -2.0)):
    env.reset()
    vals = []
    for _ in range(20):
      o, _, d = env.step(idx)
      vals.append(float(o['vel'][comp]))
      if d:
        break
    m = np.mean(vals[5:])
    print('[real] action %-10s mean vel[%d] = %+.3f' % (name, comp, m))
    assert lo <= m <= hi, (name, m, lo, hi)

  rng = np.random.default_rng(2)
  raw_min = np.full(2, np.inf)
  raw_max = np.full(2, -np.inf)
  pos_min = np.full(2, np.inf)
  pos_max = np.full(2, -np.inf)
  reward_events, episodes = [], 0
  obs = env.reset()
  t0 = time.time()
  t_step = 0.0
  for t in range(steps):
    ts = time.time()
    obs, r, d = env.step(int(rng.integers(NUM_ACTIONS)))
    t_step += time.time() - ts
    check_obs(obs)
    if r != 0:
      reward_events.append((t, r))
    raw_min = np.minimum(raw_min, env.last_raw_pos)
    raw_max = np.maximum(raw_max, env.last_raw_pos)
    pos_min = np.minimum(pos_min, obs['pos'])
    pos_max = np.maximum(pos_max, obs['pos'])
    if d:
      episodes += 1
      obs = env.reset()
  dt = time.time() - t0
  print('[real 1 env] %d steps (repeat=4): %.1f agent-steps/s '
        '= %.0f frames/s incl. %d maze-regen resets; '
        '%.1f steps/s = %.0f frames/s stepping only'
        % (steps, steps / dt, 4 * steps / dt, episodes,
           steps / t_step, 4 * steps / t_step))
  print('[real] raw DEBUG.POS.TRANS range: x[%.1f, %.1f] y[%.1f, %.1f] '
        '(arena [0, 1100], centre offset 550)'
        % (raw_min[0], raw_max[0], raw_min[1], raw_max[1]))
  print('[real] centred pos range (m): x[%.3f, %.3f] y[%.3f, %.3f] '
        '(|pos| should be <= %.3f)'
        % (pos_min[0], pos_max[0], pos_min[1], pos_max[1], EXTENT / 2))
  print('[real] reward events: %d %s'
        % (len(reward_events), reward_events[:10]))
  assert np.all(np.abs([pos_min, pos_max]) <= EXTENT / 2 + 0.1), \
      'positions not centred; adjust origin offset'
  env.close()

  vec = SubprocVecEnv(vec_n, LEVEL, base_seed=1000, fake=False)
  obs = vec.reset_all()
  check_obs(obs, vec_n)
  t0 = time.time()
  vec_rewards = 0
  for _ in range(vec_steps):
    a = rng.integers(NUM_ACTIONS, size=vec_n)
    obs, rewards, dones = vec.step(a)
    check_obs(obs, vec_n)
    vec_rewards += int((rewards > 0).sum())
  dt = time.time() - t0
  print('[real vec n=%d] %d steps: %.1f env-steps/s = %.0f frames/s, '
        '%d reward events'
        % (vec_n, vec_steps, vec_steps * vec_n / dt,
           4 * vec_steps * vec_n / dt, vec_rewards))
  vec.close()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--real', action='store_true',
                  help='also test real DM-Lab (run inside dmlab container)')
  ap.add_argument('--steps', type=int, default=1000)
  args = ap.parse_args()
  assert len(ACTIONS) == 6
  assert all(a.dtype == np.intc and a.shape == (7,) for _, a in ACTIONS)
  test_fake_single()
  test_fake_vec()
  if args.real:
    test_real(steps=args.steps)
  print('ALL TESTS PASSED')
  return 0


if __name__ == '__main__':
  sys.exit(main())
