"""Subprocess-vectorised env: n workers, auto-reset on done."""
import multiprocessing as mp
import traceback

import numpy as np


def _worker(conn, level, seed, fake, env_kwargs):
  """Own one env; serve reset/step/close over the pipe."""
  try:
    if fake:
      try:
        from .fake_env import FakeEnv as Env
      except ImportError:
        from fake_env import FakeEnv as Env
    else:
      try:
        from .env import LabEnv as Env
      except ImportError:
        from env import LabEnv as Env
    env = Env(level, seed, **env_kwargs)
    while True:
      cmd, arg = conn.recv()
      if cmd == 'reset':
        conn.send(env.reset())
      elif cmd == 'step':
        idx, repeat = arg
        obs, reward, done = env.step(idx, repeat)
        if done:  # auto-reset: return first obs of the next episode
          obs = env.reset()
        conn.send((obs, reward, done))
      elif cmd == 'close':
        env.close()
        conn.send(None)
        break
  except (KeyboardInterrupt, EOFError):
    pass
  except Exception:
    err = ('__error__', traceback.format_exc())
    try:  # keep answering until the parent closes, so it sees the error
      conn.send(err)
      while True:
        conn.recv()
        conn.send(err)
    except (EOFError, BrokenPipeError, OSError):
      pass
  finally:
    conn.close()


def _stack(obs_list):
  """Stack per-env obs dicts into batched arrays."""
  return {
      'rgb': np.stack([o['rgb'] for o in obs_list]),
      'vel': np.stack([o['vel'] for o in obs_list]),
      'pos': np.stack([o['pos'] for o in obs_list]),
      'hd': np.asarray([o['hd'] for o in obs_list], dtype=np.float32),
  }


class SubprocVecEnv:
  """n LabEnvs (or FakeEnvs) in spawned subprocesses, batched API."""

  def __init__(self, n, level, base_seed, fake=False, env_kwargs=None):
    ctx = mp.get_context('spawn')
    self.n = int(n)
    self._conns = []
    self._procs = []
    kwargs = dict(env_kwargs or {})
    for i in range(self.n):
      parent, child = ctx.Pipe()
      p = ctx.Process(target=_worker,
                      args=(child, level, int(base_seed) + i, fake, kwargs),
                      daemon=True)
      p.start()
      child.close()
      self._conns.append(parent)
      self._procs.append(p)
    self._closed = False

  def _recv(self, conn):
    try:
      msg = conn.recv()
    except (EOFError, ConnectionResetError) as e:
      self.close()
      raise RuntimeError('vec env worker died without reporting an error '
                         '(killed, or crashed during interpreter '
                         'bootstrap)') from e
    if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == '__error__':
      self.close()
      raise RuntimeError('vec env worker failed:\n' + msg[1])
    return msg

  def reset_all(self):
    """Reset every env -> batched obs dict."""
    for c in self._conns:
      c.send(('reset', None))
    return _stack([self._recv(c) for c in self._conns])

  def step(self, actions, repeat=4):
    """Step all envs -> (batched obs, rewards [n] f32, dones [n] bool).

    Envs that finish are auto-reset; their obs is the first of the new
    episode while reward/done refer to the finished one.
    """
    actions = np.asarray(actions).reshape(-1)
    if actions.shape[0] != self.n:
      raise ValueError('expected %d actions, got %d' % (self.n, len(actions)))
    for c, a in zip(self._conns, actions):
      c.send(('step', (int(a), int(repeat))))
    replies = [self._recv(c) for c in self._conns]
    obs = _stack([r[0] for r in replies])
    rewards = np.asarray([r[1] for r in replies], dtype=np.float32)
    dones = np.asarray([r[2] for r in replies], dtype=bool)
    return obs, rewards, dones

  def close(self):
    """Shut down workers."""
    if self._closed:
      return
    self._closed = True
    for c in self._conns:
      try:
        c.send(('close', None))
        c.recv()
      except (BrokenPipeError, EOFError, OSError):
        pass
      c.close()
    for p in self._procs:
      p.join(timeout=10)
      if p.is_alive():
        p.terminate()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
