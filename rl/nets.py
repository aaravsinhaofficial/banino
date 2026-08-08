"""Networks for the scaled-down RL reproduction (Banino et al. 2018, Figs 2-3).

Three separately-trained modules, per the paper (no gradient sharing):
  VisionCNN    frames -> place/HD code predictions (supervised from replay)
  GridModule   velocities + 95%-masked visual codes -> grid code g_t
               (supervised path integration, hyperparameters from the
               supervised experiments; LSTM state zero-initialised per
               episode as in the paper's agent)
  PolicyNet    own conv net + LSTM over [visual feats, grid code, goal grid
               code, prev action, prev reward] -> policy logits + value
"""

import torch
from torch import nn


class VisionCNN(nn.Module):
  """Predicts place/HD ensemble codes from 84x84 RGB frames."""

  def __init__(self, n_pc=256, n_hdc=12):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
    )
    self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 9 * 9, 256),
                            nn.ReLU())
    self.pc_head = nn.Linear(256, n_pc)
    self.hd_head = nn.Linear(256, n_hdc)

  def forward(self, rgb):
    """rgb float [B,3,84,84] in [0,1] -> (pc_logits, hd_logits)."""
    f = self.fc(self.conv(rgb))
    return self.pc_head(f), self.hd_head(f)


class GridModule(nn.Module):
  """Streaming grid network for the agent (LSTMCell + dropout bottleneck)."""

  def __init__(self, n_pc=256, n_hdc=12, nh_lstm=128, nh_bottleneck=512,
               dropout=0.5):
    super().__init__()
    self.n_in = 3 + n_pc + n_hdc
    self.cell = nn.LSTMCell(self.n_in, nh_lstm)
    self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.pc_logits = nn.Linear(nh_bottleneck, n_pc)
    self.hd_logits = nn.Linear(nh_bottleneck, n_hdc)
    self.nh_lstm = nh_lstm

  def zero_state(self, batch, device):
    return (torch.zeros(batch, self.nh_lstm, device=device),
            torch.zeros(batch, self.nh_lstm, device=device))

  def step(self, vel, vis_pc, vis_hd, mask, state):
    """One timestep. mask [B,1] in {0,1}: 1 = visual code passed through.

    vel [B,3]; vis_pc [B,n_pc]; vis_hd [B,n_hdc]. Returns (g, next_state).
    Dropout stays active whenever self.training (the paper found dropout on
    the bottleneck essential for grid-like firing in the agent too).
    """
    x = torch.cat([vel, vis_pc * mask, vis_hd * mask], dim=-1)
    h, c = self.cell(x, state)
    g = self.dropout(self.bottleneck(h))
    return g, (h, c)

  def heads(self, g):
    return self.pc_logits(g), self.hd_logits(g)

  def regularized_params(self):
    return [self.bottleneck.weight, self.pc_logits.weight,
            self.hd_logits.weight]


class PolicyNet(nn.Module):
  """A2C actor-critic with its own conv net (no gradients into GridModule)."""

  def __init__(self, n_actions=6, n_grid=512, nh=256):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
    )
    self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 9 * 9, 256),
                            nn.ReLU())
    self.lstm = nn.LSTMCell(256 + 2 * n_grid + n_actions + 1, nh)
    self.pi = nn.Linear(nh, n_actions)
    self.v = nn.Linear(nh, 1)
    self.nh = nh
    self.n_actions = n_actions

  def zero_state(self, batch, device):
    return (torch.zeros(batch, self.nh, device=device),
            torch.zeros(batch, self.nh, device=device))

  def step(self, rgb, grid_code, goal_code, prev_action, prev_reward, state):
    """rgb [B,3,84,84]; grid/goal codes [B,n_grid]; prev_action [B] long;
    prev_reward [B]. Returns (pi_logits, value, next_state)."""
    f = self.fc(self.conv(rgb))
    pa = torch.zeros(rgb.shape[0], self.n_actions, device=rgb.device)
    pa.scatter_(1, prev_action.unsqueeze(1), 1.0)
    x = torch.cat([f, grid_code, goal_code, pa,
                   prev_reward.unsqueeze(1)], dim=-1)
    h, c = self.lstm(x, state)
    return self.pi(h), self.v(h).squeeze(-1), (h, c)
