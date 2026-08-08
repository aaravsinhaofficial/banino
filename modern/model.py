"""Grid-cell network (PyTorch port of model.py).

Architecture per Banino et al. 2018 Methods: single-layer LSTM (128 units,
no peepholes — matches nn.LSTM), whose initial state is a linear function of
the ground-truth place/HD codes at t=0; a linear bottleneck g (512 units,
no bias) with dropout 0.5 applied at every time step; linear softmax heads
predicting place-cell and HD-cell posteriors.
"""

import torch
from torch import nn


class GridNetwork(nn.Module):

  def __init__(self, n_pc=256, n_hdc=12, nh_lstm=128, nh_bottleneck=512,
               dropout=0.5, bottleneck_has_bias=False):
    super().__init__()
    self.state_init = nn.Linear(n_pc + n_hdc, nh_lstm)
    self.cell_init = nn.Linear(n_pc + n_hdc, nh_lstm)
    self.lstm = nn.LSTM(3, nh_lstm, batch_first=True)
    self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck,
                                bias=bottleneck_has_bias)
    self.dropout = nn.Dropout(dropout)
    self.pc_logits = nn.Linear(nh_bottleneck, n_pc)
    self.hd_logits = nn.Linear(nh_bottleneck, n_hdc)

  def forward(self, init_pc, init_hd, ego_vel):
    """init_pc [B,Npc], init_hd [B,Nhd], ego_vel [B,T,3]."""
    init = torch.cat([init_pc, init_hd], dim=-1)
    h0 = self.state_init(init).unsqueeze(0)
    c0 = self.cell_init(init).unsqueeze(0)
    lstm_out, _ = self.lstm(ego_vel, (h0, c0))
    bottleneck = self.dropout(self.bottleneck(lstm_out))
    return self.pc_logits(bottleneck), self.hd_logits(bottleneck), bottleneck

  def regularized_params(self):
    """Weights carrying L2 weight decay in the original (bottleneck+heads)."""
    return [self.bottleneck.weight, self.pc_logits.weight,
            self.hd_logits.weight]
