import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl
import sys

from src.l0_regularizer import *
class Sparser(pl.LightningModule):
    def __init__(self,
                 dim_Q: int,
                 dim_K: int,
                 dim_out: int,
                 num_heads: int,
                 reg_type: str,
                 ln: bool,
                 loc_mean = 1,
                 loc_sdev = 0.01,
                 beta = 2 / 3,
                 gamma = -0.1,
                 zeta = 1.1,
                 fix_temp = True):
        super(Sparser, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_out = dim_out

        self.num_heads = num_heads
        self.dim_head = dim_K // num_heads

        self.W_q = nn.ModuleList([
            nn.Linear(dim_Q, self.dim_head) for _ in range(num_heads)
        ])
        self.W_k = nn.ModuleList([
            nn.Linear(dim_K, self.dim_head) for _ in range(num_heads)
        ])

        l0_params = {
            'loc_mean': loc_mean,
            'loc_sdev': loc_sdev,
            'beta': beta,
            'gamma': gamma,
            'zeta': zeta,
            'fix_temp': fix_temp
        }

        self.reg_type = reg_type
        if reg_type == 'linear':
            # self.l0_gate = nn.ModuleList([
            #     L0Linear(dim_Q, dim_Q, **l0_params) for _ in range(num_heads)
            # ])
            self.l0_gate = L0Linear(dim_Q, dim_Q, loc_mean=1)
        elif reg_type == 'conv2d':
            self.l0_gate = nn.ModuleList([
                L0Conv2d(dim_Q, dim_Q, kernel_size=1, stride=1) for _ in range(num_heads)
            ])
            # self.l0_gate = L0Conv2d(dim_Q, dim_Q, kernel_size=1, stride=1)

        if ln:
            self.ln_q = nn.LayerNorm(dim_Q)
            self.ln_k = nn.LayerNorm(dim_K)

        for head in range(num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)

    def forward(self, Q, K):
        Q_norm = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K_norm = K if getattr(self, 'ln_k', None) is None else self.ln_k(K)

        head_outputs = []
        total_penalty = 0
        for head in range(self.num_heads):
            Q_ = self.W_q[head](Q_norm)
            K_ = self.W_k[head](K_norm)

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_head), 2)

            # gate, penalty = self.l0_gate[head](A)
            gate, penalty = self.l0_gate(A)
            total_penalty += penalty
            head_outputs.append(gate)

        O = torch.stack(head_outputs, dim=1)
        O = O.mean(dim=1)

        return O, total_penalty
            



