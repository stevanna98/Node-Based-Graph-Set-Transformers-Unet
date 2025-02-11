import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl
import sys
import random

class Sparser(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_out, num_heads, ln):
        super(Sparser, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_out = dim_out  

        self.num_heads = num_heads  
        self.dim_head = dim_K // num_heads

        self.W_q = nn.ModuleList([nn.Linear(dim_Q, self.dim_head) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(dim_K, self.dim_head) for _ in range(num_heads)])

        self.sparser = nn.ModuleList([nn.Conv2d(dim_Q, dim_Q, kernel_size=1, stride=1, padding=0) for _ in range(num_heads)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm2d(dim_Q) for _ in range(num_heads)])

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
        attn_weights = []
        for head in range(self.num_heads):
            Q_ = self.W_q[head](Q_norm)
            K_ = self.W_k[head](K_norm)

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_head), 2)
            A = A.permute(0, 2, 1).unsqueeze(-1)
            A_ = self.sparser[head](A)
            A_ = self.batch_norm[head](A_)
            A_ = A_.squeeze(-1).permute(0, 2, 1)

            head_output = A_.bmm(Q)
            head_outputs.append(head_output)
            attn_weights.append(A_)

        O = torch.stack(head_outputs, dim=1)
        O = O.mean(dim=1)

        return O, attn_weights

        