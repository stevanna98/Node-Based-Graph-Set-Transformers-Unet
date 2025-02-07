import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import lightning.pytorch as pl

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

class MMAB(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln, dropout):
        super(MMAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V

        self.num_heads = num_heads
        self.dim_head = dim_V // num_heads

        self.W_q = nn.ModuleList([nn.Linear(dim_Q, self.dim_head) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(dim_K, self.dim_head) for _ in range(num_heads)])
        self.W_v = nn.ModuleList([nn.Linear(dim_K, self.dim_head) for _ in range(num_heads)])

        self.dropout = nn.Dropout(dropout)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
            self.ln_q = nn.LayerNorm(dim_Q)
            self.ln_k = nn.LayerNorm(dim_K)

        self.fc_o = nn.Linear(dim_V, dim_V)

        for head in range(num_heads):
            nn.init.xavier_uniform_(self.W_q[head].weight)
            nn.init.xavier_uniform_(self.W_k[head].weight)
            nn.init.xavier_uniform_(self.W_v[head].weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

    def mask_attention(self, A, M):
        if M is not None:
            M = M.to(device)
            A = A.masked_fill(M == 0, -1e20)
        return A
    
    def forward(self, Q, K, M):
        return self.compute_attention(Q, K, M)
    
    def compute_attention(self, Q, K, M):
        Q = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K = K if getattr(self, 'ln_k', None) is None else self.ln_k(K)

        heads_outputs = []
        for head in range(self.num_heads):
            Q_ = self.W_q[head](Q)
            K_ = self.W_k[head](K)
            V_ = self.W_v[head](K)

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_head), 2)
            A = self.mask_attention(A, M)
            # head_output = Q_ + A.bmm(V_)
            head_output = A.bmm(V_)
            heads_outputs.append(head_output)

        O = torch.cat(heads_outputs, 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        O = self.dropout(O)
        return O
    
class MSAB(pl.LightningModule):
    def __init__(self, dim_in, dim_out, num_heads, ln, dropout):
        super(MSAB, self).__init__()
        self.mmab = MMAB(dim_in, dim_in, dim_out, num_heads, ln, dropout)

    def forward(self, X, M):
        return self.mmab(X, X, M)
    