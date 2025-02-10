import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl

class MAB(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln, dropout):
        super(MAB, self).__init__()
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

    def forward(self, Q, K):
        Q = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K = K if getattr(self, 'ln_k', None) is None else self.ln_k(K)

        heads_outputs = []
        for head in range(self.num_heads):
            Q_ = self.W_q[head](Q)
            K_ = self.W_k[head](K)
            V_ = self.W_v[head](K)

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_head), 2)
            head_output = A.bmm(V_)
            heads_outputs.append(head_output)

        O = torch.cat(heads_outputs, 2)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        O = self.dropout(O)
        return O
    
class SAB(pl.LightningModule):
    def __init__(self, dim_in, dim_out, num_heads, ln, dropout):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln, dropout)

    def forward(self, X):
        return self.mab(X, X)

class PMA(pl.LightningModule):
    def __init__(self, dim, num_heads, num_seeds, ln, dropout):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln, dropout)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    