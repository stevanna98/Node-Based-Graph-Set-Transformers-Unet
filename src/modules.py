import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning.pytorch as pl

# ORIGINAL SET TRANSFORMERS #
class MAB(pl.LightningModule):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln, dropout):
        super(MAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
            self.ln_q = nn.LayerNorm(dim_Q)
            self.ln_k = nn.LayerNorm(dim_K)

        self.fc_o = nn.Linear(dim_V, dim_V)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc_q[0].weight)
        nn.init.xavier_uniform_(self.fc_k[0].weight)
        nn.init.xavier_uniform_(self.fc_v[0].weight)
        nn.init.xavier_uniform_(self.fc_o[0].weight)

    def forward(self, Q, K):
        Q = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K = K if getattr(self, 'ln_k', None) is None else self.ln_k(K)

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(dim_split), 2)

        # Output
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
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
    
