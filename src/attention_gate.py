import lightning.pytorch as pl
import torch.nn as nn
import torch
import torch.nn.functional as F

# class AttentionGate(pl.LightningModule):
#     def __init__(self, dim_in: int):
#         super(AttentionGate, self).__init__()   
#         self.W_res = nn.Sequential(
#             nn.Linear(dim_in, dim_in),
#             nn.LayerNorm(dim_in)
#         )
#         self.W_up = nn.Sequential(
#             nn.Linear(dim_in, dim_in),
#             nn.LayerNorm(dim_in)
#         )
#         self.psi = nn.Sequential(
#             nn.Linear(dim_in, 1),
#             nn.LayerNorm(1)
#         )

#     def forward(self, res, up):
#         res_proj = self.W_res(res)
#         up_proj = self.W_up(up)     

#         attn_map = F.leaky_relu(res_proj + up_proj)
#         # attn_map = torch.sigmoid(self.psi(attn_map))
#         attn_map = torch.softmax(self.psi(attn_map), dim=1)

#         res_weighted = attn_map * res  
#         return res_weighted, attn_map

class AttentionGate(pl.LightningModule):
    def __init__(self, K):
        super(AttentionGate, self).__init__()   
        self.W_res = nn.Sequential(
            nn.Conv2d(K, K, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(K)
        )
        self.W_up = nn.Sequential(
            nn.Conv2d(K, K, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(K)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(K, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )

    def forward(self, res, up):
        res = res.permute(0, 2, 1).unsqueeze(-1)  
        up = up.permute(0, 2, 1).unsqueeze(-1)    

        res_proj = self.W_res(res)
        up_proj = self.W_up(up)     

        attn_map = F.leaky_relu(res_proj + up_proj)
        # attn_map = torch.sigmoid(self.psi(attn_map))
        attn_map = torch.softmax(self.psi(attn_map), dim=1)

        res_weighted = attn_map * res  
        res_weighted = res_weighted.squeeze(-1).permute(0, 2, 1)
        return res_weighted, attn_map.squeeze(-1)