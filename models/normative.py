import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import sys

from collections import defaultdict

from torch import Tensor
from torch_geometric.utils.repeat import repeat 
from torch_geometric.nn import TopKPooling, SAGPooling

from src.modules import *
from src.mask_modules import *
from src.utils import *
from src.masking import *

class AttentionGate(pl.LightningModule):
    def __init__(self, dim_in: int):
        super(AttentionGate, self).__init__()   
        self.W_res = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LayerNorm(dim_in)
        )
        self.W_up = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LayerNorm(dim_in)
        )
        self.psi = nn.Sequential(
            nn.Linear(dim_in, 1),
            nn.LayerNorm(1)
        )

    def forward(self, res, up):
        res_proj = self.W_res(res)
        up_proj = self.W_up(up)     

        attn_map = F.leaky_relu(res_proj + up_proj)
        # attn_map = torch.sigmoid(self.psi(attn_map))
        attn_map = torch.softmax(self.psi(attn_map), dim=1)

        res_weighted = attn_map * res  
        return res_weighted, attn_map
    
class NodeBasedGraphSetTransformers(pl.LightningModule):
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 num_heads: int,
                 ln: bool,
                 dropout: float):
        super(NodeBasedGraphSetTransformers, self).__init__()
        self.enc_msab1 = MSAB(dim_in, dim_out, num_heads, ln, dropout)
        self.enc_msab2 = MSAB(dim_out, dim_out, num_heads, ln, dropout)
        self.enc_sab1 = SAB(dim_out, dim_out, num_heads, ln, dropout)

    def forward(self, X, M):
        enc1 = self.enc_msab1(X, M)
        enc2 = self.enc_msab2(enc1, M) + enc1
        enc3 = self.enc_sab1(enc2) + enc2
        return enc3
    
class NormativeGUNet(pl.LightningModule):
    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 dropout_ratio: float,
                 num_heads: int,
                 ln: bool,
                 depth: int,
                 pooling_ratio: float,
                 sum_res: bool,
                 lr: float):
        super(NormativeGUNet, self).__init__()
        self.save_hyperparameters()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dropout_ratio = dropout_ratio
        self.num_heads = num_heads
        self.ln = ln
        self.depth = depth
        self.pooling_ratio = repeat(pooling_ratio, depth)
        self.sum_res = sum_res
        self.lr = lr

        # ENCODER #
        self.down_in_hid_net = NodeBasedGraphSetTransformers(dim_in=dim_input, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)
        self.down_hid_net = NodeBasedGraphSetTransformers(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)

        channels = dim_hidden
        in_channels = channels if sum_res else 2 * channels

        # DECODER #
        self.up_in_hid_net = NodeBasedGraphSetTransformers(dim_in=in_channels, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)
        self.up_in_out_net = NodeBasedGraphSetTransformers(dim_in=in_channels, dim_out=dim_input, num_heads=1, ln=ln, dropout=dropout_ratio)

        self.down_nets = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_nets.append(self.down_in_hid_net)
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pooling_ratio[i]))
            self.down_nets.append(self.down_hid_net)

        self.up_nets = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_nets.append(self.up_in_hid_net)
        self.up_nets.append(self.up_in_out_net)

        # ATTENTION #
        self.attns = torch.nn.ModuleList()
        for i in range(depth):
            self.attns.append(AttentionGate(dim_in=dim_hidden))

        # Storage
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}

    def masking(self, x: Tensor, edge_index: Tensor, mask_dim: Tensor, b_map: Tensor, batch_size: int) -> Tensor:
        embed_dim = x.size(-1)
        M = node_mask(edge_index, b_map, mask_dim, batch_size)
        X = x.view(batch_size, mask_dim, embed_dim)
        return X, M

    def forward(self, batch: Tensor) -> Tensor:
        x, edge_index, _, b_map = batch.x, batch.edge_index, batch.edge_weight, batch.batch

        mask_dim = batch[0].num_nodes
        batch_size = b_map.max().item() + 1
        X, M = self.masking(x, edge_index, mask_dim, b_map, batch_size)
        x = self.down_nets[0](X, M)
        x = x.view(-1, x.size(-1))  

        xs = [x]
        edge_indices = [edge_index]
        b_maps = [b_map]
        self.perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, _, b_map, perm, _ = self.pools[i - 1](x, edge_index, None, b_map)

            batch_size = b_map.max().item() + 1
            mask_dim = x.shape[0] // batch_size
            X, M = self.masking(x, edge_index, mask_dim, b_map, batch_size)
            x = self.down_nets[i](X, M)
            x = x.view(-1, x.size(-1))

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                b_maps += [b_map]
            self.perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            b_map = b_maps[j]
            perm = self.perms[j]

            up = torch.zeros_like(res)
            up[perm] = x

            batch_size = b_map.max().item() + 1
            matrix_dim = up.shape[0] // batch_size
            res = res.view(batch_size, matrix_dim, -1)
            up = up.view(batch_size, matrix_dim, -1)

            # Attention gate mechanism
            res_, attn_weight = self.attns[j](res, up)
            concatenation = torch.concat([res_, up], dim=-1)
            concatenation = concatenation.view(-1, concatenation.size(-1))

            mask_dim = concatenation.shape[0] // batch_size
            X, M = self.masking(concatenation, edge_index, mask_dim, b_map, batch_size)
            x = self.up_nets[i](X, M)
            x = x.view(-1, x.size(-1))

        x = F.dropout(x, p=self.dropout_ratio)
        x = x.view(batch_size, mask_dim, x.size(-1))
        return x
    
    def task_loss(self, true_x, recon_x):
        recon_loss = F.mse_loss(recon_x, true_x, reduction='mean')
        return recon_loss
    
    def _step(self, batch, batch_idx):
        recon_x = self.forward(batch)

        x = batch.x
        batch_size = batch.batch.max().item() + 1
        matrix_dim = batch[0].num_nodes
        n_features = x.size(-1)
        true_x = x.view(batch_size, matrix_dim, n_features)

        loss = self.task_loss(true_x, recon_x)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        
        self.train_outputs[self.current_epoch].append({
            'loss': loss
        })

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        
        self.validation_outputs[self.current_epoch].append({
            'loss': loss
        })

        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        
        self.test_outputs[self.current_epoch].append({
            'loss': loss
        })

        self.log('test_loss', loss)
        return loss
    
    def _get_metrics_epoch_end(self, all_loss):
        loss_ = torch.stack(all_loss).mean()
        return loss_
    
    def on_train_epoch_end(self, unused=None):
        all_loss = [elem['loss'] for elem in self.train_outputs[self.current_epoch]]

        mean_loss = self._get_metrics_epoch_end(all_loss)

        print(f"Epoch {self.current_epoch + 1} - Training Loss: {mean_loss.item():.4f}")

        self.train_metrics_per_epoch['mean_loss'] = mean_loss.item()

        del self.train_outputs[self.current_epoch]
        del mean_loss

    def on_validation_epoch_end(self, unused=None):
        all_loss = [elem['loss'] for elem in self.validation_outputs[self.current_epoch]]

        mean_loss = self._get_metrics_epoch_end(all_loss)

        print(f"Epoch {self.current_epoch + 1} - Validation Loss: {mean_loss.item():.4f}")

        self.validation_metrics_per_epoch['mean_loss'] = mean_loss.item()

        del self.validation_outputs[self.current_epoch]
        del mean_loss

    def on_test_epoch_end(self, unused=None):
        all_loss = [elem['loss'] for elem in self.test_outputs[self.current_epoch]]

        mean_loss = self._get_metrics_epoch_end(all_loss)

        print(f"Epoch {self.current_epoch + 1} - Test Loss: {mean_loss.item():.4f}")

        self.test_metrics_per_epoch['mean_loss'] = mean_loss.item()

        del self.test_outputs[self.current_epoch]
        del mean_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
