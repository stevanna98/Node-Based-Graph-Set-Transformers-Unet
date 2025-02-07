import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import sys

from collections import defaultdict

from torch import Tensor
from torch_geometric.utils.repeat import repeat 
from torch_geometric.nn import TopKPooling, SAGPooling

from src.modulesv2 import *
from src.mask_modulesv2 import *
from src.utils import *
from src.masking import *
from src.attention_gate import AttentionGate

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
        print(enc1.shape)
        sys.exit()
        enc2 = self.enc_msab2(enc1, M) + enc1
        enc3 = self.enc_sab1(enc2) + enc2
        return enc3

class NBGSTUnet(pl.LightningModule):
    def __init__(self,
                 dim_input: int,
                 dim_hidden: int,
                 output_intermediate_dim: int,
                 dim_output: int,
                 dropout_ratio: float,
                 num_heads: int,
                 num_seeds: int,
                 ln: bool,
                 depth: int,
                 pooling_ratio: float,
                 attention_gate: bool,
                 sum_res: bool,
                 lr: float,):
        super(NBGSTUnet, self).__init__()
        self.save_hyperparameters()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.output_intermediate_dim = output_intermediate_dim
        self.dim_output = dim_output
        self.dropout_ratio = dropout_ratio
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.ln = ln
        self.depth = depth  
        self.pooling_ratio = repeat(pooling_ratio, depth)
        self.sum_res = sum_res
        self.attention_gate = attention_gate
        self.lr = lr

        # ENCODER #
        self.down_in_hid_net = NodeBasedGraphSetTransformers(dim_in=dim_input, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)
        self.down_hid_net = NodeBasedGraphSetTransformers(dim_in=dim_hidden, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)

        channels = dim_hidden
        in_channels = channels if sum_res else 2 * channels

        # DECODER #
        self.up_in_hid_net = NodeBasedGraphSetTransformers(dim_in=in_channels, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)
        self.up_in_out_net = NodeBasedGraphSetTransformers(dim_in=in_channels, dim_out=dim_hidden, num_heads=num_heads, ln=ln, dropout=dropout_ratio)

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
            self.attns.append(AttentionGate(K=dim_hidden))
    
        # POOLING BY MULTIHEAD ATTENTION #
        self.pma = PMA(dim_hidden, num_heads, num_seeds, ln, dropout_ratio)
        if self.num_seeds > 1:
            self.dec_sab = SAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)

        # OUTPUT CLASSIFIER #
        self.output_mlp = nn.Sequential(
            nn.Linear(dim_hidden, output_intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(output_intermediate_dim, dim_output)
        )

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

            if self.attention_gate:
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
            else:
                x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

                batch_size = b_map.max().item() + 1
                mask_dim = x.shape[0] // batch_size
                X, M = self.masking(x, edge_index, mask_dim, b_map, batch_size)

            x = self.up_nets[i](X, M)
            x = x.view(-1, x.size(-1))

        x = F.dropout(x, p=self.dropout_ratio)
        x = x.view(batch_size, mask_dim, x.size(-1))
        encoded = self.pma(x)
        if self.num_seeds > 1:
            dec = self.dec_sab(encoded)
            readout = torch.mean(dec, dim=1, keepdim=True)
            out = self.output_mlp(readout)
        else:
            out = self.output_mlp(encoded)
        return out
    
    def task_loss(self, y_pred, y_true):
        y_true = y_true.view(y_pred.shape)
        loss = F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float())

        l1_lambda = 1e-4
        l2_lambda = 1e-4
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss += l1_lambda * l1_norm + l2_lambda * l2_norm

        return loss
    
    def _step(self, batch, batch_idx):
        y = batch.y
        out = self.forward(batch)
        loss = self.task_loss(out, y)
        return loss, out, y
    
    def training_step(self, batch, batch_idx):
        loss, outs, ys = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        self.train_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs})
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, outs, ys = self._step(batch, batch_idx)
        self.log('val_loss', loss)
        self.validation_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs})
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, outs, ys = self._step(batch, batch_idx)
        self.log('test_loss', loss)
        self.test_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs})
        return loss
    
    def _get_metrics_epoch_end(self, all_y_true, all_y_pred):
        all_y_pred = torch.sigmoid(all_y_pred.float())
        all_y_pred = torch.where(all_y_pred > 0.5, 1.0, 0.0).long()
        return get_classification_metrics(y_true=all_y_true.long().detach().cpu().numpy(), y_pred=all_y_pred.detach().cpu().numpy())
    
    def on_train_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.train_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.train_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.train_metrics_per_epoch[self.current_epoch] = metrics

        # Print metrics
        print(f"Epoch {self.current_epoch} - Training Metrics:")
        print_classification_metrics(metrics)

        del self.train_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred

    def on_validation_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.validation_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.validation_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.validation_metrics_per_epoch[self.current_epoch] = metrics

        # Print metrics
        print(f"Epoch {self.current_epoch} - Validation Metrics:")
        print_classification_metrics(metrics)

        del self.validation_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred

    def on_test_epoch_end(self, unused=None):     
        all_y_true = [elem['y_true'] for elem in self.test_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.test_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.test_metrics_per_epoch[self.current_epoch] = metrics

        # Print metrics
        print(f"Epoch {self.current_epoch} - Test Metrics:")
        print_classification_metrics(metrics)

        del self.test_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }