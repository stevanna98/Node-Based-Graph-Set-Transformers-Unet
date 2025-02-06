import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import sys

from typing import Callable, List, Union
from torch import Tensor
from torch_geometric.nn import TopKPooling, TransformerConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils.repeat import repeat

from src.utils import *
from collections import defaultdict
from src.modules import *

class GTUNet(pl.LightningModule):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            output_intermediate_dim: int,
            dim_output: int,
            num_heads: int,
            num_seeds: int,
            depth: int,
            lr: int,
            ln: bool,
            edge_dim: int = 1,
            pool_ratios: Union[float, List[float]] = 0.5,
            dropout: float = 0.3,
            sum_res: bool = True,
            act: Union[str, Callable] = 'relu',
    ):
        super(GTUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.lr = lr
        self.num_heads = num_heads 
        self.num_seeds = num_seeds
        self.pool_ratios = repeat(pool_ratios, depth)
        self.dropout = dropout
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        self.down_in_hid_conv = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=1, beta=True)
        self.down_hid_conv = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=1, beta=True)

        channels = hidden_channels
        in_channels = channels if sum_res else 2 * channels

        self.up_in_hid_conv = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=1, beta=True)
        self.up_in_out_conv = TransformerConv(in_channels, out_channels, edge_dim=edge_dim, heads=1, beta=True)

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(self.down_in_hid_conv)
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(self.down_hid_conv)

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(self.up_in_hid_conv)
        self.up_convs.append(self.up_in_out_conv)

        self.reset_parameters()

        # POOLING BY MULTIHEAD ATTENTION #
        self.pma = PMA(out_channels, num_heads, num_seeds, ln, dropout)
        if self.num_seeds > 1:
            self.dec_sab = SAB(out_channels, out_channels, num_heads, ln, dropout)

        # Classifier
        self.output_mlp = nn.Sequential(
            nn.Linear(out_channels, output_intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_intermediate_dim, dim_output)
        )

        # Storage
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        """"""  # noqa: D419
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1)).view(-1, 1)

        batch_size = batch.max().item() + 1
        matrix_dim = x.size(0) // batch_size

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        self.perms = []

        for i in range(1, self.depth + 1):
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            self.perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = self.perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        x = F.dropout(x, p=self.dropout)
        x = x.view(batch_size, matrix_dim, x.size(-1))
        encoded = self.pma(x)
        if self.num_seeds > 1:
            dec = self.dec_sab(encoded)
            readout = torch.mean(dec, dim=1, keepdim=True)
            out = self.output_mlp(readout)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')
    
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
        x, edge_index, batch_, y = batch.x, batch.edge_index, batch.batch, batch.y
        out = self.forward(x, edge_index, batch=batch_)
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }