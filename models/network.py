import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys

from collections import defaultdict

from src.modules import *
from src.mask_modules import *
from src.utils import *
from src.masking import *

class MaskedAttentionGraphs(pl.LightningModule):
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hidden: int,
                 dim_intermediate_output: int,
                 dropout_ratio: float,
                 num_heads: int,
                 num_seeds: int,
                 ln: bool,
                 lr: float):
        super(MaskedAttentionGraphs, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout_ratio = dropout_ratio

        # Encoder
        self.enc_msab1 = MSAB(dim_input, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_msab2 = MSAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_msab3 = MSAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_msab4 = MSAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_sab1 = SAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)

        self.pma = PMA(dim_hidden, num_heads, num_seeds, ln, dropout_ratio)

        self.num_seeds = num_seeds
        if self.num_seeds > 1:
            self.dec_sab = SAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)

        self.dec_sab = SAB(dim_hidden, dim_hidden, num_heads, ln, dropout_ratio)

        self.output_mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_intermediate_output),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(dim_intermediate_output, dim_output)
        )

        # Storage 
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}

    def forward(self, X, M):
        enc1 = self.enc_msab1(X, M)
        enc2 = self.enc_msab2(enc1, M) + enc1
        enc3 = self.enc_msab3(enc2, M) + enc2
        enc4 = self.enc_msab4(enc3, M) + enc3
        enc5 = self.enc_sab1(enc4) + enc4

        encoded = self.pma(enc5)

        if self.num_seeds > 1:
            dec = self.dec_sab(encoded)
            dec = torch.mean(dec, dim=1, keepdim=True)

            output = self.output_mlp(dec)
        else:
            output = self.output_mlp(encoded)

        return output
    
    def task_loss(self, y_pred, y_true):
        y_true = y_true.view(y_pred.shape)
        loss = F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float())
        return loss
    
    def _step(self, batch, batch_idx):
        x, b_ei, b_map, y = batch.x, batch.edge_index, batch.batch, batch.y
        batch_size = b_map.max().item() + 1
        mask_dim = batch[0].num_nodes
        n_features = x.size(-1)

        M = node_mask(b_ei, b_map, mask_dim, batch_size)
        X = x.view(batch_size, mask_dim, n_features)
        # X = torch.triu(X, diagonal=1)

        out = self.forward(X, M)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
