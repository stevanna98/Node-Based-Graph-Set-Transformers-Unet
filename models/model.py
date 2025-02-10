import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import sys

from collections import defaultdict

from torch import Tensor
from torch_geometric.utils.repeat import repeat 

from src.modulesv2 import *
from src.mask_modulesv2 import *
from src.sparser import Sparser
from src.utils import *

class Model(pl.LightningModule):
    def __init__(self,
                 dim_input: int,
                 dim_output: int,
                 dim_hidden: int,
                 dim_hidden_: int,
                 output_intermediate_dim: int,
                 dropout_ratio: float,
                 num_heads: int,
                 num_seeds: int,
                 ln: bool,
                 lr: float,
                 l1_lambda: float,
                 lambda_sym: float):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden
        self.dim_hidden_ = dim_hidden_
        self.output_intermediate_dim = output_intermediate_dim
        self.dropout_ratio = dropout_ratio
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.ln = ln
        self.lr = lr

        self.l1_lambda = l1_lambda  
        self.lambda_sym = lambda_sym

        # ENCODER #
        # self.enc_sab = SAB(dim_input, dim_hidden, num_heads, ln, dropout_ratio)
        # self.sparser = nn.Sequential(
        #     nn.Conv2d(dim_hidden, dim_input, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(dim_input),
        #     nn.ReLU()
        # )
        self.sparser = Sparser(dim_input, dim_input, dim_hidden, num_heads, ln)

        self.enc_msab1 = MSAB(dim_input, dim_hidden, num_heads, ln, dropout_ratio)
        self.enc_msab2 = MSAB(dim_hidden, dim_hidden_, num_heads, ln, dropout_ratio)
        self.enc_msab3 = MSAB(dim_hidden_, dim_hidden_, num_heads, ln, dropout_ratio)
        self.enc_sab2 = SAB(dim_hidden_, dim_hidden_, num_heads, ln, dropout_ratio)

        # DECODER #
        self.pma = PMA(dim_hidden_, num_heads, num_seeds, ln, dropout_ratio)
        if self.num_seeds > 1:
            self.dec_sab = SAB(dim_hidden_, dim_hidden_, num_heads, ln, dropout_ratio)

        # CLASSIFIER #
        self.output_mlp = nn.Sequential(
            nn.Linear(dim_hidden_, output_intermediate_dim),
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

    def forward(self, X):
        # x = self.enc_sab(X)
        # x_ = x.permute(0, 2, 1).unsqueeze(-1) 

        # mask = self.sparser(x_)
        # mask = mask.squeeze(-1).permute(0, 2, 1)

        mask = self.sparser(X, X)

        enc1 = self.enc_msab1(X, mask) 
        enc2 = self.enc_msab2(enc1, mask) + enc1
        enc3 = self.enc_msab3(enc2, mask) + enc2
        enc4 = self.enc_sab2(enc3) + enc3

        encoded = self.pma(enc4)
        if self.num_seeds > 1:
            decoded = self.dec_sab(encoded)
            readout = torch.mean(decoded, dim=1, keepdim=True)

            out = self.output_mlp(readout)
        else:
            out = self.output_mlp(encoded)

        return out, mask
    
    def loss_function(self, y_true, y_pred, mask):
        y_true = y_true.view(y_pred.shape)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float())

        l1_reg = self.l1_lambda * torch.sum(torch.abs(mask))
        
        sym_diff = mask - mask.transpose(1, 2)
        sym_reg = self.lambda_sym * torch.sum(sym_diff ** 2)

        loss = bce_loss + l1_reg + sym_reg

        # print(f"bce_loss: {bce_loss.item()}, l1_reg: {l1_reg.item()}, sym_reg: {sym_reg.item()}, total_loss: {loss.item()}")
    
        return loss
    
    def _step(self, batch, batch_idx):
        X, y = batch
        out, mask = self.forward(X)
        loss = self.loss_function(y, out, mask)
        return loss, y, out, mask
    
    def training_step(self, batch, batch_idx):
        loss, ys, outs, mask = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        self.train_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs})
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, ys, outs, mask = self._step(batch, batch_idx)
        self.log('val_loss', loss)
        self.validation_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': outs})
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, ys, outs, mask = self._step(batch, batch_idx)
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





