import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch_geometric

from collections import defaultdict
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, global_add_pool, global_mean_pool, global_max_pool

from src.utils import *

class GNN(pl.LightningModule):
    def __init__(self,
                 conv_type: str,
                 in_channels: int,
                 gnn_intermediate_dim: int,
                 gnn_output_node_dim: int,
                 output_nn_intermediate_dim: int,
                 output_nn_out_dim: int,
                 readout: str,
                 gat_heads: int,
                 gat_dropouts:int,
                 dropout_ratio: float,
                 lr: float,
                 num_layers: int,
                 ):
        super(GNN, self).__init__()
        self.save_hyperparameters()
        self.conv_type = conv_type 
        self.in_channels = in_channels
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.gnn_output_node_dim = gnn_output_node_dim
        self.output_nn_intermediate_dim = output_nn_intermediate_dim
        self.output_nn_out_dim = output_nn_out_dim
        self.readout = readout
        self.gat_heads = gat_heads
        self.gat_dropouts = gat_dropouts
        self.lr = lr
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        # Storage
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}

        assert self.conv_type in ['gcn', 'gat', 'gatv2', 'gin']
        assert self.readout in ['sum', 'mean', 'max', 'dense']

        print(f'Training with {self.num_layers} layers')

        # Convolutional layers #
        convs = []

        # GCN:
        if self.conv_type == 'gcn':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GCNConv(
                        in_channels=self.in_channels, 
                        out_channels=self.gnn_intermediate_dim, 
                        cached=False, 
                        normalize=True), 
                        'x, edge_index -> x'))
                elif i != self.num_layers -1:
                    convs.append((GCNConv(
                        in_channels=self.gnn_intermediate_dim, 
                        out_channels=self.gnn_intermediate_dim, 
                        cached=False, 
                        normalize=True), 
                        'x, edge_index -> x'))
                else:
                    convs.append((GCNConv(
                        in_channels=self.gnn_intermediate_dim, 
                        out_channels=self.gnn_output_node_dim, 
                        cached=False, 
                        normalize=True), 
                        'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

        # GAT:
        elif self.conv_type == 'gat':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GATConv(
                        in_channels=self.in_channels, 
                        out_channels=self.gnn_intermediate_dim, 
                        heads=self.gat_heads, 
                        dropout=self.gat_dropouts, 
                        concat=True), 'x, edge_index -> x'))
                elif i != self.num_layers -1:
                    convs.append((GATConv(
                        in_channels=self.gnn_intermediate_dim * self.gat_heads, 
                        out_channels=self.gnn_intermediate_dim, 
                        heads=self.gat_heads, 
                        dropout=self.gat_dropouts, 
                        concat=True), 
                        'x, edge_index -> x'))
                else:
                    convs.append((GATConv(
                        in_channels=self.gnn_intermediate_dim * self.gat_heads, 
                        out_channels=self.gnn_output_node_dim, 
                        heads=self.gat_heads, 
                        dropout=self.gat_dropouts, 
                        concat=False), 
                        'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

        # GATv2:
        elif self.conv_type == 'gatv2':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GATv2Conv(
                        in_channels=self.in_channels, 
                        out_channels=self.gnn_intermediate_dim, 
                        heads=self.gat_heads,
                        concat=True, 
                        dropout=self.gat_dropouts), 
                        'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GATv2Conv(
                        in_channels=self.gnn_intermediate_dim * self.gat_heads, 
                        out_channels=self.gnn_intermediate_dim,
                        heads=self.gat_heads, 
                        concat=True, 
                        dropout=self.gat_dropouts), 
                        'x, edge_index -> x'))
                else:
                    convs.append((GATv2Conv(
                        in_channels=self.gnn_intermediate_dim * self.gat_heads, 
                        out_channels=self.gnn_output_node_dim,
                        heads=self.gat_heads, 
                        concat=False, 
                        dropout=self.gat_dropouts), 
                        'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))
        
        # GIN:
        elif self.conv_type == 'gin':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GINConv(
                        nn.Sequential(Linear(in_features=self.in_channels, out_features=self.gnn_intermediate_dim),
                                      BatchNorm1d(self.gnn_intermediate_dim),
                                      ReLU(),
                                      Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                                      BatchNorm1d(self.gnn_intermediate_dim),
                                      ReLU())
                    ), 'x, edge_index -> x'))
                elif i != self.num_layers -1:
                    convs.append((GINConv(
                        nn.Sequential(Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                                      BatchNorm1d(self.gnn_intermediate_dim),
                                      ReLU(),
                                      Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                                      BatchNorm1d(self.gnn_intermediate_dim),
                                      ReLU())
                    ), 'x, edge_index -> x'))
                else:
                    convs.append((GINConv(
                        nn.Sequential(Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_output_node_dim),
                                      BatchNorm1d(self.gnn_output_node_dim),
                                      ReLU(),
                                      Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                                      BatchNorm1d(self.gnn_output_node_dim),
                                      ReLU())
                    ), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=False))
                
        # Classification
        in_dim = self.gnn_output_node_dim
        self.output_nn = torch.nn.Sequential(
            Linear(in_features=in_dim, out_features=self.output_nn_intermediate_dim),
            ReLU(),
            Dropout(p=self.dropout_ratio),
            Linear(in_features=self.output_nn_intermediate_dim, out_features=self.output_nn_out_dim)
        )

        self.convs = torch_geometric.nn.Sequential('x, edge_index', convs)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = x.float()
        x = self.convs(x, edge_index)

        if self.readout == 'sum':
            graph_x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            graph_x = global_mean_pool(x, batch)
        elif self.readout == 'max':
            graph_x = global_max_pool(x, batch)

        task_predictions = self.output_nn(graph_x)

        return task_predictions, graph_x, x
    
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
        x, edge_index, batch_ids, y = batch.x, batch.edge_index, batch.batch, batch.y
        task_predictions, graph_x, x = self.forward(x=x, edge_index=edge_index, batch=batch_ids)

        loss = self.task_loss(task_predictions, y)

        return loss, task_predictions, y, graph_x
    
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