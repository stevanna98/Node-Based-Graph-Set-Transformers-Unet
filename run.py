import torch
import numpy as np
import pandas as pd
import os
import lightning as L
import warnings
import sys
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from models.set_transformers_graph_unet import NBGSTUnet
from models.gnn import GNN
from models.gtunet import GTUNet
from src.utils import GraphDataset

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Set random seed
seed_value = 42
seed_everything(seed_value, workers=True)

warnings.filterwarnings('ignore')

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='GTUNet', help='Model type to train')
    parser.add_argument('--conv_type', type=str, default='trans', help='Convolutional layer type')

    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--label_dir', type=str, help='Labels directory')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    parser.add_argument('--thr', type=int, default=10, help='Threshold for functional connectivity matrices')

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds for cross-validation')

    parser.add_argument('--dim_hidden', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--gnn_intermediate_dim', type=int, default=128, help='Intermediate GNN dimension')
    parser.add_argument('--out_channels', type=int, default=128, help='Output channels') # For GTUNet
    parser.add_argument('--gnn_out_node_dim', type=int, default=128, help='Output node dimension')
    parser.add_argument('--output_intermediate_dim', type=int, default=64, help='Intermediate output dimension')
    parser.add_argument('--dim_output', type=int, default=1, help='Output dimension')
    parser.add_argument('--dropout_ratio', type=float, default=0.3, help='Dropout ratio')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--num_seeds', type=int, default=32, help='Number of seeds')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--gat_heads', type=int, default=8, help='Number of GAT heads')
    parser.add_argument('--readout', type=str, default='mean', help='Readout function')
    parser.add_argument('--ln', default=True, help='Layer normalization')
    parser.add_argument('--depth', type=int, default=3, help='Depth of GTUNet')
    parser.add_argument('--pooling_ratio', type=float, default=0.7, help='TopK pooling ratio')
    parser.add_argument('--sum_res', default=True, help='Sum residual')

    args = parser.parse_args()

    # LOAD DATA #
    fc_matrices = np.load(args.data_dir)
    labels = np.load(args.label_dir).reshape(-1, 1)
    print('Functional Connectivity Matrices shape:', fc_matrices.shape)
    print('Labels shape:', labels.shape)

    dataset = GraphDataset(
        func_matrices=fc_matrices,
        labels=labels,
        threshold=args.thr
    )

    print(f'\nDataset: {len(dataset)} subjects')
    print(f'Number of node features: {dataset.num_node_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # Print a random sample
    rng = np.random.default_rng()
    random_sample = rng.integers(0, len(dataset))
    print(f'\nRandom sample: {random_sample}')
    print(dataset[random_sample])

    # K-FOLD CROSS-VALIDATION #
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=seed_value)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset.y)):
        print(f'\n=== Fold {fold + 1}/{args.kfolds} ===')

        train_set = dataset[train_idx]
        test_set = dataset[test_idx]

        print(f'Train set: {len(train_set)} subjects')
        print(f'Test set: {len(test_set)} subjects')

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        n_features = dataset.num_node_features

        # MODEL # 
        if args.model_type == 'NBGSTUnet':
            model = NBGSTUnet(
                dim_input=n_features,
                dim_hidden=args.dim_hidden,
                output_intermediate_dim=args.output_intermediate_dim,
                dim_output=args.dim_output,
                dropout_ratio=args.dropout_ratio,
                num_heads=args.num_heads,
                num_seeds=args.num_seeds,
                ln=args.ln,
                depth=args.depth,
                pooling_ratio=args.pooling_ratio,
                sum_res=args.sum_res,
                lr=args.lr
            ).to(device)
        elif args.model_type == 'GNN':
            model = GNN(
                conv_type=args.conv_type,
                in_channels=n_features,
                gnn_intermediate_dim=args.dim_hidden,
                gnn_output_node_dim=args.gnn_intermediate_dim,
                output_nn_intermediate_dim=args.output_intermediate_dim,
                output_nn_out_dim=args.dim_output,
                num_layers=args.num_layers,
                gat_heads=args.gat_heads,
                readout=args.readout,
                gat_dropouts=args.dropout_ratio,
                dropout_ratio=args.dropout_ratio,
                lr=args.lr
            ).to(device)
        elif args.model_type == 'GTUNet':
            model = GTUNet(
                in_channels=n_features,
                hidden_channels=args.dim_hidden,
                out_channels=args.out_channels,
                output_intermediate_dim=args.output_intermediate_dim,
                dim_output=args.dim_output,
                num_heads=args.num_heads,
                num_seeds=args.num_seeds,
                depth=args.depth,
                lr=args.lr,
                ln=args.ln,
                pool_ratios=args.pooling_ratio,
                dropout=args.dropout_ratio,
                sum_res=args.sum_res,
                conv_layer=args.conv_type
            ).to(device)

        # TRAINING #
        monitor = 'val_loss'
        early_stopping = EarlyStopping(monitor=monitor, patience=30, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks = [early_stopping, lr_monitor]

        tensorboardlogger = TensorBoardLogger(args.log_dir, name=f'{args.model_type}_fold_{fold}')

        trainer = L.Trainer(
            max_epochs=args.epochs,
            callbacks=callbacks,
            accelerator=device,
            logger=tensorboardlogger,
            enable_progress_bar=True
        )

        print(f'Training on Fold {fold + 1}...')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        fold_results.append(model.validation_metrics_per_epoch)

    # AVERAGE METRICS #out
    accs = []
    f1s = []
    roc_aucs = []
    mccs = []
    for fold in range(args.kfolds):
        for key, value in fold_results[fold].items():
            accs.append(value[1])
            f1s.append(value[2])
            mccs.append(value[3])
            roc_aucs.append(value[4])

    print(f'\nAverage Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'Average F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'Average MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}')
    print(f'Average ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}')
            
if __name__ == '__main__':
    print(f'Using device: {device}')
    main()

