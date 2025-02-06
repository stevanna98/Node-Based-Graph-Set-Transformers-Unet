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

    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--label_dir', type=str, help='Labels directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    parser.add_argument('--thr', type=int, default=10, help='Threshold for functional connectivity matrices')

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--kfolds', type=int, default=10, help='Number of folds for cross-validation')

    parser.add_argument('--out_channels', type=int, default=128, help='Output channels') # For GTUNet
    parser.add_argument('--output_intermediate_dim', type=int, default=64, help='Intermediate output dimension')
    parser.add_argument('--dim_output', type=int, default=1, help='Output dimension')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout ratio')
    parser.add_argument('--ln', default=True, help='Layer normalization')
    parser.add_argument('--depth', type=int, default=3, help='Depth of GTUNet')
    parser.add_argument('--sum_res', default=False, help='Sum residual')

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

    # HYPERPARAMETERS GRID
    # NBGSTUnet
    param_grid_1 = {
        'batch_size': [32, 64],
        'num_heads': [4, 8, 16],
        'dim_hidden': [64, 128, 256],
        'seeds': [8, 16, 32], 
        'pooling_ratio': [0.7, 0.9]
    }
    # GTUNet
    param_grid_2 = {
        'batch_size': [32, 64],
        'dim_hidden': [64, 128, 256],
        'pooling_ratio': [0.7, 0.9],
        'seeds': [8, 16, 32]
    }

    best_params = None
    best_val_scores = float('inf')
    best_results = []

    for params in ParameterGrid(param_grid_1):
        print(f'Tuning with params: {params}')

        # K-FOLD CROSS-VALIDATION #
        skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=seed_value)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset.y)):
            print(f'\n=== Fold {fold + 1}/{args.kfolds} ===')

            train_set = dataset[train_idx]
            test_set = dataset[test_idx]

            print(f'Train set: {len(train_set)} subjects')
            print(f'Test set: {len(test_set)} subjects')

            train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False)

            n_features = dataset.num_node_features

            # MODEL # 
            if args.model_type == 'NBGSTUnet':
                model = NBGSTUnet(
                    dim_input=n_features,
                    dim_hidden=params['dim_hidden'],
                    output_intermediate_dim=args.output_intermediate_dim,
                    dim_output=args.dim_output,
                    dropout_ratio=args.dropout_ratio,
                    num_heads=params['num_heads'],
                    num_seeds=params['seeds'],
                    ln=args.ln,
                    depth=args.depth,
                    pooling_ratio=params['pooling_ratio'],
                    sum_res=args.sum_res,
                    lr=args.lr
                ).to(device)
            elif args.model_type == 'GTUNet':
                model = GTUNet(
                    in_channels=n_features,
                    hidden_channels=params['dim_hidden'],
                    out_channels=args.out_channels,
                    output_intermediate_dim=args.output_intermediate_dim,
                    dim_output=args.dim_output,
                    num_heads=args.num_heads,
                    num_seeds=params['seeds'],
                    depth=args.depth,
                    lr=args.lr,
                    ln=args.ln,
                    pool_ratios=params['pooling_ratio'],
                    dropout=args.dropout_ratio,
                    sum_res=args.sum_res
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

        val_f1_score = []
        for fold in range(args.kfolds):
            for key, value in fold_results[fold].items():
                val_f1_score.append(value[2])

        avg_val_score = np.mean(val_f1_score)
        if avg_val_score < best_val_scores:
            print('BEST PARAMETERS UPDATED')
            best_val_scores = avg_val_score
            best_params = params
            best_results = fold_results

    # AVERAGE METRICS #
    accs = []
    f1s = []
    roc_aucs = []
    mccs = []
    for fold in range(args.kfolds):
        for key, value in best_results[fold].items():
            accs.append(value[1])
            f1s.append(value[2])
            mccs.append(value[3])
            roc_aucs.append(value[4])

    print(f'\nBest parameters: {best_params}')
    print(f'\nAverage Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'Average F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'Average MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}')
    print(f'Average ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}')

    # Save results to CSV
    avg_metrics = {
        'Metric': ['Accuracy', 'F1-Score', 'MCC', 'ROC-AUC'],
        'Average': [np.mean(accs), np.mean(f1s), np.mean(mccs), np.mean(roc_aucs)],
        'Std Dev': [np.std(accs), np.std(f1s), np.std(mccs), np.std(roc_aucs)]
    }

    avg_metrics_df = pd.DataFrame(avg_metrics)

    save_out_path = os.path.join(args.output_dir, f'avg_metrics_mag_{args.thr}.csv')
    avg_metrics_df.to_csv(save_out_path, index=False)
    print(f'\nResults saved to: {save_out_path}')
            
if __name__ == '__main__':
    print(f'Using device: {device}')
    main()

