import torch
import numpy as np
import lightning as L
import warnings

from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset, Subset
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
from sklearn.model_selection import StratifiedKFold

from models.model import Model

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Set random seed
seed_value = 27
seed_everything(seed_value, workers=True)

warnings.filterwarnings('ignore')

# Dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.matrices = data
        self.y = labels

    def __len__(self):
        return len(self.matrices)
    
    def __getitem__(self, idx):
        matrix = self.matrices[idx]
        label = self.y[idx]

        matrix = torch.tensor(matrix, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return matrix, label

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--label_dir', type=str, help='Labels directory')
    parser.add_argument('--log_dir', type=str, help='Log directory')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of folds for cross-validation')

    parser.add_argument('--dim_hidden', type=int, default=1024, help='Hidden dimension')
    parser.add_argument('--dim_hidden_', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--output_intermediate_dim', type=int, default=128, help='Intermediate output dimension')
    parser.add_argument('--dim_output', type=int, default=1, help='Output dimension')
    parser.add_argument('--dropout_ratio', type=float, default=0.3, help='Dropout ratio')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of heads')    
    parser.add_argument('--num_seeds', type=int, default=32, help='Number of seeds')
    parser.add_argument('--ln', default=True, help='Layer normalization')
    parser.add_argument('--reg_type', type=str, default='conv2d', help='Regularization type')
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help='L1 regularization lambda')
    parser.add_argument('--l2_lambda', type=float, default=1e-4, help='L2 regularization lambda')
    parser.add_argument('--lambda_sym', type=float, default=1e-3, help='Symmetry regularization lambda')
    parser.add_argument('--mask_thr', type=float, default=1e-8, help='Mask threshold')
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=1e-3)

    args = parser.parse_args()

    # LOAD DATA #
    matrices = np.load(args.data_dir)
    labels = np.load(args.label_dir)
    
    dataset = MyDataset(data=matrices, labels=labels)

    print(f'\nDataset: {len(dataset)} subjects')
    print('Functional Connectivity Matrices shape:', matrices.shape)
    print('Number of classes:', len(np.unique(labels)))

    # K-FOLD CROSS-VALIDATION #
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=seed_value)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(matrices, labels)):
        print(f'\n=== Fold {fold + 1}/{args.kfolds} ===')

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)

        train_len = int(0.8 * len(train_set))
        val_len = len(train_set) - train_len
        train_subset, val_subset = torch.utils.data.random_split(train_set, [train_len, val_len])

        print(f'Train subset: {len(train_subset)} subjects')
        print(f'Validation subset: {len(val_subset)} subjects')
        print(f'Test subset: {len(test_set)} subjects')

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        n_features = dataset.matrices.shape[1]

        model = Model(
            dim_input=n_features,
            dim_output=args.dim_output,
            dim_hidden=args.dim_hidden,
            dim_hidden_=args.dim_hidden_,
            output_intermediate_dim=args.output_intermediate_dim,
            dropout_ratio=args.dropout_ratio,
            num_heads=args.num_heads,
            num_seeds=args.num_seeds,
            ln=args.ln,
            lr=args.lr,
            reg_type=args.reg_type,
            l1_lambda=args.l1_lambda,
            alpha=args.alpha,
            beta=args.beta,
            l2_lambda=args.l2_lambda,
            lambda_sym=args.lambda_sym,
            mask_thr=args.mask_thr
        ).to(device)

        # TRAINING #
        monitor = 'val_loss'
        early_stopping = EarlyStopping(monitor=monitor, patience=15, mode='min')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # callbacks = [early_stopping, lr_monitor]
        callbacks = [lr_monitor]

        tensorboardlogger = TensorBoardLogger(args.log_dir, name=f'fold_{fold}')

        trainer = L.Trainer(
            max_epochs=args.epochs,
            callbacks=callbacks,
            accelerator=device,
            logger=tensorboardlogger,
            enable_progress_bar=True
        )

        print(f'Training on Fold {fold + 1}...')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # TESTING #
        trainer.test(model, dataloaders=test_loader)
        fold_results.append(model.test_metrics_per_epoch)

    # AVERAGE METRICS #
    print('\n=== Average Metrics: ===')
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

        

        





