import numpy as np
import torch
import networkx as nx

from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef, classification_report

# def thresholding(matrix, thr):
#     triu_ids = np.triu_indices_from(matrix, k=1)
#     fc_values = matrix[triu_ids]

#     threshold = np.percentile(fc_values, 100 - thr)

#     adjacency_matrix = (matrix >= threshold).astype(int)
#     adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
#     np.fill_diagonal(adjacency_matrix, 0)

#     return adjacency_matrix

def thresholding(matrix, thr=95):
    adj_matrix = np.zeros_like(matrix, dtype=int)
    
    for i in range(matrix.shape[0]):
        threshold = np.percentile(matrix[i, :], thr)
        adj_matrix[i, :] = (matrix[i, :] >= threshold).astype(int)
    
    # Make the matrix symmetric
    adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
    
    # np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix

class GraphDataset(InMemoryDataset):
    def __init__(self, func_matrices, labels, threshold=5, root=None, transform=None, pre_transform=None, weights=None):
        self.func_matrices = func_matrices
        self.labels = labels  
        self.threshold = threshold
        self.weights = weights if weights is not None else np.ones(len(labels))
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        data_list = []

        for idx, matrix in enumerate(self.func_matrices):
            adj_matrix = thresholding(matrix, self.threshold)
            node_features = torch.tensor(matrix, dtype=torch.float32)
            edge_indices = np.array(np.nonzero(adj_matrix))
            edge_indices = torch.tensor(edge_indices, dtype=torch.long)

            graph = Data(
                x=node_features,
                edge_index=edge_indices,
                y=torch.tensor(self.labels[idx], dtype=torch.int64),
                w=torch.tensor(self.weights[idx], dtype=torch.float32)
            )
            data_list.append(graph)
        
        return self.collate(data_list)
    
    def select(self, ids, subset_dir=None):
        selected_func_matrices = [self.func_matrices[i] for i in ids]
        selected_labels = [self.labels[i] for i in ids]
        selected_weights = [self.weights[i] for i in ids]
        return GraphDataset(root=self.root, func_matrices=selected_func_matrices, labels=selected_labels,
                            threshold=self.threshold, weights=selected_weights, transform=self.transform,
                            pre_transform=self.pre_transform)
    
def get_classification_metrics(y_true, y_pred, digits=4):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = None
    
    conf_mat = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1_score_ = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=digits)

    return conf_mat, acc, f1_score_, mcc, roc_auc, report

def print_classification_metrics(metrics):
    cm, acc, f1, mcc, roc_auc, cr = metrics
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')
    print(f'Matthews Correlation Coefficient: {mcc}')
    print(f'ROC AUC: {roc_auc}')
    print(f'Classification Report:\n{cr}')