import numpy as np
import torch

from torch_geometric.utils import unbatch_edge_index

# MASKING NODES #
def node_mask(b_ei, b_map, mask_dim, batch_size):
    '''
    Args:
        b_ei (torch.Tensor): Edge index for a batch of graphs.
        b_map (torch.Tensor): Mapping of nodes to graphs.
        mask_dim (int): Mask size.
        batch_size (int): Number of graphs in the batch.
    '''
    adj_matrices = []
    unbatched_edges = unbatch_edge_index(b_ei, b_map)
    
    for i in range(batch_size):
        edge_index_graph = unbatched_edges[i]
        
        adj_matrix = torch.zeros(mask_dim, mask_dim, dtype=torch.float32)
    
        adj_matrix[edge_index_graph[0], edge_index_graph[1]] = 1
        adj_matrix[edge_index_graph[1], edge_index_graph[0]] = 1  

        adj_matrix.fill_diagonal_(0)
        adj_matrices.append(adj_matrix)

    return torch.stack(adj_matrices)