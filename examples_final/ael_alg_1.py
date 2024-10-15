import torch
import random

def custom(edge_index, x):
    edge_index_shuffled = edge_index[:, torch.randperm(edge_index.size(1))]
    x_noisy = x + torch.randn_like(x)
    
    return edge_index,x