import torch
import random

def custom(edge_index, x):
    random.shuffle(edge_index)
    random.shuffle(x)
    
    return edge_index,x