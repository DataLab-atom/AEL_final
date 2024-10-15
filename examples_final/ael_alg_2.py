import torch
import random

def custom(edge_index, x):
    random.shuffle(edge_index)
    random_number = random.uniform(0, 1)
    x *= random_number
    
    return edge_index,x