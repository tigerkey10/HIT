
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as auc3
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pickle
import numpy as np
import torch
import torch.optim as optim

import os
import random
import warnings
warnings.filterwarnings(action='ignore')


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_device(cuda_num=None):
    if cuda_num in [0, 1, 2, 3]:
        cuda = "cuda:"+str(cuda_num)
        device = cuda if torch.cuda.is_available() else "cpu"
    else:  
        device = "cpu"
    return device

def count_parameters(module):
    counts = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return counts

def get_optimizer(params, opt_name, lr=1e-4, w_decay=None):
    if opt_name in ['AdamW', 'adamw', 'AdamW', 'adamW']:
        weight_decay = 0 if w_decay is None else w_decay
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name in ['Adam',  'adam']:
        weight_decay = 0 if w_decay is None else w_decay
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name in ['SGD', 'sgd']:
        weight_decay = 0 if w_decay is None else w_decay
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    
def get_params(args_dict):
    params = {
        'name': args_dict['name'],
        'depth_g' : args_dict['depth_g'],
        'dim_in' : args_dict['dim_in'],
        'dim_out' : args_dict['dim_out'],
        'depth_d' : args_dict['depth_d'],
    }
    return params

def split_data(labels, valid_test_ratio=0.2, seed=315):
    """Splits the nodes into train, validation and test sets."""
    x = list(range(len(labels)))
    y = labels[:, 2]
    
    train, temp, _, y_temp = train_test_split(x, y, test_size=valid_test_ratio, random_state=seed, stratify=y)
    valid, test, _, _ = train_test_split(temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    return train, valid, test

def get_DataLoader(result, batch_size=128, shuffle=True):
    if batch_size is None:
        batch_size = len(result)
    dataset = DatasetID(result[:, 0], result[:, 1], result[:, 2])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class DatasetID(Dataset): 
    def __init__(self, ids_gene, ids_disease, labels):
        self.ids_gene = ids_gene
        self.ids_disease = ids_disease
        self.labels = labels

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx): 
        return self.ids_gene[idx], self.ids_disease[idx], self.labels[idx]
    