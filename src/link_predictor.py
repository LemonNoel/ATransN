import sys
import logging
import pickle
import numpy as np
import torch
import numba
from torch.utils.data import DataLoader
from dataset import TestDataset

# @numba.njit
# def index(array, item):
#     for idx, val in np.ndenumerate(array):
#         if val == item:
#             return idx
#     # If no item was found return None, other return types might be a problem due to
#     # numbas type inference.

def index(array, item):
    return (array == item).nonzero()[0].item()

def link_prediction(args, model, dataloaders):
    """
    github.com/DeepGraphLearning/KnowledgeGraphEmbedding/
    """

    model.eval()
    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % (args.gpu_id))

    logs = []

    # import time
    with torch.no_grad():
        for dataloader in dataloaders:
            assert isinstance(dataloader.dataset, TestDataset)
            for positive_sample, filter_bias, mode in dataloader:

                positive_sample = positive_sample.to(device)
                filter_bias = filter_bias.to(device)

                score = model.infer_all(positive_sample, mode)
                score += filter_bias
                
                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)
                        
                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                for i in range(argsort.shape[0]):
                    ranking = 1 + index(argsort[i], positive_arg[i])
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@2': 1.0 if ranking <= 2 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics
