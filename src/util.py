import torch
import math
import random
import numpy as np
import logging
import json
import os
from torch.optim.lr_scheduler import LambdaLR
from collections import namedtuple

def anneal_fn(fn, t, T, init_value=0.0, final_value=1.0):
    if not fn or fn == "none":
        return final_value
    elif fn == "logistic":
        K = 8 / T
        return float(init_value + (final_value-init_value)/(1+np.exp(-K*(t-T/2))))
    elif fn == "linear":
        return float(init_value + (final_value-init_value) * t/T)
    elif fn == "cosine":
        return float(init_value + (final_value-init_value) * (1 - math.cos(math.pi * t/T))/2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, init_value, final_value)
        else:
            return anneal_fn(fn.split("_", 1)[1], t-R*T, R*T, final_value, init_value)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R*T, init_value, final_value)
        else:
            return final_value
    else:
        raise NotImplementedError

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles=4.0, last_epoch=-1, min_percent=0.0):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return min_percent
        return max(min_percent, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def str2bool(x):
    x = x.lower()
    if x == "y" or x == "yes" or x == "t" or x == "true":
        return True
    elif x == "n" or x == "no" or x == "f" or x == "false":
        return False
    raise NotImplementedError

def str2list(x):
    return x.split(",")


def set_seed(seed=1029):
    """set seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def set_logger(save_path):
    """set logger handlers"""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=save_path,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    """print logging metrics"""
    for metric in metrics:
        logging.info('%s %s at step %d: %f' %
                     (mode, metric, step, metrics[metric]))


def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f, object_hook=lambda d: namedtuple(
            'args', d.keys())(*d.values()))
    return config


def save_config(config, filename):
    with open(filename, "w") as f:
        json.dump(vars(config), f)


def save_model(save_variables, save_path):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    saved_dict = {}
    for k, v in save_variables.items():
        try:
            saved_dict[k] = v.state_dict()
        except:
            saved_dict[k] = v
    torch.save(saved_dict, save_path)

    # torch.save(model.entity_embedding, os.path.join(save_path, 'entity_%d.pt' % (step)))
    # torch.save(model.relation_embedding, os.path.join(save_path, 'relation_%d.pt' % (step)))
