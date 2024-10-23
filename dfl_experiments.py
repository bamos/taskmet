#!/usr/bin/env python

import os
import sys
from functools import partial
import random
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
from copy import deepcopy, copy
import pickle as pkl
from omegaconf import OmegaConf
from IPython.core import ultratb

from taskmet import TaskMet, Predictor
from task import DFL
from metric import Metric

import warnings

warnings.filterwarnings("ignore")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

# Loading configs
base_conf = OmegaConf.load("./config/dfl/default.yaml")
base_conf = OmegaConf.merge(base_conf, OmegaConf.from_cli())
problem_conf = OmegaConf.load("./config/dfl/problem/" + base_conf.problem + ".yaml")
cfg = OmegaConf.merge(base_conf, problem_conf)

# set these after loading the problem for reproducibility
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

print("Loading task")
# create task
task = DFL(cfg)
print("Task loaded")

# create predictor
ipdim, opdim = task.get_modelio_shape()

predictor = Predictor(
    ipdim,
    opdim,
    intermediate_size=500,
    output_activation=task.problem.get_output_activation(),
    **dict(cfg.predictor_kwargs),
)

# create metric
metric = Metric(ipdim, opdim, **cfg.metric_kwargs)

# create taskmet
taskmet = TaskMet(cfg.taskmet_kwargs, predictor, task, metric)

X_train, Y_train, Y_train_aux = task.problem.get_train_data()
X_val, Y_val, Y_val_aux = task.problem.get_val_data()

print("Starting training")
final_metrics = taskmet.train(
    X_train,
    Y_train,
    cfg.taskmet_kwargs.batchsize,
    cfg.taskmet_kwargs.outer_iters,
    Y_train_aux,
    **cfg.taskmet_kwargs.inner,
)
print("Training complete")
