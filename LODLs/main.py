#!/usr/bin/env python3

from functools import partial
import os
import sys

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

# Makes sure hashes are consistent
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import ast
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import random
import pdb
import matplotlib.pyplot as plt
from copy import deepcopy

import hydra
import setproctitle

from BudgetAllocation import BudgetAllocation
from BipartiteMatching import BipartiteMatching
from PortfolioOpt import PortfolioOpt
from RMAB import RMAB
from CubicTopK import CubicTopK
from models import model_dict
from losses import MSE, get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu
# from workspace import Workspace
# import workspace

@hydra.main(config_path='config', config_name='main.yaml', version_base='1.1')
def main_function(cfg):
    import ipdb; ipdb.set_trace()
    # workspace = Workspace(cfg)
    # workspace.run()

if __name__ == '__main__':
    main_function()
