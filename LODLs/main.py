#!/usr/bin/env python3

import os
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Plain', color_scheme='Neutral', call_pdb=1)

# Makes sure hashes are consistent
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from workspace import Workspace

import hydra

@hydra.main(config_path='config', config_name='main.yaml', version_base='1.1')
def main_function(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main_function()
