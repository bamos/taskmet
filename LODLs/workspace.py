from functools import partial
import os
import sys

import sys

import argparse
import ast
import torch
import random
import pdb
import matplotlib.pyplot as plt
from copy import deepcopy, copy

import hydra
import setproctitle
import pickle as pkl

from BudgetAllocation import BudgetAllocation
from BipartiteMatching import BipartiteMatching
from PortfolioOpt import PortfolioOpt
from RMAB import RMAB
from CubicTopK import CubicTopK
from models import model_dict
from losses import MSE, get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu

class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.load_problem()

        # Load an ML model to predict the parameters of the problem
        print(f"Building {self.cfg.model} Model...")
        ipdim, opdim = self.problem.get_modelio_shape()
        model_builder = model_dict[self.cfg.model]
        self.model = model_builder(
            num_features=ipdim,
            num_targets=opdim,
            num_layers=self.cfg.layers,
            intermediate_size=500,
            output_activation=self.problem.get_output_activation(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def run(self):
        # Load a loss function to train the ML model on
        #   TODO: Figure out loss function "type" for mypy type checking. Define class/interface?
        print(f"Loading {self.cfg.loss} Loss Function...")
        loss_fn = get_loss_fn(
            self.cfg.loss,
            self.problem,
            sampling=self.cfg.sampling,
            num_samples=self.cfg.numsamples,
            rank=self.cfg.quadrank,
            sampling_std=self.cfg.samplingstd,
            quadalpha=self.cfg.quadalpha,
            lr=self.cfg.losslr,
            serial=self.cfg.serial,
            dflalpha=self.cfg.dflalpha,
        )

        # Train neural network with a given loss function
        print(f"Training {self.cfg.model} model on {self.cfg.loss} loss...")
        #   Move everything to GPU, if available
        if torch.cuda.is_available():
            move_to_gpu(self.problem)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)

        # Get data
        X_train, Y_train, Y_train_aux = self.problem.get_train_data()
        X_val, Y_val, Y_val_aux = self.problem.get_val_data()
        X_test, Y_test, Y_test_aux = self.problem.get_test_data()

        best = (float("inf"), None)
        time_since_best = 0
        for iter_idx in range(self.cfg.iters):
            # Check metrics on val set
            if iter_idx % self.cfg.valfreq == 0:
                self.save()
                # Compute metrics
                datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
                metrics = print_metrics(datasets, self.model, self.problem, self.cfg.loss, loss_fn, f"Iter {iter_idx},")

                # Save model if it's the best one
                if best[1] is None or metrics['val']['loss'] < best[0]:
                    best = (metrics['val']['loss'], deepcopy(self.model))
                    time_since_best = 0

                # Stop if model hasn't improved for patience steps
                if self.cfg.earlystopping and time_since_best > self.cfg.patience:
                    break

            # Learn
            losses = []
            for i in random.sample(range(len(X_train)), min(self.cfg.batchsize, len(X_train))):
                pred = self.model(X_train[i]).squeeze()
                losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i))
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            time_since_best += 1

        if self.cfg.earlystopping:
            self.model = best[1]

        # Document how well this trained model does
        print("\nBenchmarking Model...")
        # Print final metrics
        datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
        print_metrics(datasets, self.model, self.problem,
                      self.cfg.loss, loss_fn, "Final")

        #   Document the value of a random guess
        objs_rand = []
        for _ in range(10):
            Z_test_rand = self.problem.get_decision(torch.rand_like(Y_test), aux_data=Y_test_aux, isTrain=False)
            objectives = self.problem.get_objective(Y_test, Z_test_rand, aux_data=Y_test_aux)
            objs_rand.append(objectives)
        print(f"\nRandom Decision Quality: {torch.stack(objs_rand).mean().item()}")

        #   Document the optimal value
        Z_test_opt = self.problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
        objectives = self.problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
        print(f"Optimal Decision Quality: {objectives.mean().item()}")
        print()

    # #   Plot predictions on test data
    # plt.scatter(Y_test.sum(dim=-1).flatten().detach().tolist(), pred.sum(dim=-1).flatten().detach().tolist(), )
    # plt.title(self.cfg.loss)
    # plt.xlabel("True")
    # plt.ylabel("Predicted")
    # plt.xlim([0, 0.5])
    # plt.ylim([0, 0.5])
    # plt.show()

    def save(self, tag='latest'):
        path = os.path.join(self.work_dir, f'{tag}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def __getstate__(self):
        d = copy(self.__dict__)
        del d['problem']
        return d


    def __setstate__(self, d):
        self.__dict__ = d
        self.load_problem()

    def load_problem(self):
        print(f"Loading {self.cfg.problem} Problem...")
        init_problem = partial(init_if_not_saved, load_new=self.cfg.loadnew)
        if self.cfg.problem == 'budgetalloc':
            problem_kwargs =    {'num_train_instances': self.cfg.instances,
                                'num_test_instances': self.cfg.testinstances,
                                'num_targets': self.cfg.numtargets,
                                'num_items': self.cfg.numitems,
                                'budget': self.cfg.budget,
                                'num_fake_targets': self.cfg.fakefeatures,
                                'rand_seed': self.cfg.seed,
                                'val_frac': self.cfg.valfrac,}
            problem = init_problem(BudgetAllocation, problem_kwargs)
        elif self.cfg.problem == 'cubic':
            problem_kwargs =    {'num_train_instances': self.cfg.instances,
                                'num_test_instances': self.cfg.testinstances,
                                'num_items': self.cfg.numitems,
                                'budget': self.cfg.budget,
                                'rand_seed': self.cfg.seed,
                                'val_frac': self.cfg.valfrac,}
            problem = init_problem(CubicTopK, problem_kwargs)
        elif self.cfg.problem == 'bipartitematching':
            problem_kwargs =    {'num_train_instances': self.cfg.instances,
                                'num_test_instances': self.cfg.testinstances,
                                'num_nodes': self.cfg.nodes,
                                'val_frac': self.cfg.valfrac,
                                'rand_seed': self.cfg.seed,}
            problem = init_problem(BipartiteMatching, problem_kwargs)
        elif self.cfg.problem == 'rmab':
            problem_kwargs =    {'num_train_instances': self.cfg.instances,
                                'num_test_instances': self.cfg.testinstances,
                                'num_arms': self.cfg.numarms,
                                'eval_method': self.cfg.eval,
                                'min_lift': self.cfg.minlift,
                                'budget': self.cfg.rmabbudget,
                                'gamma': self.cfg.gamma,
                                'num_features': self.cfg.numfeatures,
                                'num_intermediate': self.cfg.scramblingsize,
                                'num_layers': self.cfg.scramblinglayers,
                                'noise_std': self.cfg.noisestd,
                                'val_frac': self.cfg.valfrac,
                                'rand_seed': self.cfg.seed,}
            problem = init_problem(RMAB, problem_kwargs)
        elif self.cfg.problem == 'portfolio':
            problem_kwargs =    {'num_train_instances': self.cfg.instances,
                                'num_test_instances': self.cfg.testinstances,
                                'num_stocks': self.cfg.stocks,
                                'alpha': self.cfg.stockalpha,
                                'val_frac': self.cfg.valfrac,
                                'rand_seed': self.cfg.seed,}
            problem = init_problem(PortfolioOpt, problem_kwargs)
        self.problem = problem
