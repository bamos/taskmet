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
from models import model_dict, MetricModel
from losses import MSE, get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.load_problem()

        ipdim, opdim = self.problem.get_modelio_shape()
        if self.cfg.loss == "metric":
            model_builder = MetricModel
        else:
            model_builder = model_dict[self.cfg.pred_model]
        self.model = model_builder(
            num_features=ipdim,
            num_targets=opdim,
            num_layers=self.cfg.layers,
            intermediate_size=500,
            output_activation=self.problem.get_output_activation(),
            **dict(self.cfg.model_kwargs),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def run(self):
        loss_fn = get_loss_fn(
            self.cfg.loss if not isinstance(self.model, MetricModel) else "dfl",
            self.problem,
            **dict(self.cfg.loss_kwargs),
        )

        #   Move everything to GPU, if available
        if torch.cuda.is_available():
            move_to_gpu(self.problem)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)

        if hasattr(self.problem, "plot"):
            self.problem.plot("latest.png", self)

        # Get data
        X_train, Y_train, Y_train_aux = self.problem.get_train_data()
        X_val, Y_val, Y_val_aux = self.problem.get_val_data()
        X_test, Y_test, Y_test_aux = self.problem.get_test_data()

        # TODO Set batch sizes/num_iters/opts somewhere else?
        if self.cfg.loss == "metric":
            self.model.update_predictor(
                X_train, Y_train, num_iters=self.cfg.num_inner_iters_init
            )

        best = (float("inf"), None)
        time_since_best = 0
        for iter_idx in range(self.cfg.iters):
            # Check metrics on val set
            if iter_idx % self.cfg.valfreq == 0:
                self.save()

                # TODO: copy instead of re-plotting
                if hasattr(self.problem, "plot"):
                    self.problem.plot("latest.png", self)
                    self.problem.plot(f"vis_{iter_idx:05d}.png", self)

                # print(f'  metric weight value: {self.model.metric_params[0].item():.2f}')
                # print(f'  metric weight grad: {self.model.metric_params[0].grad.item():.2f}')

                # Compute metrics
                datasets = [
                    (X_train, Y_train, Y_train_aux, "train"),
                    (X_val, Y_val, Y_val_aux, "val"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.model,
                    self.problem,
                    self.cfg.loss,
                    loss_fn,
                    f"Iter {iter_idx},",
                )

                # Save model if it's the best one
                assert not self.cfg.earlystopping  # TODO
                # if best[1] is None or metrics["val"]["loss"] < best[0]:
                #     best = (metrics["val"]["loss"], deepcopy(self.model))
                #     time_since_best = 0

                # Stop if model hasn't improved for patience steps
                # if self.cfg.earlystopping and time_since_best > self.cfg.patience:
                #     break

            # Learn
            losses = []
            for i in random.sample(
                range(len(X_train)), min(self.cfg.batchsize, len(X_train))
            ):
                pred = self.model(X_train[i]).squeeze()
                losses.append(
                    loss_fn(
                        pred,
                        Y_train[i],
                        aux_data=Y_train_aux[i],
                        partition="train",
                        index=i,
                    )
                )
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            time_since_best += 1

            # TODO Set batch sizes/num_iters/opts somewhere else?
            if self.cfg.loss == "metric":
                self.model.update_predictor(
                    X_train, Y_train, num_iters=self.cfg.num_inner_iters
                )

        assert not self.cfg.earlystopping  # TODO
        # if self.cfg.earlystopping:
        #     self.model = best[1]

        # Document how well this trained model does
        print("\nBenchmarking Model...")
        # Print final metrics
        datasets = [
            (X_train, Y_train, Y_train_aux, "train"),
            (X_val, Y_val, Y_val_aux, "val"),
            (X_test, Y_test, Y_test_aux, "test"),
        ]
        metrics = print_metrics(
            datasets, self.model, self.problem, self.cfg.loss, loss_fn, "Final"
        )

        #   Document the value of a random guess
        objs_rand = []
        for _ in range(10):
            Z_test_rand = self.problem.get_decision(
                torch.rand_like(Y_test), aux_data=Y_test_aux, isTrain=False
            )
            objectives = self.problem.get_objective(
                Y_test, Z_test_rand, aux_data=Y_test_aux
            )
            objs_rand.append(objectives)
        random_dq = torch.stack(objs_rand).mean().item()
        print(f"\nRandom Decision Quality: {random_dq:.2f} (normalized: 0)")

        #   Document the optimal value
        Z_test_opt = self.problem.get_decision(
            Y_test, aux_data=Y_test_aux, isTrain=False
        )
        objectives = self.problem.get_objective(Y_test, Z_test_opt, aux_data=Y_test_aux)
        optimal_dq = objectives.mean().item()
        print(f"Optimal Decision Quality: {optimal_dq:.2f} (normalized: 1)")
        print()
        self.save()

        dq_range = optimal_dq - random_dq
        test_dq = metrics["test"]["objective"]
        normalized_test_dq = (test_dq - random_dq) / dq_range
        print(f"Normalized Test Decision Quality: {normalized_test_dq:.2f}")

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def __getstate__(self):
        d = copy(self.__dict__)
        del d["problem"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.load_problem()

    def load_problem(self):
        init_problem = partial(init_if_not_saved, load_new=self.cfg.loadnew)
        problem_cls = hydra.utils._locate(self.cfg.problem_cls)
        self.problem = init_problem(problem_cls, dict(self.cfg.problem_kwargs))
