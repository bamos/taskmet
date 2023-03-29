from functools import partial
import os
import sys

import sys

import argparse
import ast
import torch
import random
import numpy as np
import json
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
from logger import Logger


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.load_problem()

        self.logger = Logger(os.getcwd(), "log.txt")

        # set these after loading the problem for reproducibility
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        ipdim, opdim = self.problem.get_modelio_shape()
        if self.cfg.loss == "metric":
            model_builder = MetricModel
            lr = cfg.metric_lr
        else:
            model_builder = model_dict[self.cfg.pred_model]
            lr = cfg.pred_lr
        self.model = model_builder(
            num_features=ipdim,
            num_targets=opdim,
            num_layers=self.cfg.layers,
            intermediate_size=500,
            output_activation=self.problem.get_output_activation(),
            **dict(self.cfg.model_kwargs),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_iter = 0
        self.best_val_loss = float("inf")

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

        # if hasattr(self.problem, "plot"):
        #     self.problem.plot("latest.png", self)

        # Get data
        X_train, Y_train, Y_train_aux = self.problem.get_train_data()
        X_val, Y_val, Y_val_aux = self.problem.get_val_data()
        X_test, Y_test, Y_test_aux = self.problem.get_test_data()

        # TODO Set batch sizes/num_iters/opts somewhere else?
        if self.cfg.loss == "metric" and self.train_iter == 0:
            # self.model.pretrain_metric(X_train)
            self.model.update_predictor(
                X_train, Y_train, num_iters=self.cfg.num_inner_iters_init
            )
            self.save()
            self.save("best")

        while self.train_iter < self.cfg.iters:
            # Check metrics on val set
            if self.train_iter % self.cfg.valfreq == 0:
                self.save()

                # Compute metrics
                losses = []
                DQ = []
                mse = []
                for i in range(len(X_val)):
                    pred = self.model(X_val[i]).squeeze()
                    losses.append(
                        loss_fn(
                            pred,
                            Y_val[i],
                            aux_data=Y_val_aux[i],
                            partition="validation",
                            index=i,
                        ).item()
                    )
                    Zs_pred = self.problem.get_decision(
                        pred, aux_data=Y_val_aux[i], isTrain=True
                    )
                    DQ.append(
                        self.problem.get_objective(
                            Y_val[i], Zs_pred, aux_data=Y_val_aux[i]
                        ).item()
                    )
                    mse.append((pred - Y_val[i]).pow(2).mean().item())

                metrics = {
                    "loss": np.mean(losses),
                    "DQ": np.mean(DQ),
                    "MSE": np.mean(mse),
                    "iter": self.train_iter,
                }

                print(f"val | Iteration {self.train_iter}: {metrics}")

                self.logger.log(val_metrics=metrics)
                # Save model if it's the best one
                if metrics["loss"] < self.best_val_loss:
                    self.save("best")
                    self.best_val_loss = metrics["loss"]

            # Learn
            losses = []
            DQ = []
            mse = []
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
                Zs_pred = self.problem.get_decision(
                    pred, aux_data=Y_train_aux[i], isTrain=True
                )
                DQ.append(
                    self.problem.get_objective(
                        Y_train[i], Zs_pred, aux_data=Y_train_aux[i]
                    ).item()
                )
                mse.append((pred - Y_train[i]).pow(2).mean().item())

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()

            # check of gradient is nan
            for param in self.model.parameters():
                assert not torch.isnan(param.grad).any()

            self.optimizer.step()

            metrics = {
                "outer_loss": loss.item(),
                "DQ": np.mean(DQ),
                "MSE": np.mean(mse),
                "iter": self.train_iter,
            }

            if self.cfg.loss == "metric":
                predictor_metric = self.model.update_predictor(
                    X_train, Y_train, num_iters=self.cfg.num_inner_iters
                )
                metrics.update(predictor_metric)
            if self.train_iter % self.cfg.valfreq == 0:
                print(f"train - Iteration {self.train_iter}: {metrics}")
            self.logger.log(train_metrics=metrics)
            self.train_iter += 1

        self.save()
        # self.logger.plot()
        # self.logger.save()

    def test(self):
        # Document how well this trained model does
        print("\nBenchmarking Model...")

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

        X_train, Y_train, Y_train_aux = self.problem.get_train_data()
        X_val, Y_val, Y_val_aux = self.problem.get_val_data()
        X_test, Y_test, Y_test_aux = self.problem.get_test_data()

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

        test_stats = {
            "random_dq_unnorm": random_dq,
            "optimal_dq_unnorm": optimal_dq,
            "test_dq_unnorm": test_dq,
            "test_dq_norm": normalized_test_dq,
        }

        fname = "test_stats.json"
        print(f"writing to {fname}")
        with open(fname, "w") as f:
            json.dump(test_stats, f)

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
