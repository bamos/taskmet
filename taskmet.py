import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable
from utils import dense_nn, View
import functorch
import torchopt
import random
from functools import partial
import ipdb


class Predictor(nn.Module):
    def __init__(
        self, num_features, num_targets, num_layers=2, intermediate_size=10, **kwargs
    ):
        super().__init__()
        self.model = dense_nn(
            num_features,
            num_targets,
            num_layers,
            intermediate_size,
            **kwargs,
        )

    def forward(self, x):
        return self.model(x)


class CustomOptimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.opt_state = None

    def step(self, loss, params):
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(params)
        grads = torch.autograd.grad(loss, params)

        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        params = torchopt.apply_updates(params, updates)
        return params, grads


class TaskMet(object):
    def __init__(self, cfg, predictor, task, metric=None) -> None:
        self.metric_def, self.metric_param = functorch.make_functional(metric)
        self.predictor_def, self.pred_param = functorch.make_functional(predictor)
        self.task = task

        self.predictor_optimizer = CustomOptimizer(
            torchopt.chain(
                torchopt.adam(lr=cfg.inner.inner_lr),
                torchopt.clip_grad_norm(
                    cfg.inner.predictor_grad_clip_norm, error_if_nonfinite=True
                ),
            )
        )

        self.metric_optimizer = CustomOptimizer(
            torchopt.adam(lr=cfg.metric_lr),
        )

    def pred_loss(self, pred_param, metric_param, x, y):
        if metric_param is None:
            metric_param = self.metric_param
        A = self.metric_def(metric_param, x)
        yhat = self.predictor_def(pred_param, x).squeeze()
        err = (yhat - y).view(A.shape[0], A.shape[1], 1)
        # print(A.shape, err.shape)
        loss = (err.transpose(1, 2) @ A @ err).mean()
        return loss

    def pred_optimality(self, pred_param, metric_param, x, y, **kwargs):
        losses = []
        num_samples = min(kwargs.get("implicit_diff_batchsize"), len(x))
        for i in random.sample(range(len(x)), num_samples):
            losses.append(self.pred_loss(pred_param, metric_param, x=x[i], y=y[i]))
        loss = torch.stack(losses).mean()
        return loss

    def make_pred_differentiable(self, x, y, **kwargs):
        def pred_optimality(pred_param, metric_param):
            losses = []
            num_samples = min(kwargs.get("implicit_diff_batchsize"), len(x))
            for i in random.sample(range(len(x)), num_samples):
                losses.append(self.pred_loss(pred_param, metric_param, x=x[i], y=y[i]))
            loss = torch.stack(losses).mean()
            return loss

        solver = torchopt.linear_solve.solve_normal_cg(
            maxiter=kwargs.get("implicit_diff_iters", 100), atol=0.0, ridge=1e-5
        )

        @torchopt.diff.implicit.custom_root(
            functorch.grad(pred_optimality, argnums=0),
            argnums=1,
            solve=solver,
        )
        def solve(pred_param, metric_param):
            return pred_param

        self.pred_param = solve(self.pred_param, self.metric_param)

        if torch.tensor([torch.isnan(param).any() for param in self.pred_param]).any():
            print("WARNING: NaN in new_params")
            print(self.pred_param)
            self.pred_param = self.pred_param

    def train_predictor(
        self,
        x,
        y,
        batch_size,
        num_iters,
        predictor_grad_norm_threshold=1e-3,
        **kwargs,
    ):
        """
        Inner level optimization
        Learns the optimal predictor parameters given the current metric parameters.

        Args:
            pred_param: current predictor parameters
            metric_param: current metric parameters
            x: input data
            y: output data
            batch_size: batch size for training
            num_iters: number of iterations for training
            **kwargs: additional arguments for training

        Returns:
            pred_param: updated predictor parameters
            loss: loss value
        """
        # Fit the predictor to the data with the current metric value
        for train_iter in range(num_iters):
            losses = []
            num_samples = min(batch_size, len(x))
            for i in random.sample(range(len(x)), num_samples):
                try:
                    losses.append(
                        self.pred_loss(self.pred_param, self.metric_param, x[i], y[i])
                    )
                except ValueError as e:
                    print(
                        f"Error in pred_loss at iteration {train_iter}, sample {i}, metric A: {self.metric(self.metric_param, x[i])}, err: {e}"
                    )
                    # ipdb.set_trace()
                    raise e

            # calculate the loss
            loss = torch.stack(losses).mean()

            # calculate the gradients and update the predictor parameters
            self.pred_param, grads = self.predictor_optimizer.step(
                loss, self.pred_param
            )

            # check if the gradient norm is below the threshold to stop the training
            g = torch.cat([p.flatten() for p in grads if p is not None])
            if g.norm() < predictor_grad_norm_threshold:
                break

            # print the loss and gradient norm
            if train_iter % 30 == 0 and kwargs.get("verbose", False):
                # ipdb.set_trace()
                print(
                    f"inner iter {train_iter} loss: {loss.item():.2e} grad norm: {g.norm():.2e}"
                )

        # make the predictor parameters implicit function of the metric parameters
        self.make_pred_differentiable(x, y, **kwargs)

        return {"inner_loss": loss.item()}

    def train(self, x, y, batch_size, iters, aux_data=None, **kwargs):
        """
        Outer level training loop for TaskMet.
        Learning the optimal metric and predictor parameters.
        """
        metrics = {}

        for iter in range(iters):
            inner_metrics = self.train_predictor(
                x,
                y,
                kwargs["inner_batchsize"],
                (
                    kwargs["inner_iters"]
                    if iter != 0
                    else kwargs.get("inner_iters_init", 500)
                ),
                **kwargs,
            )
            metrics.update(inner_metrics)

            losses = []
            DQ = []
            for i in random.sample(range(len(x)), min(batch_size, len(x))):
                pred = self.predictor_def(self.pred_param, x[i]).squeeze()
                loss = self.task.loss(
                    pred, y[i], aux_data=aux_data[i], isTrain=True, **kwargs
                )
                losses.append(loss)
                DQ.append(
                    self.task.objective(
                        y[i],
                        self.task.decision(pred, aux_data=aux_data[i], isTrain=False),
                        aux_data=aux_data[i],
                    ).item()
                )

            loss = torch.stack(losses).mean()
            DQ = np.mean(DQ)

            # ipdb.set_trace()
            metrics["outer_loss"] = loss.item()
            metrics["DQ"] = DQ

            print("metrics: ", metrics)

            self.metric_param, metric_grads = self.metric_optimizer.step(
                loss, self.metric_param
            )

            if iter % 10 == 0 and kwargs.get("verbose", False):
                print(f"outer iter {iter} loss: {loss.item():.2e}")

        return metrics


if __name__ == "__main__":
    # write some demo toy experiment for TaskMet
    pass
