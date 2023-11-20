import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable
from utils import dense_nn, View
import functorch
import torchopt
import random

class Metric(nn.Module):
    def __init__(
        self,
        num_features,
        num_output,
        num_hidden,
        identity_init,
        identity_init_scale,
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output * num_output),
        )
        self.identity_fac_log = torch.nn.parameter.Parameter(torch.zeros([]))
        if identity_init:
            last_layer = self.base[-1]
            last_layer.weight.data.div_(identity_init_scale)
            last_layer.bias.data = torch.eye(num_output).view(-1)

        self.num_output = num_output

    def forward(self, x):
        # A = torch.nn.functional.softplus(self.base(x))
        identity_fac = torch.exp(self.identity_fac_log)
        L = self.base(x)
        L = L.view(L.shape[0], self.num_output, self.num_output)
        A = (
            torch.bmm(L, L.transpose(1, 2))
            + identity_fac * torch.eye(self.num_output).repeat(x.shape[0], 1, 1).cuda()
        )
        # TODO: extend for PSD matrices with bounds from the
        # identity metric
        return A

class Task(object):
    def __init__(self, cfg):
        super().__init__()

    def decision(self, x):
        pass

    def objective(self, x, y):
        pass

class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = dense_nn()
   
    def forward(self, x):
        return self.model(x)
    
class TaskMet(object):
    def __init__(self, cfg, predictor, task) -> None:
        self.metric, self.metric_param = functorch.make_functional(Metric())
        self.predictor, self.pred_param = functorch.make_functional(predictor)
        self.task = task
        self.predictor_optimizer = torchopt.FuncOptimizer(
            torchopt.chain([torchopt.adam(lr=1e-3), 
                            torchopt.clip(self.predictor_grad_clip_norm)]))
        self.metric_optimizer = torch.optim.Adam(self.metric.parameters(), lr=1e-3)
        
    def pred_loss(self, pred_param, metric_param, x, y):
        if metric_param is None:
            metric_param = self.metric_param
        A = self.metric(metric_param, x)
        yhat = self.predictor(pred_param, x)
        err = (yhat - y).view(A.shape[0], A.shape[1], 1)
        # print(A.shape, err.shape)
        return (err.transpose(1, 2) @ A @ err).mean()
    
    def pred_optimality(self, pred_param, metric_param, x, y):
        losses = []
        num_samples = min(self.implicit_diff_batchsize, len(x))
        for i in random.sample(range(len(x)), num_samples):
            pred = self.predictor(pred_param, x[i]).squeeze()
            losses.append(
                self.pred_loss(
                    pred_param, x[i], y[i], metric_params=metric_param
                )
            )
        loss = torch.stack(losses).mean()
        return loss
    
    def train_predictor(self, pred_param, metric_param, x, y, batch_size, num_iters, **kwargs):
        # Fit the predictor to the data with the current metric value
        for train_iter in range(num_iters):
            losses = []
            num_samples = min(batch_size, len(x))
            for i in random.sample(range(len(x)), num_samples):
                losses.append(self.pred_loss(pred_param, metric_param, x[i], y[i]))
            loss = torch.stack(losses).mean()
            pred_param = self.predictor_optimizer.step(loss, pred_param)
                  
            g = torch.cat([p.flatten() for p in torch.autograd.grad(loss, pred_param)])
            if g.norm() < self.predictor_grad_norm_threshold:
                break

            if train_iter == 0 or train_iter % 10 == 0 and kwargs.get("verbose", False):
                print(
                    f"inner iter {train_iter} loss: {loss.item():.2e} grad norm: {g.norm():.2e}"
                )
       
        @torchopt.diff.implicit.custom_root(
            functorch.grad(self.pred_optimality, argnums=1),
            argnums=1,
            solve=torchopt.linear_solve.solve_normal_cg(
                    maxiter=self.implicit_diff_iters, atol=0.0, ridge=1e-5),
            )
        def solve(pred_param, metric_param):
            return pred_param

        pred_param = solve(pred_param, metric_param)
        
        return pred_param, loss.item()
    
    def train(self, x, y, batch_size, iters):
        for iter in iters:
            self.pred_param = self.train_predictor(self.pred_param, self.metric_param, x, y)

            losses=[]
            for i in random.sample(
                range(len(x)), min(self.cfg.batchsize, len(x))
            ):
                pred = self.predictor(self.pred_param, x).squeeze()

                Zs = self.task.decision(pred, isTrain=True, **kwargs)
                obj = self.task.get_objective(y, Zs, isTrain=True, **kwargs)
                loss = -obj
                losses.append(loss)

            loss = torch.stack(losses).mean()
            self.metric_optimizer.zero_grad()
            loss.backward()

            self.metric_optimizer.step()

if __name__ == '__main__':
    # write some demo toy experiment for TaskMet
    pass