import torch
import torch.nn as nn

from abc import ABC, abstractmethod

import sys

sys.path.append("./third_party/")

from LODLs.BudgetAllocation import BudgetAllocation
from LODLs.PortfolioOpt import PortfolioOpt
from LODLs.CubicTopK import CubicTopK
from LODLs.utils import init_if_not_saved

tasks = {
    "advertising": BudgetAllocation,
    "portfolio": PortfolioOpt,
    "cubic": CubicTopK,
}


class Task(ABC):
    """
    Template class to define a downstream task.

    Task is defined by an objective function g(z,y)
    where z is the decision variable and y is parameter
    of the task which is considered being predicted by
    a learned model. objective should be differentiable
    w.r.t. y.

    The task is to find optimal decision variable z^*
    z^* = argmin_z g(z,y)
    """

    def __init__(self, cfg):
        super().__init__()

    @abstractmethod
    def decision(self, y, aux_data=None, **kwargs):
        """
        Given model prediction output, which parameterize the downstream task,
        outputs the decision variable.
        z^* = argmin_z g(z,y)

        -- HAVE TO ENSSURE THAT z^* IS DIFFERENTIABLE W.R.T. Y --

        Args:
            y: model prediction output

        Returns:
            z*: decision variable
        """
        pass

    @abstractmethod
    def objective(self, y, z, aux_data=None, **kwargs):
        """
        Given model prediction output and decision variable,
        outputs the objective value.
        obj = g(z, y)

        Args:
            y: model prediction output
            z: decision variable

        Returns:
            obj: objective value
        """
        pass

    @abstractmethod
    def get_modelio_shape(self):
        """
        Returns the input and output dimensions for the model.

        Returns:
            Tuple[int, int]: A tuple containing the input dimension and output dimension.
        """
        pass

    @abstractmethod
    def loss(self, pred, y, aux_data=None, **kwargs):
        """
        Value of objective function, under true y parameters
        and z find using predicted y parameters

        Args:
            pred: predicted y parameters
            y: true y parameters
            isTrain: whether in training mode

        Returns:
            loss: objective value
        """
        pass


class DFL(Task):
    """
    Task class for the DFL problem, to use code of
    [Shah et.al. arXiv preprint arXiv:2203.16067]

    The task would have a predictor associated with it
    which takes input x and outputs y. The task is to
    minimize the objective function g(z,y) where z is
    the decision variable and y is the parameter of the
    task which is considered being predicted by the model.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        print(f"Initializing task {cfg.problem} with kwargs {cfg.problem_kwargs}")
        self.cfg = cfg
        self.problem = init_if_not_saved(
            tasks[cfg.problem], dict(cfg.problem_kwargs), load_new=cfg.loadnew
        )

    def decision(self, y, aux_data=None, **kwargs):
        return self.problem.get_decision(y, aux_data=aux_data, **kwargs)

    def objective(self, y, z, aux_data=None, **kwargs):
        # In Shah et.al. the task is to maximize the objective
        return self.problem.get_objective(y, z, aux_data=aux_data, **kwargs)

    def _get_loss_function(self, name):
        if name == "mse":

            def loss_fn(pred, y, aux_data=None, **kwargs):
                loss = nn.MSELoss(pred, y)
                return loss

            return loss_fn
        elif name == "dfl":

            def loss_fn(pred, y, aux_data=None, **kwargs):
                z_star = self.decision(pred, aux_data=aux_data, **kwargs)
                obj = self.objective(y, z_star, aux_data=aux_data, **kwargs)
                loss = -obj

                # addin mse term of task_loss -> to replicate dfl training
                if self.cfg.taskmet_kwargs.dflalpha:
                    loss += self.cfg.taskmet_kwargs.dflalpha * (pred - y).pow(2).mean()
                return loss

            return loss_fn
        else:
            raise ValueError(f"Loss function {name} not supported.")

    def loss(self, pred, y, aux_data=None, **kwargs):
        loss_fn = self._get_loss_function(self.cfg.loss)
        return loss_fn(pred, y, aux_data=aux_data, **kwargs)

    def get_modelio_shape(self):
        return self.problem.get_modelio_shape()
