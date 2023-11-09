import torch
import numpy as np

class TaskMet(object):
    def __init__(self) -> None:
        self.metric = None
        self.predictor = None
        self.loss = None

    def train_predictor(self):
        pass

    def train(self):
        pass

if __name__ == '__main__':
    # write some demo toy experiment for TaskMet
    pass