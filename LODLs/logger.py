import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Logger object to store the train and validation metrics to be used in workspcae object during training and also enable plotting at the of training
class Logger:
    def __init__(self, work_dir, filename):
        self.work_dir = work_dir
        self.filename = filename

        self.train_metrics = {}
        self.val_metrics = {}

    def log(self, train_metrics={}, val_metrics={}):
        # append the existing metrics and add new key if metric is not present
        for metric in train_metrics.keys():
            if metric not in self.train_metrics.keys():
                self.train_metrics[metric] = {}
            self.train_metrics[metric][train_metrics["iter"]] = train_metrics[metric]

        for metric in val_metrics.keys():
            if metric not in self.val_metrics.keys():
                self.val_metrics[metric] = {}
            self.val_metrics[metric][val_metrics["iter"]] = val_metrics[metric]

    def save(self):
        with open(os.path.join(self.work_dir, self.filename), "wb") as f:
            pkl.dump(self, f)

    # plot all the metrices and save them in work_dir
    def plot(self):
        # create individual plots for each metric
        for metric in self.train_metrics.keys():
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(
                list(self.train_metrics[metric].keys()),
                list(self.train_metrics[metric].values()),
                label=f"train_{metric}",
            )
            ax.plot(
                list(self.val_metrics[metric].keys()),
                list(self.val_metrics[metric].values()),
                label=f"val_{metric}",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.legend()
            fig.savefig(os.path.join(self.work_dir, f"{metric}.png"))
            ax.cla()

    @staticmethod
    def load(work_dir, filename):
        with open(os.path.join(work_dir, filename), "rb") as f:
            logger = pkl.load(f)
        return logger
