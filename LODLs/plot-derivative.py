#!/usr/bin/env python3

"""
Reading files from multiple set of experiments of directory and plot their output
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle as pkl

exp_dir = sys.argv[1]

exp_dirs = os.listdir(exp_dir)

DQ = []
metric = []
for exp in exp_dirs:
    exp_path = os.path.join(exp_dir, exp)
    if os.path.isdir(exp_path) and "latest.pkl" in os.listdir(exp_path):
        print(f"Reading from {exp_path}")
        with open(os.path.join(exp_path, "latest.pkl"), "rb") as f:
            exp = pkl.load(f)
        X_test, Y_test, Y_test_aux = exp.problem.get_test_data()
        X_test = X_test.to("cuda")
        Y_test = Y_test.to("cuda")
        # Y_test_aux = Y_test_aux.to("cuda")

        pred = exp.model(X_test).squeeze()
        Zs_pred = exp.problem.get_decision(pred, aux_data=Y_test_aux, isTrain=False)
        objectives = exp.problem.get_objective(Y_test, Zs_pred, aux_data=Y_test_aux)
        objective = objectives.mean().item()
        DQ.append(objective)
        metric.append(exp.cfg.metricscale)
        y_pred = exp.model(X_test[0].to("cuda")).cpu().detach().numpy()


plt.scatter(metric, DQ, label="DQ vs Metric")
plt.legend()
plt.savefig("plot.png")
plt.clf()
