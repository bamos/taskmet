import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp

plt.style.use("bmh")
# plt.rc("text", u`setex=True)
# plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
# plt.rc("font", size=20)

def get_smooth(value, N_downsample, n_convolve):
    value = np.convolve(value, np.ones(n_convolve)/n_convolve, mode="valid")
    it = np.arange(len(value))
    # _it = np.linspace(it.min(), it.max(), num=N_downsample)
    # _v = sp.interpolate.interp1d(it, value)(_it)
    # return _it+n_convolve, _v
    return it+n_convolve-1, value

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
for filename in ["metric.csv","omd.csv", "mle.csv"]:
    df = pd.read_csv("exp/"+filename)
    df = df[df.columns.drop(list(df.filter(regex='__MIN')))]
    df = df[df.columns.drop(list(df.filter(regex='__MAX')))]
    if filename == "mle.csv":
        df = df.filter(like='train/loss_T', axis=1)
    else:
        df = df.filter(like='train/next_obs_nll', axis=1)
    # import pdb; pdb.set_trace()
    
    df = df.values.T
    index = get_smooth(df[0], 100, 10)[0]
    df = np.array([get_smooth(value, 100, 10)[1] for value in df])
    df = df.T
    mean = df.mean(axis=1)
    std_err = sp.stats.sem(df, axis=1)
    # print(mean, std_err)

    ax.plot(index,mean, label=filename.split(".")[0])
    ax.fill_between(index, mean-std_err, mean+std_err, alpha=0.2)


ax.set_yscale("log")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Dynamics (MSE)")
ax.set_xticklabels([f"{int(i)}K" if i!=0 else "0" for i in ax.get_xticks()])
plt.savefig("mse_hidden.pdf", transparent=True, bbox_inches="tight", pad_inches=0.0)
