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
for filename in [ "metric_reward.csv", "omd_reward.csv", "mle_reward.csv"]:
    df = pd.read_csv("exp/"+filename)
    df = df[df.columns.drop(list(df.filter(regex='__MIN')))]
    df = df[df.columns.drop(list(df.filter(regex='__MAX')))]
    df = df.filter(like='eval/episode_reward', axis=1)
    
    df = df.values.T
    index = get_smooth(df[0], 100, 10)[0]
    df = np.array([get_smooth(value, 100, 10)[1] for value in df])
    df = df.T
    mean = df.mean(axis=1)
    std_err = sp.stats.sem(df, axis=1)
    # print(mean, std_err)

    # ax.plot(range(len(mean)),mean, label=filename.split(".")[0])
    # ax.fill_between(range(len(mean)), mean-std_err, mean+std_err, alpha=0.2)

    ax.plot(index,mean, label=filename.split(".")[0])
    ax.fill_between(index, mean-std_err, mean+std_err, alpha=0.2)


ax.set_xlabel("Environment Steps")
ax.set_ylabel("Episode Return")
ax.set_xticklabels([f"{int(i)}K" if i!=0 else "0" for i in ax.get_xticks()])
ax.set_yticks([0, 100, 200, 300, 400, 500])
ax.grid(visible=True)
plt.savefig("reward.pdf", transparent=True, bbox_inches="tight", pad_inches=0.0)
