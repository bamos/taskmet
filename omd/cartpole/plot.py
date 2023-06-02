import os
import sys

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{mathtools}")
# plt.rc("font", size=20)

# x = np.array([16,32,64,128,256, 512])
x = np.array([4,5,6,7,8,9])

metric = np.array([497, 463, 387, 378, 252, 310])
metric_std = np.array([7, 110, 153, 163, 208, 200])/np.sqrt(10)
omd = np.array([402, 422, 340, 220, 172, 80])
omd_top = np.array([457, 482, 389, 267, 208.5, 104])
omd_std = omd_top - omd
mle = np.array([437, 250, 205, 38, 51, 37])
mle_top = np.array([488, 304, 260, 52, 60, 45])
mle_std = mle_top - mle

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.errorbar(x, metric, yerr=metric_std, label="TaskMet", color="C0", capsize=3, fmt="-o")
ax.errorbar(x, omd, yerr=omd_std, label="OMD", color="C1", capsize=3, fmt="-o")
ax.errorbar(x, mle, yerr=mle_std, label="MLE", color="C2", capsize=3, fmt="-o")

# x = np.array([16,32,64,128,256, 512])
# plt.xticks(x)
ax.set_xticks(x)
ax.set_xticklabels([ 2**i for i in x])
# plt.grid()
ax.grid(visible=True)
# plt.legend()
fig.tight_layout()
ax.set_xlabel("\# Distractors")
ax.set_ylabel("Episode Return")
fig.savefig("distractors.pdf", transparent=True, bbox_inches="tight", pad_inches=0.0)
plt.show()




