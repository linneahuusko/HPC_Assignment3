"""
Script for running and plotting all different versions of Gauss-Seidel solvers
of the Poisson equation
"""

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from cythonGaussSeidel import (
    naive_cython_gauss_seidel,
    types_cython_gauss_seidel,
    types_cython_gauss_seidel_roll,
)
from pythonGaussSeidel import gauss_seidel, gauss_seidel_np, gauss_seidel_roll
from GPUGaussSeidel import torch_gauss_seidel, cupy_gauss_seidel

GPU_times = [
    [0.004567292000018597, 0.10005591799998115, 10.13182396900001, 1077.561042582],
    [
        0.045910001999999395,
        0.048963942000000316,
        0.08176970499999925,
        6.6680153939998945,
    ],
    [5.6200081120000505, 0.12964738400000897, 0.12450293500000953, 0.4039404929999364],
    [2.182354301000032, 0.44279291599991666, 0.461066843000026, 0.7351919529999122],
]
GPU_methods = [torch_gauss_seidel, cupy_gauss_seidel]
N = [3, 10, 100, 1000]
methods = [
    gauss_seidel,
    gauss_seidel_np,
    gauss_seidel_roll,
    naive_cython_gauss_seidel,
    types_cython_gauss_seidel,
    types_cython_gauss_seidel_roll,
]
times = [[] for i in methods]
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for n in N:
    print(n)
    for count, function in enumerate(methods):
        time0 = timer()
        x = np.random.rand(n, n)
        for i in range(1000):
            x = function(x)
        times[count].append(timer() - time0)
        print(function.__name__)

for count, function in enumerate(methods):
    axes[0].plot(N, times[count], "o", label=function.__name__)

N = [3, 10, 100, 1000]
colors = ["C6", "C7"]
for count, function in enumerate(GPU_methods):
    axes[1].plot(N, GPU_times[count], "o", label=function.__name__, color=colors[count])


for ax in axes:
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array size (N)")
axes[0].set_ylabel("Runtime [s]")
axes[0].set_title("CPU")
axes[1].set_title("GPU")
plt.subplots_adjust(wspace=0.05)
plt.savefig("All_versions_including_GPU_runtimes.png", bbox_inches="tight")
print(times)
