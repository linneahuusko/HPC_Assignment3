"""
Script for solving the Poisson equation and saving the data to a hdf5 file
"""
import h5py
from pythonGaussSeidel import gauss_seidel_np
import numpy as np

x = np.random.rand(10, 10)
for i in range(1000):
    x = gauss_seidel_np(x)

f = h5py.File("GaussSeidel_output.hdf5", "w")

f.create_dataset("/gauss_seidel_np/x", data=x)
f.close()
