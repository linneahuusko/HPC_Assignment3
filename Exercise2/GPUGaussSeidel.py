"""
Functions for solcing the Poisson equation using Gauss-Seidel iteration,
using the roll function and running pn GPU
"""
from torch import roll
import cupy as cp


def cupy_gauss_seidel(f):
    """
    Gauss-Seidel solver using the CuPy roll function, to be run on GPU

    Parameters:
        f (array): numpy array of size NxN

    Returns:
        newf (array): solution to the Poisson equation of f
    """
    newf = f.copy()
    # Boundary conditions
    newf[0, :] = 0
    newf[-1, :] = 0
    newf[:, 0] = 0
    newf[:, -1] = 0
    # Gauss-Seidel iteration
    newf = 0.25 * (
        cp.roll(newf, +1, 0)
        + cp.roll(newf, -1, 0)
        + cp.roll(newf, +1, 1)
        + cp.roll(newf, -1, 1)
    )

    return newf


def torch_gauss_seidel(f):
    """
    Gauss-Seidel solver using the PyTorch roll function, to be run on GPU

    Parameters:
        f (array): numpy array of size NxN

    Returns:
        newf (array): solution to the Poisson equation of f
    """
    newf = f.clone().cuda()
    # Boundary conditions
    newf[0, :] = 0
    newf[-1, :] = 0
    newf[:, 0] = 0
    newf[:, -1] = 0
    # Gauss-Seidel iteration
    newf = 0.25 * (
        roll(newf, +1, 0) + roll(newf, -1, 0) + roll(newf, +1, 1) + roll(newf, -1, 1)
    )

    return newf
