"""
Three approaches to solving
the Poisson equation using Gauss-Seidel iteration
"""
import numpy as np


def gauss_seidel(f):
    """
    Gauss-Seidel solver using a for-loop

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
    for i in range(1, newf.shape[0] - 1):
        for j in range(1, newf.shape[1] - 1):
            newf[i, j] = 0.25 * (
                newf[i, j + 1] + newf[i, j - 1] + newf[i + 1, j] + newf[i - 1, j]
            )
    return newf


def gauss_seidel_np(f):
    """
    Gauss-Seidel solver using NumPy vector operations

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
    newf[1:-1, 1:-1] = 0.25 * (
        newf[1:-1, 2:] + newf[1:-1, :-2] + newf[2:, 1:-1] + newf[:-2, 1:-1]
    )

    return newf


def gauss_seidel_roll(f):
    """
    Gauss-Seidel solver using the NumPy roll function

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
        np.roll(newf, +1, 0)
        + np.roll(newf, -1, 0)
        + np.roll(newf, +1, 1)
        + np.roll(newf, -1, 1)
    )

    return newf
