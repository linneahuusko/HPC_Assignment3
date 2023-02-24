"""
Cythonised versions of three approaches to solving
the Poisson equation using Gauss-Seidel iteration
"""
import numpy as np
cimport numpy as np

def naive_cython_gauss_seidel(f):
    """
    Gauss-Seidel solver using a for-loop, compiled with Cython

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

def types_cython_gauss_seidel(double[:,:] f):
    """
    Gauss-Seidel solver using a for-loop, compiled with Cython including type definitions

    Parameters:
        f (array): numpy array of size NxN

    Returns:
        newf (array): solution to the Poisson equation of f
    """
    cdef double[:,:] newf
    cdef unsigned int i, j

    newf = f.copy()
    # Boundary conditions
    newf[0, :] = 0.0
    newf[-1, :] = 0.0
    newf[:, 0] = 0.0
    newf[:, -1] = 0.0
    # Gauss-Seidel iteration
    for i in range(1, newf.shape[0] - 1):
        for j in range(1, newf.shape[1] - 1):
            newf[i, j] = 0.25 * (
                newf[i, j + 1] + newf[i, j - 1] + newf[i + 1, j] + newf[i - 1, j]
            )

    return newf

def types_cython_gauss_seidel_roll(double[:,:] f):
    """
    Gauss-Seidel solver using the NumPy roll function,
    compiled with Cython including type definitions

    Parameters:
        f (array): numpy array of size NxN

    Returns:
        newf (array): solution to the Poisson equation of f
    """
    cdef double[:,:] newf

    newf = f.copy()
    # Boundary conditions
    newf[0, :] = 0.0
    newf[-1, :] = 0.0
    newf[:, 0] = 0.0
    newf[:, -1] = 0.0
    # Gauss-Seidel iteration
    newf = 0.25 * (np.roll(newf, +1, 0) + np.roll(newf, -1, 0) + np.roll(newf, +1, 1) + np.roll(newf, -1, 1))

    return newf
