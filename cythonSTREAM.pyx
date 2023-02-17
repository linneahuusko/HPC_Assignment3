"""
Cython function for calculating STREAM benchmark
"""
from timeit import default_timer as timer
import numpy as np
cimport numpy as np

def STREAM(double[:] a, double[:] b, double[:] c, double[:] times, float scalar = 2.0):
    """
    Cythonized version of the STREAM benchmark

    Computes the operations of the STREAM benchmark for the three
    provided arrays, and saves the execution time for each operation
    in 'times'.

    Parameters:
        a, b, c (arrays): The arrays for which the STREAM calculations should be performed
        times (array): An array for storing the execution time for each calculation
    """

    cdef unsigned int j, STREAM_ARRAY_SIZE

    STREAM_ARRAY_SIZE = len(a)

    # copy
    times[0] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j]
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        b[j] = scalar * c[j]
    times[1] = timer() - times[1]
    # sum
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j] + b[j]
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j] + scalar * c[j]
    times[3] = timer() - times[3]

    return times
