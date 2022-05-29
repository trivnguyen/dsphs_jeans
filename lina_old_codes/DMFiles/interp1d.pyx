###############################################################################
# interp1d.pyx
###############################################################################
#
# 1d Linear interpolation, adapted from http://stackoverflow.com/a/13494757
#
###############################################################################


import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log10, exp

DTYPEf = np.float64
ctypedef np.float_t DTYPEf_t

cdef extern from "math.h":
    double pow(double x, double y) nogil

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
def interp1d(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=1] y,
               np.ndarray[DTYPEf_t, ndim=1] new_x):
    """
    interp1d(x, y, new_x)

    Performs linear interpolation in 1d.

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 1-D ndarray (double type)
        Array containing the y values to interpolate. Must have same length as x.
    x_new: 1-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 1-D ndarray
        Interpolated values.
    """
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    cdef int nx_new = new_x.shape[0]

    assert (nx == ny), \
        "x and y must have equal length!"
    cdef Py_ssize_t i, j
    cdef np.ndarray[DTYPEf_t, ndim=1] new_y = np.zeros(nx_new, dtype=np.float64)

    for i in range(nx_new):
        if x[0] >= new_x[i]:
            new_y[i] = y[0]
            print("Certain values in new_x too small!", i, new_x[i])
        elif x[nx-1] <= new_x[i]:
            new_y[i] = y[nx-1]
            print("Certain values in new_x too large!", i, new_x[i])
        else:
            for j in range(1, nx):
                if x[j] > new_x[i]:
                 new_y[i] = (y[j] - y[j-1]) / (x[j] - x[j-1]) * (new_x[i] - x[j-1]) + y[j-1]
                 break
    return new_y

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
def loginterp1d(double[:] x, double[:] y, double[:] new_x, skipzero = True):
    """
    interp1d(x, y, new_x)

    Performs log-linear interpolation in 1d.

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 1-D ndarray (double type)
        Array containing the y values to interpolate. Must have same length as x.
    x_new: 1-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 1-D ndarray
        Interpolated values.
    """
    cdef int nx = len(x)
    cdef int ny = len(y)
    cdef int nx_new = len(new_x)

    assert (nx == ny), \
        "x and y must have equal length!"
    cdef Py_ssize_t i, j
    cdef double[:] new_y = np.empty(nx_new, dtype=np.float)

    for i in range(nx_new):
        if skipzero and new_x[i] == 0:
            new_y[i] = 0
        elif x[0] >= new_x[i]:
            new_y[i] = y[0]
            # print("Certain values in new_x too small!")
        elif x[nx-1] <= new_x[i]:
            new_y[i] = y[nx-1]
            # print("Certain values in new_x too large!")
        else:
            for j in range(1, nx):
                if x[j] == new_x[i]:
                    new_y[i] = y[j]
                    break
                elif x[j] > new_x[i]:
                    new_y[i] = (y[j] - y[j-1]) / (log10(x[j]) - log10(x[j-1])) * (log10(new_x[i]) - log10(x[j-1])) + y[j-1]
                    break
    return new_y

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
def logloginterp1d(double[:] x, double[:] y, double[:] new_x, double offset = 0, skipzero = True):
    """
    interp1d(x, y, new_x)

    Performs log-log interpolation in 1d.

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 1-D ndarray (double type)
        Array containing the y values to interpolate. Must have same length as x.
    x_new: 1-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 1-D ndarray
        Interpolated values.
    """
    cdef int nx = len(x)
    cdef int ny = len(y)
    cdef int nx_new = len(new_x)

    assert (nx == ny), \
        "x and y must have equal length!"

    cdef Py_ssize_t i, j
    cdef double[:] new_y = np.empty(nx_new, dtype=np.float)

    for i in range(nx_new):
        if skipzero and new_x[i] == 0:
            new_y[i] = 0
        elif x[0] >= new_x[i]:
            new_y[i] = y[0]
            # print("Certain values in new_x too small!")
        elif x[nx-1] <= new_x[i]:
            new_y[i] = y[nx-1]
            # print("Certain values in new_x too large!")
        else:
            for j in range(1, nx):
                if x[j] == new_x[i]:
                    new_y[i] = y[j]
                    break
                elif x[j] > new_x[i]:
                    new_y[i] = pow(10, (log10(y[j]+offset) - log10(y[j-1]+offset)) / (log10(x[j]) - log10(x[j-1])) * (log10(new_x[i]) - log10(x[j-1])) + log10(y[j-1]+offset)) - offset
                    break
    return new_y

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off bounds-checking for entire function
# def loginterp0d(double[:] x, double[:] y, double new_x, skipzero = False):

#     cdef int nx = len(x)
#     cdef int ny = len(y)
#     assert (nx == ny), \
#         "x and y must have equal length!"

#     cdef Py_ssize_t j
#     cdef float new_y
#     if skipzero and new_x == 0:
#         new_y = np.nan
#     elif x[0] >= new_x:
#         new_y = y[0]
#     elif x[nx-1] <= new_x:
#         new_y = y[nx-1]
#     else: 
#         for j in range(1, nx):
#             if x[j] > new_x:
#                 new_y = (y[j] - y[j-1]) / (np.log10(x[j]) - np.log10(x[j-1])) * (np.log10(new_x) - np.log10(x[j-1])) + y[j-1]
#                 break
#     return new_y
