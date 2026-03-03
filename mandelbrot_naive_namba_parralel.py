
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, jit, prange

# test_point = 1 + 1j*0.5

@jit
def mandelbrot_point(x, y, max_iter):
    z = 0j
    c = x + 1j*y
    for n in range( max_iter ):
        if z.real*z.real + z.imag*z.imag > 4:
            return n
        z = z*z + c
    return max_iter

@jit(parallel=True)
def compute_mandelbrot_grid_naive_namba_parralel(x_min, x_max, y_min, y_max, dimsize_x, dimsize_y, max_iter, dtype):

    X = np.linspace(x_min, x_max, num=dimsize_x).astype(dtype)
    Y = np.linspace(y_min, y_max, num=dimsize_y).astype(dtype)
    
    result = np.zeros((dimsize_x,dimsize_y), dtype = np.int32) 
    for i in prange(dimsize_x):
        for j in range (dimsize_y):
            result[i, j ] = mandelbrot_point(X[i],Y[j] , max_iter )


    #print(result)

    #plt.imshow(result, cmap='hot', vmin=0, vmax=100)
    #plt.show()
    return result


#compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 8192, 8192, max_iter=100)

#print(f" Computation took {elapsed:.3f} seconds ")
