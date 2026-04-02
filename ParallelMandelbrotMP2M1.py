
import time
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, jit
from evaluations import benchmark

# test_point = 1 + 1j*0.5

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit
def mandelbrot_chunk(row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N

    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for col in range(N):
            out[r, col] = mandelbrot_pixel(x_min + col * dx, c_imag, max_iter)

    return out


def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
    print()
    return mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)



parameters = (1024, -2, 1, -1.5, 1.5, 100)

results = benchmark(mandelbrot_serial, *parameters, ignore_dtype=True)
print (f" Chunk-L4-M1 : { results[0]:.3f}s")