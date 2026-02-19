
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_mandelbrot_grid(x_min, x_max, y_min, y_max, dimsize_x, dimsize_y, max_iter):
    x = np.linspace(x_min, x_max, dimsize_x)
    y = np.linspace(y_min, y_max, dimsize_y)
    X, Y = np.meshgrid(x,y)
    C = X + 1j*Y
    # print (f" Shape : {C. shape }") # (1024 , 1024)
    # print (f" Type : {C. dtype }") # complex128

    Z = np.zeros_like(C)
    M = np.zeros_like(C, dtype=np.int16)

    iter_count = 0
    for i in range(max_iter):
        mask = np.abs(Z) <=2
        # print(iter_count)
        iter_count += 1
    
        Z[mask] = Z[mask]**2 + C[mask]

        M[mask] += 1
        

    # #print(grid_colour_values)
    # # print(M)
    # plt.imshow(M, cmap='hot', vmin=0, vmax=max_iter)
    # plt.show()


# start = time.time()

# compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, max_iter=100)


# elapsed = time.time() - start

# print(f" Computation took {elapsed:.3f} seconds ")
