
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# @profile
def compute_mandelbrot_grid_numpy(x_min, x_max, y_min, y_max, dimsize_x, dimsize_y, max_iter, dtype=np.float64):
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
    

    #print(grid_colour_values)
    # print(M)
    plt.imshow(M, extent=[x_min,x_max,y_min,y_max], cmap='hot', vmin=0, vmax=max_iter, origin="lower")
    plt.show()

    return M

    


# start = time.time()


# params = (-2, 1, -1.5, 1.5, 1024, 1024, 100)


# PARAMS FOR MP3 L8 MP1

params = (-0.7530, -0.7490, 0.0990, 0.1030, 512, 512, 1000)


result = compute_mandelbrot_grid_numpy(*params)

# compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 1024, 1024, max_iter=100)


# elapsed = time.time() - start

# print(f" Computation took {elapsed:.3f} seconds ")
