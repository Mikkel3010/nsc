
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

test_point = 1 + 1j*0.5


def mandelbrot_point(x, y, max_iter):
    z = 0
    c = x + 1j*y
    iter_count = 0
    while abs(z) < 2 and iter_count < max_iter:
        z = z**2 + c
        iter_count += 1
    return iter_count


def compute_mandelbrot_grid_naive(x_min, x_max, y_min, y_max, dimsize_x, dimsize_y, max_iter):
    point_list = []
    for x in np.linspace(x_min, x_max, num=dimsize_x):
        for y in np.linspace(y_min, y_max, num=dimsize_y):
            point_list.append((x, y))

    grid_colour_values = []
    for point in point_list:
        # print("processing")
        grid_colour_values.append(
            mandelbrot_point(point[0], point[1], max_iter=max_iter))

    grid_np_array = np.array(grid_colour_values)

    grid_np_array = grid_np_array.reshape(dimsize_x, dimsize_y)

    #print(grid_colour_values)

    #plt.imshow(grid_np_array, cmap='hot', vmin=0, vmax=100)
    #plt.show()
    return grid_np_array


#compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 8192, 8192, max_iter=100)

#print(f" Computation took {elapsed:.3f} seconds ")
