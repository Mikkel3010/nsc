import numpy as np

test_point = 1 + 1j*0.5

# @profile
def mandelbrot_point(x, y, max_iter):
    z = 0j
    c = x + 1j*y

    for n in range(max_iter):
        if z.real*z.real + z.imag*z.imag > 4:
            return n
        z = z*z + c

    return max_iter

# @profile
def compute_mandelbrot_grid_naive(x_min: np.float32, x_max: np.float32, y_min: np.float32, y_max: np.float32, dimsize_x: int, dimsize_y: int, max_iter: int) -> np.array:
    """Computes a grid a of mandelbrot point and outputs the results as a numpy array of the iteration count.

    Args:
        x_min (np.float32): Value of the min range of x-axis
        x_max (np.float32): Value of the max range of x-axis
        y_min (np.float32): Value of the min range of y-axis
        y_max (np.float32): Value of the max range of y-axis
        dimsize_x (int): Pixel count on x-axis
        dimsize_y (int): Pixel count on y-axis
        max_iter (int): max allowed iteration on each pixel

    Returns:
        np.array: list of pixel values. 
    """
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

params = (-2, 1, -1.5, 1.5, 1024, 1024, 100)


results = compute_mandelbrot_grid_naive(*params)

#compute_mandelbrot_grid(-2, 1, -1.5, 1.5, 8192, 8192, max_iter=100)

#print(f" Computation took {elapsed:.3f} seconds ")
