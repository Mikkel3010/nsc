import time , statistics
from mandelbrot_naive import compute_mandelbrot_grid_naive
from mandelbrot_numpy import compute_mandelbrot_grid_numpy
from mandelbrot_naive_namba import compute_mandelbrot_grid_naive_namba
from mandelbrot_naive_namba_parralel import compute_mandelbrot_grid_naive_namba_parralel
import numpy as np
import matplotlib.pyplot as plt
def benchmark(func, *args, dtype=np.float64, n_runs=3):
    """Time func, return median of n_runs."""
    times = []
    args = tuple(list(args) + [dtype])
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    median_t = statistics.median(times)
    print(
        f"Median: {median_t:.4f}s "
        f"(min={min(times):.4f}, max={max(times):.4f})"
    )
    return median_t, result

# print("h")
# # naive
width , height = 1024 , 1024
args = (-2 ,1,-1.5, 1.5, width ,height, 100)

warmup_params = (-2 ,1,-1.5, 1.5, 64 ,64, 100)

t_naive = benchmark(compute_mandelbrot_grid_naive , *args )[0]
t_numpy = benchmark(compute_mandelbrot_grid_numpy , *args )[0]
# precompile
compute_mandelbrot_grid_naive_namba(*warmup_params, dtype=np.float64)
t_numba = benchmark(compute_mandelbrot_grid_naive_namba , *args)[0]

# numpy_namba_args = (X,Y, warmup_params[-1])
# t_numpy_namba = benchmark(compute_mandelbrot_grid_numpy_namba, *numpy_namba_args)[0]
print (f" Naive : { t_naive :.3f}s")
print (f" NumPy : { t_numpy :.3f}s ({ t_naive / t_numpy :.1f}x)")
print (f" Naive Numba : { t_numba :.3f}s ({ t_naive / t_numba :.1f}x)")
# print (f" NumPy Numba : { t_numba :.3f}s ({ t_naive / t_numpy_namba :.1f}x)")

# Lecture 3 - Milestone 4 - Data Type Optimization.

compute_mandelbrot_grid_naive_namba(*tuple(list(warmup_params) +[np.float64]))
t_numba64, picture64 = benchmark(compute_mandelbrot_grid_naive_namba, *args, dtype=np.float64)
compute_mandelbrot_grid_naive_namba(*tuple(list(warmup_params) +[np.float32]))

t_numba32, picture32 = benchmark(compute_mandelbrot_grid_naive_namba, *args, dtype=np.float32)
# compute_mandelbrot_grid_naive_namba(*tuple(list(warmup_params) +[np.float16]))
# t_numba16 = benchmark(compute_mandelbrot_grid_naive_namba, *args, dtype=np.float16)[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, result, title in zip(
    axes,
    [picture32, picture64],
    ["float32", "float64 (ref)"],
):
    ax.imshow(result, cmap="hot")
    ax.set_title(title)
    ax.axis("off")
plt.savefig("precision_comparison.png", dpi=150)

print (f" Naive Numba float64 : { t_numba64 :.3f}s")
print (f" Naive Numba float32 : { t_numba32 :.3f}s ({ t_numba64 / t_numba32 :.1f}x)")
# print (f" Naive Numba float16: { t_numba16 :.3f}s ({ t_numba / t_numba16 :.1f}x)")


# Milestone 4

# Experiment 2
print("Parallel Namba")
parralel, picture32 = benchmark(compute_mandelbrot_grid_naive_namba_parralel, *args, dtype=np.float32)

print (f" Naive Numba float32 : { parralel :.3f}s ({ t_numba64 / parralel :.1f}x)")

# Experiment 3

sizes = ((512,512), (1024,1024), (4098,4096))

grid_sizes = [] 
time_results=[]
time_results_naive =[]
speed_ups = []

for width, height in sizes:
    args = (-2 ,1,-1.5, 1.5, width ,height, 100)
    naive, picture32 = benchmark(compute_mandelbrot_grid_naive_namba, *args, dtype=np.float64)
    parralel, picture32 = benchmark(compute_mandelbrot_grid_naive_namba_parralel, *args, dtype=np.float64)
    gridsize = width**2
    grid_sizes.append(gridsize)
    time_results.append(parralel)
    time_results_naive.append(naive)
    speed_ups.append(parralel/naive)



plt.figure(figsize=(8, 5))

plt.plot(grid_sizes, time_results, marker="o", label="Parallel")
plt.plot(grid_sizes, time_results_naive, marker="s", label="Naive")
plt.plot(grid_sizes, speed_ups, marker=">", label="speedups")
plt.xlabel("Grid Size (width × height)")
plt.ylabel("Execution Time (seconds)")
plt.title("Mandelbrot Performance vs Grid Size")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
    




# runtime Median: 3.3566s (min=3.3511, max=3.3771)


# numpy vector (lecture 2)

# t , M = benchmark(mandelbrot_numpy.compute_mandelbrot_grid, -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)
# print(t)
# print(M)

# runtime  Median: 0.6365s (min=0.6208, max=0.6400)











# for lec2 milestone 4

# time_results = []
# for size in [256,512,1024,2048, 4096]:
#     print("size: ", size )
#     t , M = benchmark(mandelbrot_numpy.compute_mandelbrot_grid, -2, 1, -1.5 , 1.5 , size , size , 100)
#     print(t)
#     print(M)

#     time_results.append((size, t))
