import time , statistics
from mandelbrot_naive import compute_mandelbrot_grid_naive
from mandelbrot_numpy import compute_mandelbrot_grid_numpy

def benchmark(func, *args, n_runs=3):
    """Time func, return median of n_runs."""
    times = []
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
t_naive = benchmark(compute_mandelbrot_grid_naive , * args )[0]
t_numpy = benchmark(compute_mandelbrot_grid_numpy , * args )[0]
# t_numba = benchmark() mandelbrot_naive_numba , * args )
print (f" Naive : { t_naive :.3f}s")
print (f" NumPy : { t_numpy :.3f}s ({ t_naive / t_numpy :.1f}x)")
# print (f" Numba : { t_numba :.3f}s ({ t_naive / t_numba :.1f}x)")


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
