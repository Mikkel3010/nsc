import time , statistics
import mandelbrot

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

print("h")

t , M = benchmark(mandelbrot.compute_mandelbrot_grid, -2, 1, -1.5 , 1.5 , 1024 , 1024 , 100)
print(t)
print(M)