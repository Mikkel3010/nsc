import cProfile
import pstats

from mandelbrot_naive import compute_mandelbrot_grid_naive
from mandelbrot_numpy_namba import compute_mandelbrot_grid_numpy

def profile(func, *args, prof_file: str, top_n: int = 10, sort_by: str = "cumulative"):
    cProfile.runctx(
        "func(*args)",
        globals={"func": func, "args": args},
        locals={},
        filename=prof_file,
    )
    stats = pstats.Stats(prof_file)
    stats.sort_stats(sort_by)
    stats.print_stats(top_n)


if __name__ == "__main__":
    params = (-2, 1, -1.5, 1.5, 1024, 1024, 100)

    profile(compute_mandelbrot_grid_naive, *params, prof_file="naive_profile.prof")
    profile(compute_mandelbrot_grid_numpy, *params, prof_file="numpy_profile.prof")