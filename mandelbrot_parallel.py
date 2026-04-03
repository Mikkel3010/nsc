
import time
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from numba import njit, jit
from evaluations import benchmark
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import statistics


# test_point = 1 + 1j*0.5

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        zr2 = z_real*z_real
        zi2 = z_imag*z_imag
        if zr2 + zi2 > 4.0: return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = zr2 - zi2 + c_real
    return max_iter

@njit(cache=True)
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


def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, n_workers=4, n_chunks=None, pool=None,):
    if n_chunks is None:
        n_chunks = n_workers

    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0

    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    if pool is not None:  # caller manages Pool; skip startup + warm-up
        return np.vstack(pool.map(_worker, chunks))

    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny)  # warm-up: load JIT cache in workers
        parts = p.map(_worker, chunks)

    return np.vstack(parts)



if __name__ == "__main__":


    parameters_ = (1024, -2, 1, -1.5, 1.5, 100)
    # M1 test
    results_serial = benchmark(mandelbrot_serial, *parameters_, ignore_dtype=True)
    print (f" Chunk-L4-M1 : { results_serial[0]:.3f}s")



    # speedup_results = []

    # for i in range(1, os.cpu_count()+1):
    #     num_workers = i
    #     parameters = parameters_ + (num_workers,)
    #     # M2 test
    #     results_parallel = benchmark(mandelbrot_parallel, *parameters, ignore_dtype=True)
    #     print (f" POOL-L4-M1 : { results[0]:.3f}s, workers: {num_workers}")
    #     results = results_serial[0]/results_parallel[0]

    # plt.plot(speedup_results)
    # plt.xlabel("number of cores")
    # plt.ylabel("speed-up x")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.imshow(
    #     result,
    #     extent=[-2.5, 1.0, -1.25, 1.25],
    #     cmap="inferno",
    #     origin="lower",
    #     aspect="equal",
    # )
    # ax.set_xlabel("Re(c)")
    # ax.set_ylabel("Im(c)")

    # out = Path(__file__).parent / "mandelbrot.png"
    # fig.savefig(out, dpi=150)
    # print(f"Saved: {out}")
    # times = []
    
    # for _ in range(3):
    #     t0 = time.perf_counter()
    #     mandelbrot_serial(*parameters_)
    #     times.append(time.perf_counter() - t0)

    # t_serial = statistics.median(times)

    # N = parameters_[0]

    # for n_workers in range(1, os.cpu_count() + 1):
    #     chunk_size = max(1, N // n_workers)
    #     chunks, row = [], 0

    #     while row < N:
    #         end = min(row + chunk_size, N)
    #         chunks.append((row, end, N, *parameters_[1:]))
    #         row = end

    #     with Pool(processes=n_workers) as pool:
    #         # warm-up: Numba JIT in all workers
    #         pool.map(_worker, chunks)

    #         times = []
    #         for _ in range(3):
    #             t0 = time.perf_counter()
    #             np.vstack(pool.map(_worker, chunks))
    #             times.append(time.perf_counter() - t0)

    #     t_par = statistics.median(times)
    #     speedup = t_serial / t_par

    #     print(
    #         f"{n_workers:2d} workers: {t_par:.3f}s, "
    #         f"speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%"
    #     )