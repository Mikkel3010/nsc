
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
from dask import delayed
from dask.distributed import Client, LocalCluster
import dask, numpy as np, time, statistics


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


def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter, n_workers, n_chunks, pool=None,):
    if n_chunks is None:
        n_chunks = n_workers
    # print("Runnings parameters")
    # print(N, x_min, x_max, y_min, y_max, max_iter, n_workers, n_chunks, pool,)

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

def mandelbrot_dask(N, x_min, x_max, y_min, y_max,
    max_iter=100, n_chunks=32):
    chunk_size = max(1, N // n_chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)


if __name__ == "__main__":
    
    # # L6-M1 ---------------------------------------------------------------
    N, max_iter = 4096, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    client = Client("tcp://10.92.1.35:8786")


    serial_time = 0.991 #0.059 at 1024, 0.991s at 4096
    t_min = 100000
    n_chunk_optim = 9999
    LIF_min = (1,9999)

    client.run(lambda: mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, # warm up all workers
    Y_MIN, Y_MAX, 10))


    chunk_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    times_list = []
    for n_chunk_size in chunk_sizes:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = mandelbrot_dask(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter, n_chunks=n_chunk_size)
            times.append(time.perf_counter() - t0)
        median_t = statistics.median(times)
        lif = (6 * median_t / serial_time) - 1 # 8 from 8 workers
        times_list.append(median_t)   
        if median_t < t_min:
            n_chunk_optim=n_chunk_size
        if lif < LIF_min[1]:
            LIF_min = (n_chunk_size, lif,)
        print(f"{n_chunk_size} | {median_t} | {serial_time/median_t} | {lif}")

    client.close()

    fig, ax = plt.subplots()

    # time vs chunks
    ax.plot(chunk_sizes, times_list, marker="o")

    ax.set_xlabel("n_chunks")
    ax.set_ylabel("Time (s)")
    # best point
    best_idx = times_list.index(min(times_list))
    ax.plot(chunk_sizes[best_idx], times_list[best_idx], marker="*", markersize=15)
    ax.annotate(
        f"best: {chunk_sizes[best_idx]} ({times_list[best_idx]:.2f}s)",
        (chunk_sizes[best_idx], times_list[best_idx]),
        xytext=(5, 5),
        textcoords="offset points",
    )
    ax.set_xscale("log", base=2)

    # adding serial time as horizontal line:
    ax.hlines(serial_time, 0, 1024, label="Serial", colors="gray", linestyles="dashed")

    plt.title(f"Chunk sweep ({N}x{N}, 8 workers)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


    # L5-M2 and M3-------------------------------------------------------------------
    # N, max_iter = 4096, 100
    # n_workers = 8  # My optimum was 6
    # X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25

    # # warm up JIT
    # mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)

    # # Serial baseline
    # times = []
    # for _ in range(3):
    #     t0 = time.perf_counter()
    #     mandelbrot_chunk(0, N, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
    #     times.append(time.perf_counter() - t0)



    # t_serial = statistics.median(times)
    # print(f"Serial: {t_serial:.3f}s")

    # # Chunk-count sweep (M2): one Pool per config
    # tiny = [(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)]

    # for mult in [1, 2, 4, 8, 16]:
    #     n_chunks = mult * n_workers

    #     with Pool(processes=n_workers) as pool:
    #         # warm-up: load JIT cache in workers
    #         pool.map(_worker, tiny)

    #         times = []
    #         for _ in range(3):
    #             t0 = time.perf_counter()
    #             mandelbrot_parallel(
    #                 N,
    #                 X_MIN,
    #                 X_MAX,
    #                 Y_MIN,
    #                 Y_MAX,
    #                 max_iter,
    #                 n_workers=n_workers,
    #                 n_chunks=n_chunks,
    #                 pool=pool,
    #             )
    #             times.append(time.perf_counter() - t0)

    #     t_par = statistics.median(times)
    #     lif = n_workers * t_par / t_serial - 1

    #     print(f"{n_chunks:4d} chunks {t_par:.3f}s {t_serial/t_par:.1f}x LIF={lif:.2f}")

    #-------------------------------------------------

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