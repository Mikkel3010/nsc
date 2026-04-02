import math, random, time, statistics
import os
from multiprocessing import Pool

def estimate_pi_serial(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples


def estimate_pi_chunk(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def estimate_pi_parallel(num_samples, num_processes=4):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples


if __name__ == '__main__':
    num_samples = 10_000_000
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        pi_estimate = estimate_pi_serial(num_samples)
        times.append(time.perf_counter() - t0)

    t_serial = statistics.median(times)
    print(f"pi estimate: {pi_estimate:.6f}  (error: {abs(pi_estimate-math.pi):.6f})")
    print(f"Serial time: {t_serial:.3f}s")
    
    cores = os.cpu_count() # physical


    for core_count in range(1,cores+1):
        times=[]
        for _ in range(3):
            t0 = time.perf_counter()
            pi_estimate = estimate_pi_parallel(num_samples, num_processes=core_count)
            times.append(time.perf_counter() - t0)
        t_serial = statistics.median(times)
        print(f"Test with {core_count} Cores")
        print(f"pi estimate: {pi_estimate:.6f}  (error: {abs(pi_estimate-math.pi):.6f})")
        print(f"Serial time: {t_serial:.3f}s")



