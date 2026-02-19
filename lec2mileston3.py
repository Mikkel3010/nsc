import numpy as np
import time
N = 10000

A = np.random.rand(N, N)


start = time.perf_counter()
#row
print("rowbased")
for i in range(N):
    s = np.sum(A[i,:])
time_result = time.perf_counter()- start
print(f" Computation took {time_result:.3f} seconds ")

# colum
print("columnbased")
start = time.perf_counter()
for i in range(N):
    s = np.sum(A[:,i])
time_result = time.perf_counter()- start
print(f" Computation took {time_result:.3f} seconds ")

# rowbased
#  Computation took 0.176 seconds 
# columnbased
#  Computation took 0.202 seconds 


# 5) fortranbased
start = time.perf_counter()
#
A = np.asfortranarray(A)
print("rowbased")
for i in range(N):
    s = np.sum(A[i,:])
time_result = time.perf_counter()- start
print(f" Computation took {time_result:.3f} seconds ")

# colum
print("columnbased")
start = time.perf_counter()
for i in range(N):
    s = np.sum(A[:,i])
time_result = time.perf_counter()- start
print(f" Computation took {time_result:.3f} seconds ")

# rowbased
#  Computation took 0.640 seconds 
# columnbased
#  Computation took 0.037 seconds 