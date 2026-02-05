
import time
import numpy

test_point = 1 + 1j*0.5


def mandelbrot_point(x, y):
    z = 0
    c = x + 1j*y
    iter_count = 0
    while abs(z) < 2:
        z = z**2 + c
        iter_count += 1
    return iter_count


x, y = 1, 0.5

start = time.time()
iter_count = mandelbrot_point(x, y)

end = time.time()

period = end-start

print("time taken", period)
print("max iter", iter_count)
