import numpy as np


def test_dtype(dtype):

    count = 0

    eps = dtype(1.0)
    
    while dtype(1.0) + eps != dtype(1.0):
        eps /= 2
    print(eps)

    print(np.finfo(dtype))
    


test_dtype(np.float16)
test_dtype(np.float32)
test_dtype(np.float64)