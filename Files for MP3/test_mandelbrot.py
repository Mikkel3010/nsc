from mandelbrot_naive import compute_mandelbrot_grid_naive, mandelbrot_point
from mandelbrot_naive_namba import compute_mandelbrot_grid_naive_namba
import pytest
import pandas as pd
import numpy as np
KNOWN_CASES = [
    (0,0, 100, 100), 
    (5.0, 0, 100, 1), 
    (-2.5, 0, 100, 1),]

truth_output = results_naive = pd.read_csv("results_naive.csv").values # assuming for now that the output is correct for naive

GRID_CASES = [
    ((-2, 1, -1.5, 1.5, 32, 32, 100, np.float32), truth_output)
]

def test_mandelbrot_grid_versions():
    params = (-2, 1, -1.5, 1.5, 32, 32, 100)
    params_numba = (-2, 1, -1.5, 1.5, 32, 32, 100, np.float32)
    results_naive = compute_mandelbrot_grid_naive(*params)
    results_numba = compute_mandelbrot_grid_naive_namba(*params_numba)
    # pd.DataFrame(results_naive).to_csv("results_naive.csv", index=False) 
    assert np.array_equal(results_naive, results_numba)


@pytest.mark.parametrize("input, expected", GRID_CASES)
def test_output_numba_is_correct(input, expected):
    results_numba = compute_mandelbrot_grid_naive_namba(*input)

    assert np.array_equal(expected,results_numba)



@pytest.mark.parametrize("x, y, max_iter, expected", KNOWN_CASES)
def test_mandelbrot_point(x, y, max_iter, expected):
    print("testing mandelbrot_point")
    assert mandelbrot_point(x,y,max_iter) == expected

