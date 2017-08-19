from gradient_descent import *

def test_compute_error_for_given_points():
    error = compute_error_for_given_points(3, 0.5, [[2, 3], [5, 9]])
    assert error == 6.625



