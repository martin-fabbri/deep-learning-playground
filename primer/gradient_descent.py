# import numpy as np
from numpy import *


def compute_error_for_given_points(b, m, points):
    """Calculates the error or loss function based for a set of points

    >>> compute_error_for_given_points(3, 0.5, [[2, 3], [5, 9]])
    6.625

    """
    total_error = 0
    n = len(points)
    # print(n)

    for i in range(0, n):
        x = points[i][0]
        y = points[i][1]
        total_error += (y - (x * m + b)) ** 2
    return total_error / float(n)


def step_gradient(current_b, current_m, points, learning_rate):
    # gradient descent
    # pass
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2/n) * (y - (current_m * x + current_b))
        b_gradient += -(2/n) * x * (y - (current_m * x + current_b))

    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)

    return new_b, new_m


def gradient_descent_runner(points, starting_b, starting_m,
                            learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return b, m


def run():
    print("running ...")
    points = genfromtxt("../data/primer/data.csv", delimiter=",")

    # hyperparameters
    learning_rate = 0.0001

    # y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    numIterations = 1000

    b, m = gradient_descent_runner(points, initial_b, initial_m,
                                   learning_rate, numIterations)

    print(f"Optimized -> y = {m}x + {b}")


if __name__ == "__main__":
    run()
    # print(compute_error_for_given_points(3, .5, [[2, 3], [5, 9]]))