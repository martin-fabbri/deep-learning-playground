import numpy as np


m = np.array([[1, 2, 3], [4, 5, 6]])
print(m)

n = m * 0.25
print(n)

print(np.multiply(m, n))

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(a.shape)

b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
print(b.shape)

c = np.matmul(a, b)
print(c)
print(c.shape)

print(b)
print(b.T)

