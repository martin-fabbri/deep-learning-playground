import numpy as np


A = np.array([
    [56.0, 0.0, 4.4, 68.0],
    [1.2, 104.0, 52.0, 8.0],
    [1.8, 135.0, 99.0, 0.9]
])
print(A)

cal = A.sum(axis=0)
print(cal)
print(cal.shape)
print(type(cal))

cal = np.reshape(cal, (1, 4))
print(cal.shape)

percentage = (A / cal) * 100
print(percentage)


