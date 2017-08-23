import numpy as np
import time

a = np.array([1, 2, 3, 4, 5])
print(a)

l = np.random.rand(100000000)
m = np.random.rand(100000000)

tic = time.time()
q = np.dot(l, m)
toc = time.time()
print(f"Vectorized version took: {(toc-tic)*1000}ms")

w = 0
tic = time.time()
for i in range(100000000):
    w += l[i] * m[i]
toc = time.time()
print(f"Non-Vectorized version too: {(toc-tic)*1000}ms")


