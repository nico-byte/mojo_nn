import numpy as np
from time import time

M = 512
K = 512
A = np.random.randn(M , K)
B = np.random.randn(K, M)
start_time = time()
C = np.dot(A, B)
end_time = time()-start_time
print(f"Matrix Multiplication in Python with 512x512 matrices took {end_time * 1e3} milliseconds.")
