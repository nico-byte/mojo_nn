from toolbox import matmul, matmul_vectorized
from matrix import Matrix
from random import rand
from time import now

fn matmul_benchmark() -> None:
    var M: Int = 512
    var K: Int = 512

    var C: Matrix = Matrix(K, M)
    C.zero()

    var A: Matrix = Matrix(M, K)
    rand(A.data, M * K)

    var B: Matrix = Matrix(K, M)
    rand(B.data, K * M)

    var start_time = now()
    matmul(C, A, B)
    var end_time = now() - start_time
    print("Naive Matrix Multiplication in mojo with 512x512 matrices took " + str(end_time / 1e6) + " milliseconds")

    C.zero()

    start_time = now()
    matmul_vectorized(C, A, B)
    end_time = now() - start_time
    print("Vectorized Matrix Multiplication in mojo with 512x512 matrices took " + str(end_time / 1e6) + " milliseconds")
