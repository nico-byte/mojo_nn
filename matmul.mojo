from numjo import matmul_vectorized, matmul, Matrix
from random import rand
from time import now

fn matmul_benchmark() -> None:
    let M: Int = 512
    let K: Int = 512

    var C: Matrix = Matrix(K, M)
    C.zero()

    let A: Matrix = Matrix(M, K)
    rand(A.data, M * K)

    let B: Matrix = Matrix(K, M)
    rand(B.data, K * M)

    var start_time = now()
    matmul(C, A, B)
    var end_time = now() - start_time
    print("Naive Matrix Multiplication in mojo with 512x512 matrices took " + String(end_time / 1e6) + " milliseconds")

    C.zero()

    start_time = now()
    matmul_vectorized(C, A, B)
    end_time = now() - start_time
    print("Vectorized Matrix Multiplication in mojo with 512x512 matrices took " + String(end_time / 1e6) + " milliseconds")
