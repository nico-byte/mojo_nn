from matrix import Matrix
from algorithm.functional import vectorize
from sys import simdwidthof

alias type = DType.float32
alias nelts = simdwidthof[type]()

fn argmax(A: Matrix) -> Int:
    """Returns the index of the max value of the given row."""
    var max_idx: Int = 0
    var max_val: Float32 = A[0, 0]
    for i in range(A.rows):
        for j in range(A.cols):
            if A[i, j] > max_val:
                max_val = A[i, j]
                max_idx = j
    return max_idx

fn update(C: Matrix, A: Matrix, B: Float32):
    for m in range(C.rows):
        for k in range(C.cols):
            C[m, k] = C[m, k] - A[m, k] * B

fn mean(A: Matrix) -> Float32:
    var mean: Float32 = sum(A) / (A.rows * A.cols)
    return mean
    
fn sum(A: Matrix) -> Float32:
    var sum: Float32 = 0.0
    for i in range(A.rows):
        for j in range(A.cols):
            sum += A[i, j]
    return sum

fn sum_row(A: Matrix, row: Int) -> Float32:
    var sum: Float32 = 0.0
    for j in range(A.cols):
        sum += A[row, j]
    return sum

fn matmul_vectorized(C: Matrix, A: Matrix, B: Matrix):
    """
    Vectorized mat mul from the docs, 
    C: Output Matrix, 
    A: Input Matrix A, 
    B: Input Matrix B, 
    C += A @ B, 
    A, B and C have to meet the requirements for the matmul - A.cols == B.rows, C.rows == A.rows and C.cols == B.cols.
    """
    if A.cols != B.rows:
        print("Mat Mul not possible -> A.cols: " + str(A.cols) + " != B.rows: " + str(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + str(A.rows) + ", A.cols: " + str(A.cols) + " and B.rows: " + str(B.rows), ", B.cols: " + str(B.cols) + " don't match C.rows: " + str(C.rows) + ", C.cols: " + str(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts](C.cols)

fn matmul(C: Matrix, A: Matrix, B: Matrix):
    """
    Vectorized mat mul from the docs, 
    C: Output Matrix, 
    A: Input Matrix A, 
    B: Input Matrix B, 
    C += A @ B, 
    A, B and C have to meet the requirements for the matmul - A.cols == B.rows, C.rows == A.rows and C.cols == B.cols.
    """
    if A.cols != B.rows:
        print("Mat Mul not possible -> A.cols: " + str(A.cols) + " != B.rows: " + str(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + str(A.rows) + ", A.cols: " + str(A.cols) + " and B.rows: " + str(B.rows), ", B.cols: " + str(B.cols) + " don't match C.rows: " + str(C.rows) + ", C.cols: " + str(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(B.cols):
                C[m, n] += A[m, k] * B[k, n]
