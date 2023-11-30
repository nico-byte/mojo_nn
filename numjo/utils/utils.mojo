from numjo import Matrix
from algorithm import vectorize

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
    let mean: Float32 = sum(A) / (A.rows * A.cols)
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
        print("Mat Mul not possible -> A.cols: " + String(A.cols) + " != B.rows: " + String(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + String(A.rows) + ", A.cols: " + String(A.cols) + " and B.rows: " + String(B.rows), ", B.cols: " + String(B.cols) + " don't match C.rows: " + String(C.rows) + ", C.cols: " + String(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

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
        print("Mat Mul not possible -> A.cols: " + String(A.cols) + " != B.rows: " + String(B.rows))
        
    if C.rows != A.rows or C.cols != B.cols:
        print("Mat Mul not possible -> A.rows: " + String(A.rows) + ", A.cols: " + String(A.cols) + " and B.rows: " + String(B.rows), ", B.cols: " + String(B.cols) + " don't match C.rows: " + String(C.rows) + ", C.cols: " + String(C.cols))

    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(B.cols):
                C[m, n] += A[m, k] * B[k, n]
