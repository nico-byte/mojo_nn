from numjo import Matrix
import numjo as nj
from math import sqrt
from memory import Pointer
from memory import memset_zero
from random import randn, rand, seed

fn main() -> None:
    var test_matrix: Matrix = Matrix(Float32(3.14), 2, 2)
    test_matrix.print_all()
    let second_test_matrix: Matrix = Matrix(Float32(3.14), 2, 2)
    debug_assert(not test_matrix < second_test_matrix, "Should not be less")
    debug_assert(not test_matrix > second_test_matrix, "Should not be more")
    debug_assert(test_matrix == second_test_matrix, "Should be equal")
    debug_assert(not test_matrix != second_test_matrix, "Should not be not equal")
    debug_assert(test_matrix >= second_test_matrix, "Should be greater than or equal")
    debug_assert(test_matrix <= second_test_matrix, "Should be less than or equal")
    test_matrix[0, 0] = Float32(1.5)
    test_matrix[0, 1] = Float32(.33)
    test_matrix[1, 0] = Float32(6.5)
    test_matrix[1, 1] = Float32(2.5)
    debug_assert(test_matrix < second_test_matrix, "Should be less")
    debug_assert(not test_matrix > second_test_matrix, "Should not be more")
    debug_assert(not test_matrix == second_test_matrix, "Should not be equal")
    debug_assert(test_matrix != second_test_matrix, "Should be not equal")
    debug_assert(not test_matrix >= second_test_matrix, "Should not be greater than or equal")
    debug_assert(test_matrix <= second_test_matrix, "Should be less than or equal")
    # No more debug_asserts because its been pretty well tested
    let third_test_matrix: Matrix = test_matrix + second_test_matrix
    let fourth_test_matrix: Matrix = test_matrix - second_test_matrix
    let fifth_test_matrix: Matrix = test_matrix * second_test_matrix
    let sixth_test_matrix: Matrix = test_matrix / second_test_matrix
    let seventh_test_matrix: Matrix = test_matrix + Float32(1.0)
    let eighth_test_matrix: Matrix = test_matrix - Float32(1.0)
    let ninth_test_matrix: Matrix = test_matrix * Float32(1.0)
    let tenth_test_matrix: Matrix = test_matrix / Float32(1.0)
    @noncapturing
    fn test_func(x: Float32) -> Float32:
        return x * Float32(2.0)
    # let eleventh_test_matrix: Matrix = test_matrix.apply_function[test_func]()
    # let second_test: Matrix[] = [[Float32(1.5), Float32(.33)], [Float32(6.5), Float32(2.5)]]

    var mul_matrix_ref: Matrix = Matrix(Float32(500), 5, 5)
    var mul_matrix_C: Matrix = Matrix(5, 5)
    var mul_matrix_A: Matrix = Matrix(Float32(5.0), 5, 10)
    var mul_matrix_B: Matrix = Matrix(Float32(10.0), 10, 5)

    nj.matmul_vectorized(mul_matrix_C, mul_matrix_A, mul_matrix_B)

    mul_matrix_ref.print_all()
    mul_matrix_C.print_all()

    mul_matrix_C = Matrix(5, 5)
    nj.matmul_vectorized(mul_matrix_C, mul_matrix_B.transpose(), mul_matrix_A.transpose())

    mul_matrix_ref.print_all()
    mul_matrix_C.print_all()

    mul_matrix_A = Matrix.randn(1, 784)
    # randn(mul_matrix_A.data, mul_matrix_A.rows * mul_matrix_A.cols)
    mul_matrix_B = Matrix.randn(784, 80)#   * sqrt(2/(100 + 80))
    # randn(mul_matrix_B.data, mul_matrix_B.rows * mul_matrix_B.cols)
    # mul_matrix_B = mul_matrix_B * sqrt(2.0/80))
    mul_matrix_C = Matrix(1, 80)

    nj.matmul(mul_matrix_C, mul_matrix_A, mul_matrix_B)
    mul_matrix_C.print_all()

    let A: Matrix = Matrix(1, 50)
    rand(A.data, A.rows * A.cols)
    A.print_all()

    let B: Matrix = Matrix(50, 1)
    rand(B.data, B.rows * B.cols)
    A.print_all()

    let C: Matrix = Matrix(A.rows, B.cols)
    nj.matmul(C, A, B)
    C.print_all()

    A[0, 35] = 0.99
    print(nj.argmax(A))

    print(nj.mean(A))
    print(nj.sum(A))

    

    