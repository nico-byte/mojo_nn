from types import Matrix

fn main() -> None:
    var test_matrix: Matrix = Matrix(Float64(3.14), 2, 2)
    test_matrix.print_all()
    let second_test_matrix: Matrix = Matrix(Float64(3.14), 2, 2)
    debug_assert(not test_matrix < second_test_matrix, "Should not be less")
    debug_assert(not test_matrix > second_test_matrix, "Should not be more")
    debug_assert(test_matrix == second_test_matrix, "Should be equal")
    debug_assert(not test_matrix != second_test_matrix, "Should not be not equal")
    debug_assert(test_matrix >= second_test_matrix, "Should be greater than or equal")
    debug_assert(test_matrix <= second_test_matrix, "Should be less than or equal")
    test_matrix[0, 0] = Float64(1.5)
    test_matrix[0, 1] = Float64(.33)
    test_matrix[1, 0] = Float64(6.5)
    test_matrix[1, 1] = Float64(2.5)
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
    let seventh_test_matrix: Matrix = test_matrix + Float64(1.0)
    let eighth_test_matrix: Matrix = test_matrix - Float64(1.0)
    let ninth_test_matrix: Matrix = test_matrix * Float64(1.0)
    let tenth_test_matrix: Matrix = test_matrix / Float64(1.0)
    @noncapturing
    fn test_func(x: Float64) -> Float64:
        return x * Float64(2.0)
    # let eleventh_test_matrix: Matrix = test_matrix.apply_function[test_func]()
    # let second_test: Matrix[] = [[Float32(1.5), Float32(.33)], [Float32(6.5), Float32(2.5)]]

    # testing matmul
    var mul_mutrixC_ref: Matrix = Matrix(Float64(500.0), 5, 5)
    var mul_mutrixC: Matrix = Matrix(Float64(0), 5, 5)
    var mul_mutrixA: Matrix = Matrix(Float64(5), 5, 10)
    var mul_mutrixB: Matrix = Matrix(Float64(10), 10, 5)
    Matrix.matmul_vectorized(mul_mutrixC, mul_mutrixA, mul_mutrixB)
    mul_mutrixC.print_all()
    mul_mutrixC_ref.print_all()
    debug_assert(mul_mutrixC != mul_mutrixC_ref, "Should be equal")

    let mul_mutrixC_neg_ref = Matrix(Float64(-500), 5, 5)
    var mul_mutrixC_neg = Matrix(Float64(0), 5, 5)
    let mul_mutrixA_neg = Matrix(Float64(5), 5, 10)
    let mul_mutrixB_neg = Matrix(Float64(10), 10, 5)
    Matrix.matmul_vectorized_neg(mul_mutrixC_neg, mul_mutrixA_neg, mul_mutrixB_neg)
    mul_mutrixC_neg.print_all()
    mul_mutrixC_neg_ref.print_all()
    debug_assert(mul_mutrixC_neg == mul_mutrixC_neg_ref, "Should be equal")

    let mul_mutrixC_scal_ref = Matrix(Float64(500), 5, 5)
    var mul_mutrixC_scal = Matrix(Float64(0), 5, 5)
    let mul_mutrixA_scal = Matrix(Float64(5), 5, 10)
    var B: Float64 = 10
    Matrix.matmul_vectorized_scal(mul_mutrixC_scal, mul_mutrixA_scal, B)
    mul_mutrixC_scal.print_all()
    mul_mutrixC_scal_ref.print_all()
    debug_assert(mul_mutrixC_scal != mul_mutrixC_scal_ref, "Should be equal")

    let mul_mutrixC_scal_neg_ref: Matrix = Matrix(Float64(-500), 5, 5)
    var mul_mutrixC_scal_neg: Matrix = Matrix(Float64(0), 5, 5)
    let mul_mutrixA_scal_neg: Matrix = Matrix(Float64(5), 5, 10)
    B = 10
    Matrix.matmul_vectorized_scal_neg(mul_mutrixC_scal_neg, mul_mutrixA_scal_neg, B)
    mul_mutrixC_scal_neg.print_all()
    mul_mutrixC_scal_neg_ref.print_all()
    debug_assert(mul_mutrixC_scal_neg != mul_mutrixC_scal_neg_ref, "Should be equal")

    
