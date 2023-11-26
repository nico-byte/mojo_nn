from memory import Pointer
from memory import memset_zero
from random import randn, rand
from algorithm import vectorize

alias type = DType.float64
alias nelts = simdwidthof[type]()

struct Matrix:
    """Simple 2D Matrix that uses Float64."""
    # Expects when doing math with other Matrices that they all have the same dimensions
    var data: DTypePointer[type]
    var height: Int
    var width: Int

    # Initialize zeroeing all values
    fn __init__(inout self, height: Int, width: Int):
        self.data = DTypePointer[type].alloc(height * width)
        memset_zero(self.data, height * width)
        self.height = height
        self.width = width

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, height: Int, width: Int, data: DTypePointer[DType.float64]):
        self.data = DTypePointer[type].alloc(height * width)
        self.height = height
        self.width = width

    fn __init__(inout self, owned default_value: Float64, height: Int, width: Int) -> None:
        self.height = height if height > 0 else 1
        self.width = width if width > 0 else 1
        self.data = DTypePointer[type].alloc(height * width)
        for i in range(height * width):
            self.data.store(i, default_value)
    
    # fn __init__[*Ts: ListLiteral[Float32]](inout self, owned given_list: ListLiteral[Ts]) -> None:
    #     self.height = given_list.__len__()
    #     self.width = given_list.get[0, ListLiteral[Float32]]().__len__()
    #     self.total_items = self.height * self.width
    #     self.data = Pointer[Float32].alloc(self.total_items)
    #     let lists = Pointer.address_of(given_list).bitcast[ListLiteral[Float32]]()
    #     var current_loc = 0
    #     for i in range(self.height):
    #         var current_list: ListLiteral[Float32] = lists.load(i)
    #         let list_elems = Pointer.address_of(current_list).bitcast[Float32]()
    #         for j in range(self.width):
    #             self.data.store(current_loc, list_elems.load(j))
    #             current_loc += 1

    @staticmethod
    fn randn(height: Int, width: Int) -> Self:
        let data = DTypePointer[type].alloc(height * width)
        randn(data, height * width)
        return Self(height, width, data)

    @staticmethod
    fn rand(height: Int, width: Int) -> Self:
        let data = DTypePointer[type].alloc(height * width)
        rand(data, height * width)
        return Self(height, width, data)

    
    fn __getitem__(borrowed self, row: Int, column: Int) -> Float64:
        let loc: Int = (row * self.width) + column
        if loc > self.height * self.width:
            print("Warning: you're trying to get an index out of bounds, memory violation")
            return self.data.load(0)
        return self.data.load(loc)

    fn __setitem__(inout self, row: Int, column: Int, item: Float64) -> None:
        let loc: Int = (row * self.width) + column
        if loc > self.height * self.width:
            print("Warning: you're trying to set an index out of bounds, doing nothing")
            return
        self.data.store(loc, item)

    fn __del__(owned self) -> None:
        self.data.free()

    fn __len__(borrowed self) -> Int:
        return self.height * self.width

    fn __copyinit__(inout self, other: Self) -> None:
        self.height = other.height
        self.width = other.width
        self.data = Pointer[Float64].alloc(other.height * other.width)
        memcpy[type](self.data, other.data, other.height * other.width)

    fn __moveinit__(inout self, owned other: Self) -> None:
        self.height = other.height
        self.width = other.width
        self.data = Pointer[Float64].alloc(other.height * other.width)
        memcpy[type](self.data, other.data, other.height * other.width)

    fn __lt__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.height):
            for j in range(self.width):
                if self[i, j] < rhs[i, j]:
                    return True
        return False

    fn __gt__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.height):
            for j in range(self.width):
                if self[i, j] > rhs[i, j]:
                    return True
        return False

    fn __eq__(borrowed self, rhs: Matrix) -> Bool:
        for i in range(self.height):
            for j in range(self.width):
                let self_val: Float64 = self[i, j]
                let rhs_val: Float64 = rhs[i, j]
                if self_val < rhs_val or self_val > rhs_val:
                    return False
        return True

    fn __ne__(borrowed self, rhs: Matrix) -> Bool:
        return not self == rhs

    fn __ge__(borrowed self, rhs: Matrix) -> Bool:
        return self > rhs or self == rhs

    fn __le__(borrowed self, rhs: Matrix) -> Bool:
        return self < rhs or self == rhs

    fn __add__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] + rhs[i, j]
        return new_matrix

    fn __pow__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] ** rhs[i, j]
        return new_matrix

    fn __sub__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] - rhs[i, j]
        return new_matrix

    fn __mul__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] * rhs[i, j]
        return new_matrix

    fn __truediv__(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] / rhs[i, j]
        return new_matrix

    fn __add__(borrowed self, rhs: Float64) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] + rhs
        return new_matrix

    fn __pow__(borrowed self, rhs: Float64) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] ** rhs
        return new_matrix

    fn __sub__(borrowed self, rhs: Float64) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] - rhs
        return new_matrix

    fn __mul__(borrowed self, rhs: Float64) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] * rhs
        return new_matrix

    fn __truediv__(borrowed self, rhs: Float64) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = self[i, j] / rhs
        return new_matrix

    fn pow(borrowed self, rhs: Matrix) -> Matrix:
        var new_matrix: Matrix = Matrix(self.height, self.width)
        for i in range(self.height):
            for j in range(self.width):
                new_matrix[i, j] = rhs[i, j] ** self[i, j]
        return new_matrix
    
    fn print_all(borrowed self) -> None:
        print("[")
        for i in range(self.height):
            print_no_newline("    [")
            for j in range(self.width):
                print_no_newline(self[i, j])
                if j != self.width - 1:
                    print_no_newline(", ")
            print("]," if i != self.height - 1 else "]")
        print("]")
    
    fn __matmul__(borrowed self, rhs: Matrix) -> Matrix:
        var C: Matrix = Matrix(self.height, rhs.width)
        if self.width != rhs.height:
            print("Mat Mul not possible -> A.width: " + String(self.width) + " != B.height: " + String(rhs.height))
        for i in range(self.height):
            for j in range(rhs.width):
                for k in range(self.width):
                    C[i, j] += rhs[i, k] * rhs[k, j]
        return C
    
    fn transpose(borrowed self) -> Matrix:
        var new_matrix: Matrix = Matrix(self.width, self.height)
        for i in range(new_matrix.height):
            for j in range(new_matrix.width):
                new_matrix[i, j] += self[j, i]
        return new_matrix

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float64, nelts]):
        return self.data.simd_store[nelts](y * self.width + x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float64, nelts]:
        return self.data.simd_load[nelts](y * self.width + x)

    @staticmethod
    fn matmul_vectorized(C: Matrix, A: Matrix, B: Matrix):
        if A.width != B.height:
            print("Mat Mul not possible -> A.width: " + String(A.width) + " != B.height: " + String(B.height))
            
        if C.height != A.height or C.width != B.width:
            print("Mat Mul not possible -> A.height: " + String(A.height) + ", A.width: " + String(A.width) + " and B.height: " + String(B.height), ", B.width: " + String(B.width) + " don't match C.height: " + String(C.height) + ", C.width: " + String(C.width))
        for m in range(C.height):
            for k in range(A.width):
                @parameter
                fn dot[nelts : Int](n : Int):
                    C.store[nelts](m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n))
                vectorize[nelts, dot](C.width)

    @staticmethod
    fn matmul_vectorized_scal_neg(C: Matrix, A: Matrix, B: Float64):
        for m in range(C.height):
            for k in range(A.width):
                @parameter
                fn dot[nelts : Int](n : Int):
                    C.store[nelts](m, n, C.load[nelts](m, n) - A[m, k] * B)
                vectorize[nelts, dot](C.width)

    @staticmethod
    fn matmul_vectorized_neg(C: Matrix, A: Matrix, B: Matrix):
        if A.height != B.height != C.height and A.width != B.width != C.width:
            print("Negative Mat Mul not possible -> A.height: " + String(A.height) + ", A.width: " + String(A.width) + " and B.height: " + String(B.height), ", B.width: " + String(B.width) + " don't match C.height: " + String(C.height) + ", C.width: " + String(C.width))
        for m in range(C.height):
            for k in range(A.width):
                @parameter
                fn dot[nelts : Int](n : Int):
                    C.store[nelts](m, n, C.load[nelts](m, n) - A[m, k] * B.load[nelts](k, n))
                vectorize[nelts, dot](C.width)

    

        
