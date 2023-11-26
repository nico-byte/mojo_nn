from random import rand, randn
from math import sqrt
from memory.unsafe import DTypePointer
from types import Matrix
from time import now

struct Network:
    var _inodes: Int
    var _hnodes_l1: Int
    var _hnodes_l2: Int
    var _onodes: Int
    var lr: Float64
    var _wih: Matrix
    var _whh: Matrix
    var _who: Matrix
    var _bih_l1: Matrix
    var _bih_l2: Matrix
    var _bho: Matrix
    
    fn __init__(inout self, input_nodes: Int, hidden_nodes_l1: Int, hidden_nodes_l2: Int, output_nodes: Int, learning_rate: Float64):
        self._inodes = input_nodes
        self._hnodes_l1 = hidden_nodes_l1
        self._hnodes_l2 = hidden_nodes_l2
        self._onodes = output_nodes

        self.lr = learning_rate

        self._wih = Matrix.randn(self._inodes, self._hnodes_l1) * Matrix(Float64(sqrt(2/self._inodes)), self._inodes, self._hnodes_l1)
        self._whh = Matrix.randn(self._hnodes_l1, self._hnodes_l2) * Matrix(Float64(sqrt(2/self._hnodes_l1)), self._hnodes_l1, self._hnodes_l2)
        self._who = Matrix.randn(self._hnodes_l2, self._onodes) * Matrix(Float64(sqrt(2/self._hnodes_l2)), self._hnodes_l2, self._onodes)

        self._bih_l1 = Matrix.rand(1, self._hnodes_l1) - 0.5
        self._bih_l2 = Matrix.rand(1, self._hnodes_l2) - 0.5
        self._bho = Matrix.rand(1, self._onodes) - 0.5
        
        print('Initialized a neural network\n'
              'Input Nodes: ' + String(self._inodes) + '\n'
              'Hidden Nodes Layer 1: ' + String(self._hnodes_l1) + '\n'
              'Hidden Nodes Layer 2: ' + String(self._hnodes_l2) + '\n'
              'Output Nodes: ' + String(self._onodes))

    @staticmethod
    fn relu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.height, A.width)
        for i in range(B.height):
            for j in range(B.width):
                if A[i, j] > 0.0:
                    B[i, j] += A[i, j]
        return B
    
    @staticmethod
    fn softmax_1d(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.height, A.width)
        let e: Float64 = 2.71828
        var row_exp_sum: Float64 = 0.0
        
        for i in range(A.height):
            for j in range(A.width):
                B[i, j] = e ** A[i, j]

        for i in range(A.height):
            for j in range(A.width):
                row_exp_sum += A[i, j]

        for i in range(A.height):
            for j in range(A.width):
                B[i, j] /= row_exp_sum
        return B
    
    fn query(inout self, inputs: Matrix, targets: Matrix) -> Float64:
        let time_taken: Float64 = self.train(inputs, targets, train = False)
        return time_taken
    
    fn train(inout self, inputs: Matrix, targets: Matrix, train: Bool = True) -> Float64:
        var inputs_h1: Matrix = Matrix(inputs.width, self._wih.width)
        var inputs_h2: Matrix = Matrix(inputs_h1.height, self._whh.width)
        var outputs: Matrix = Matrix(inputs_h2.height, self._who.width)
        var output_error: Matrix = Matrix(1, self._onodes)
        var output_error_gradient: Matrix = Matrix(1, self._onodes)
        let hidden_errors_2: Matrix = Matrix(output_error_gradient.height, self._who.height)
        let hidden_errors_1: Matrix = Matrix(hidden_errors_2.height, self._whh.height)
        
        let time_now = now()
        
        Matrix.matmul_vectorized(inputs_h1, inputs.transpose(), self._wih)
        inputs_h1 = inputs_h1 + self._bih_l1
        inputs_h1 = self.relu(inputs_h1)
        
        Matrix.matmul_vectorized(inputs_h2, inputs_h1, self._whh)
        inputs_h2 = inputs_h2 + self._bih_l2
        inputs_h2 = self.relu(inputs_h2)
        
        Matrix.matmul_vectorized(outputs, inputs_h2, self._who)
        outputs = outputs + self._bho
        outputs = self.softmax_1d(outputs)
        
        output_error = targets.transpose() - outputs
        output_error_gradient = output_error
        
        Matrix.matmul_vectorized(hidden_errors_2, output_error_gradient, self._who.transpose())
        Matrix.matmul_vectorized(hidden_errors_1, (hidden_errors_2 * (inputs_h2 > Matrix(inputs_h2.height, inputs_h2.width))), self._whh.transpose())
        
        if train:
            self._update(inputs, inputs_h1, inputs_h2, hidden_errors_1, hidden_errors_2, output_error_gradient)
            let end_time = Float64(now() - time_now)
            return end_time
        else:
            let end_time = Float64(now() - time_now)
            return end_time

    fn _update(inout self, inputs: Matrix, inputs_h1: Matrix, inputs_h2: Matrix, hidden_errors_1: Matrix, hidden_errors_2: Matrix, output_error_gradient: Matrix):
        # not sure about drelu working as expected
        let ho2_drelu: Matrix = hidden_errors_2 * (inputs_h2 > Matrix(inputs_h2.height, inputs_h2.width))
        let ho1_drelu: Matrix = hidden_errors_1 * (inputs_h1 > Matrix(inputs_h1.height, inputs_h1.width))

        let ih2_o: Matrix = Matrix(inputs_h2.width, output_error_gradient.width)
        let ih1_ho2: Matrix = Matrix(inputs_h1.width, ho2_drelu.width)
        let i_ho1: Matrix = Matrix(inputs.height, ho1_drelu.width)

        let who_: Matrix = Matrix(self._who.height, self._who.width)
        let whh_: Matrix = Matrix(self._whh.height, self._whh.width)
        let wih_: Matrix = Matrix(self._wih.height, self._wih.width)

        let bho_: Matrix = Matrix(self._bho.height, output_error_gradient.width)
        let bih_l1_: Matrix = Matrix(self._bih_l1.height, ho2_drelu.width)
        let bih_l2_: Matrix = Matrix(self._bih_l2.height, ho1_drelu.height)

        Matrix.matmul_vectorized(ih2_o, inputs_h2.transpose(), output_error_gradient)
        Matrix.matmul_vectorized(ih1_ho2, inputs_h1.transpose(), ho2_drelu)
        Matrix.matmul_vectorized(i_ho1, inputs, ho1_drelu)
        
        Matrix.matmul_vectorized_scal_neg(self._who, ih2_o, self.lr)
        Matrix.matmul_vectorized_scal_neg(self._whh, ih1_ho2, self.lr)
        Matrix.matmul_vectorized_scal_neg(self._wih, i_ho1, self.lr)
        
        Matrix.matmul_vectorized_scal_neg(self._bho, output_error_gradient, self.lr)
        Matrix.matmul_vectorized_scal_neg(self._bih_l1, ho2_drelu, self.lr)
        Matrix.matmul_vectorized_scal_neg(self._bih_l2, ho1_drelu, self.lr)

        '''
        # slower than updating with scalar!
        
        let who_lr: Matrix = Matrix(Float64(self.lr),self._who.height, self._who.width)
        let whh_lr: Matrix = Matrix(Float64(self.lr),self._whh.height, self._whh.width)
        let wih_lr: Matrix = Matrix(Float64(self.lr),self._wih.height, self._wih.width)
        
        Matrix.matmul_vectorized_neg(self._who, ih2_o, who_lr)
        Matrix.matmul_vectorized_neg(self._whh, ih1_ho2, whh_lr)
        Matrix.matmul_vectorized_neg(self._wih, i_ho1, wih_lr)

        let bho_lr: Matrix = Matrix(Float64(self.lr),self._bho.height, output_error_gradient.width)
        let bih_l1_lr: Matrix = Matrix(Float64(self.lr),self._bih_l1.height, ho2_drelu.width)
        let bih_l2_lr: Matrix = Matrix(Float64(self.lr),self._bih_l2.height, ho1_drelu.height)
        
        Matrix.matmul_vectorized_neg(self._bho, bho_lr, output_error_gradient)
        Matrix.matmul_vectorized_neg(self._bih_l1, bih_l1_lr, ho2_drelu)
        Matrix.matmul_vectorized_neg(self._bih_l2, bih_l2_lr, ho1_drelu)
        '''
    