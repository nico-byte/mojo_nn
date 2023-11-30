from random import rand, randn
from math import sqrt
from memory.unsafe import DTypePointer
import numjo as nj
from numjo import Matrix
from time import now
from math import tanh
from math import exp

struct Network:
    var _inodes: Int
    var _hnodes_l1: Int
    var _hnodes_l2: Int
    var _onodes: Int
    var lr: Float32
    var _wih: Matrix
    var _whh: Matrix
    var _who: Matrix
    var _bih_l1: Matrix
    var _bih_l2: Matrix
    var _bho: Matrix
    
    fn __init__(inout self, input_nodes: Int, hidden_nodes_l1: Int, hidden_nodes_l2: Int, output_nodes: Int, learning_rate: Float32):
        self._inodes = input_nodes
        self._hnodes_l1 = hidden_nodes_l1
        self._hnodes_l2 = hidden_nodes_l2
        self._onodes = output_nodes

        self.lr = learning_rate
        
        self._wih = Matrix(Float32(0.7), self._inodes, self._hnodes_l1)
        # self._wih = Matrix.randn(self._inodes, self._hnodes_l1)
        # randn(self._wih.data, self._wih.rows * self._wih.cols)
        
        self._whh = Matrix(Float32(0.4), self._hnodes_l1, self._hnodes_l2)
        # self._whh = Matrix.randn(self._hnodes_l1, self._hnodes_l2)
        # randn(self._whh.data, self._whh.rows * self._whh.cols)
        
        self._who = Matrix(Float32(0.15), self._hnodes_l2, self._onodes)
        # self._who = Matrix.randn(self._hnodes_l2, self._onodes)
        # randn(self._who.data, self._who.rows * self._who.cols)

        self._bih_l1 = Matrix(Float32(0.2), 1, self._hnodes_l1)
        # self._bih_l1 = Matrix.rand(1, self._hnodes_l1)
        # rand(self._bih_l1.data, self._bih_l1.rows * self._bih_l1.cols)

        self._bih_l2 = Matrix(Float32(0,35), 1, self._hnodes_l2)
        # self._bih_l2 = Matrix.randn(1, self._hnodes_l2)
        # rand(self._bih_l2.data, self._bih_l2.rows * self._bih_l1.cols)
        
        self._bho = Matrix(Float32(0.45), 1, self._onodes)
        # self._bho = Matrix.randn(1, self._onodes)
        # rand(self._bho.data, self._bho.rows * self._bho.cols)
        
        print('Initialized a neural network\n'
              'Input Nodes: ' + String(self._inodes) + '\n'
              'Hidden Nodes Layer 1: ' + String(self._hnodes_l1) + '\n'
              'Hidden Nodes Layer 2: ' + String(self._hnodes_l2) + '\n'
              'Output Nodes: ' + String(self._onodes))

    @staticmethod
    fn relu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols, True)
        for i in range(B.rows):
            for j in range(B.cols):
                if A[i, j] > 0.01:
                    B[i, j] = A[i, j]
                else:
                    B[i, j] = 0.0
        return B

    @staticmethod
    fn drelu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.rows, A.cols, True)
        for i in range(B.rows):
            for j in range(B.cols):
                if A[i, j] > 0.01:
                    B[i, j] = 1.0
                else:
                    B[i, j] = 0.0
        return B

    @staticmethod
    fn tanh(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.rows, A.cols, True)
        
        for i in range(A.rows):
            for j in range(A.cols):
                B[i, j] = tanh(A[i, j])
        return B

    @staticmethod
    fn dtanh(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.rows, A.cols, True)
        
        for i in range(A.rows):
            for j in range(A.cols):
                B[i, j] = 1.0 - tanh(A[i, j]) ** 2
        return B
    
    @staticmethod
    fn softmax_1d(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.rows, A.cols, True)
        
        var row_exp_sum_mat: Matrix = Matrix(A.rows, 1, True)
        for i in range(A.rows):
            for j in range(A.cols):
                    B[i, j] += exp(A[i, j])

        for i in range(A.rows):
            for j in range(A.cols):
                    row_exp_sum_mat[i, 0] += B[i, j]

        for i in range(A.rows):
            for j in range(A.cols):
                B[i, j] /= row_exp_sum_mat[i, 0]
        return B
    
    @staticmethod
    fn dmse(output_error: Matrix) -> Matrix:
        let deriv_coef: Float32 = 2.0 / output_error.cols
        let deriv = output_error * Matrix(Float32(deriv_coef), output_error.rows, output_error.cols)
        return deriv
    
    fn query(inout self, inputs: Matrix, targets: Matrix, peval: Bool = False) -> Matrix:
        let output: Matrix = self.train(inputs, targets, train = False, peval=peval)
        return output
    
    fn train(inout self, inputs: Matrix, targets: Matrix, train: Bool = True, peval: Bool = False) -> Matrix:
        # init some matrices
        var inputs_h1: Matrix = Matrix(inputs.rows, self._wih.cols)
        var inputs_h2: Matrix = Matrix(inputs_h1.rows, self._whh.cols)
        var output_error: Matrix = Matrix(inputs_h2.rows, self._onodes)
        var output_error_gradient: Matrix = Matrix(1, self._onodes)
        var hidden_errors_2: Matrix = Matrix(output_error_gradient.rows, self._who.rows)
        var hidden_errors_1: Matrix = Matrix(hidden_errors_2.rows, self._whh.rows)
        var outputs: Matrix = Matrix(1, self._onodes)
        
        let time_now = now()
        # calc output hidden layer1
        inputs_h1.zero()
        nj.matmul_vectorized(inputs_h1, inputs, self._wih)
        inputs_h1 = inputs_h1 + self._bih_l1
        inputs_h1 = self.relu(inputs_h1)
        
        # calc output hidden layer 2
        inputs_h2.zero()
        nj.matmul_vectorized(inputs_h2, inputs_h1, self._whh)
        inputs_h2 = inputs_h2 + self._bih_l2
        inputs_h2 = self.tanh(inputs_h2)
        
        # calc output output layer
        outputs.zero()
        nj.matmul_vectorized(outputs, inputs_h2, self._who)
        outputs = outputs + self._bho
        outputs = self.softmax_1d(outputs)
        
        output_error = (targets - outputs)**2
        var loss: Matrix = Matrix(1, 1)
        loss.store[1](0, 0, nj.mean(output_error)**2)
        output_error = Matrix(Float32(loss[0, 0]), output_error.rows, output_error.cols)
        output_error_gradient = self.dmse(output_error)
        
        nj.matmul_vectorized(hidden_errors_2, output_error_gradient, self._who.transpose())
        nj.matmul_vectorized(hidden_errors_1, (hidden_errors_2 * self.dtanh(inputs_h2)), self._whh.transpose())
        
        var end_time_mat: Matrix = Matrix(1, 1)

        if train:
            self._update(inputs, inputs_h1, inputs_h2, hidden_errors_1, hidden_errors_2, output_error_gradient)
            let end_time = Float32(now() - time_now)
            end_time_mat.store[1](0, 0, end_time)
            if peval:
                return end_time_mat
            else:
                return loss
        
        let end_time = Float32(now() - time_now)
        end_time_mat.store[1](0, 0, end_time)
        
        if peval:
            return end_time_mat
        
        return outputs
        
    fn _update(inout self, inputs: Matrix, inputs_h1: Matrix, inputs_h2: Matrix, hidden_errors_1: Matrix, hidden_errors_2: Matrix, output_error_gradient: Matrix):
        let ho2_drelu: Matrix = hidden_errors_2 * self.dtanh(inputs_h2)
        let ho1_drelu: Matrix = hidden_errors_1 * self.drelu(inputs_h1)

        var ih2_o: Matrix = Matrix(inputs_h2.cols, output_error_gradient.cols)
        var ih1_ho2: Matrix = Matrix(inputs_h1.cols, ho2_drelu.cols)
        var i_ho1: Matrix = Matrix(inputs.cols, ho1_drelu.cols)
        
        ih2_o.zero()
        nj.matmul_vectorized(ih2_o, inputs_h2.transpose(), output_error_gradient)
        ih1_ho2.zero()
        nj.matmul_vectorized(ih1_ho2, inputs_h1.transpose(), ho2_drelu)
        i_ho1.zero()
        nj.matmul_vectorized(i_ho1, inputs.transpose(), ho1_drelu)

        # updating weights and biases
        nj.update(self._who, ih2_o, self.lr)
        nj.update(self._whh, ih1_ho2, self.lr)
        nj.update(self._wih, i_ho1, self.lr)

        # sum of the A matrices would be better
        nj.update(self._bho, output_error_gradient, self.lr)
        nj.update(self._bih_l1, ho2_drelu, self.lr)
        nj.update(self._bih_l2, ho1_drelu, self.lr)
        