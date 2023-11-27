from random import rand, randn
from math import sqrt
from memory.unsafe import DTypePointer
from types import Matrix
from time import now
from math import tanh
from math import exp

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
    var _outputs: Matrix
    
    fn __init__(inout self, input_nodes: Int, hidden_nodes_l1: Int, hidden_nodes_l2: Int, output_nodes: Int, learning_rate: Float64, outputs: Matrix):
        self._inodes = input_nodes
        self._hnodes_l1 = hidden_nodes_l1
        self._hnodes_l2 = hidden_nodes_l2
        self._onodes = output_nodes

        self.lr = learning_rate

        self._wih = Matrix.randn(self._inodes, self._hnodes_l1) * Matrix(Float64(sqrt(2/self._inodes)), self._inodes, self._hnodes_l1)
        self._whh = Matrix.randn(self._hnodes_l1, self._hnodes_l2) * Matrix(Float64(sqrt(2/self._hnodes_l1)), self._hnodes_l1, self._hnodes_l2)
        self._who = Matrix.randn(self._hnodes_l2, self._onodes) * Matrix(Float64(sqrt(2/self._hnodes_l2)), self._hnodes_l2, self._onodes)

        self._bih_l1 = Matrix.rand(1, self._hnodes_l1)
        self._bih_l2 = Matrix.rand(1, self._hnodes_l2)
        self._bho = Matrix.rand(1, self._onodes)

        self._outputs = outputs
        
        print('Initialized a neural network\n'
              'Input Nodes: ' + String(self._inodes) + '\n'
              'Hidden Nodes Layer 1: ' + String(self._hnodes_l1) + '\n'
              'Hidden Nodes Layer 2: ' + String(self._hnodes_l2) + '\n'
              'Output Nodes: ' + String(self._onodes))

    @staticmethod
    fn lrelu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.height, A.width)
        for i in range(B.height):
            for j in range(B.width):
                if A[i, j] > 0.01:
                    B[i, j] = A[i, j]
                else:
                    B[i, j] = A[i, j] * 0.01

        return B

    @staticmethod
    fn dlrelu(A: Matrix) -> Matrix:
        var B: Matrix = Matrix(A.height, A.width)
        for i in range(B.height):
            for j in range(B.width):
                if A[i, j] > 0.01:
                    B[i, j] = 1.0
                else:
                    B[i, j] = 0.01
        return B

    @staticmethod
    fn tanh(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.height, A.width)
        
        for i in range(A.height):
            for j in range(A.width):
                B[i, j] = tanh(A[i, j])
        return B

    @staticmethod
    fn dtanh(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.height, A.width)
        
        for i in range(A.height):
            for j in range(A.width):
                B[i, j] = 1.0 - tanh(A[i, j]) ** 2
        return B

    fn mse(inout self, A: Matrix) -> Float64:
        var sum: Float64 = 0.0
        for i in range(A.width):
            for j in range(A.height):
                sum += A[j, i]
        return (sum**2)/A.height

    
    @staticmethod
    fn softmax_1d(A: Matrix) -> Matrix:
        # could need optimization alot
        var B: Matrix = Matrix(A.height, A.width)
        var row_exp_sum: Float64 = 0.0
        
        for i in range(A.height):
            for j in range(A.width):
                B[i, j] += exp(A[i, j])

        for i in range(A.height):
            for j in range(A.width):
                row_exp_sum += B[i, j]

        for i in range(A.height):
            for j in range(A.width):
                B[i, j] /= row_exp_sum
        return B

    fn query(inout self, inputs: Matrix, targets: Matrix, peval: Bool = False) -> Float64:
        let output: Float64 = self.train(inputs, targets, train = False, peval=peval)
        return output
    
    fn train(inout self, inputs: Matrix, targets: Matrix, train: Bool = True, peval: Bool = False) -> Float64:
        var inputs_h1: Matrix = Matrix(inputs.width, self._wih.width)
        var inputs_h2: Matrix = Matrix(inputs_h1.height, self._whh.width)
        var output_error: Matrix = Matrix(1, self._onodes)
        var output_error_gradient: Matrix = Matrix(1, self._onodes)
        var hidden_errors_2: Matrix = Matrix(output_error_gradient.height, self._who.height)
        var hidden_errors_1: Matrix = Matrix(hidden_errors_2.height, self._whh.height)
        
        let time_now = now()
        
        Matrix.matmul_vectorized(inputs_h1, inputs.transpose(), self._wih)
        inputs_h1 = inputs_h1 + self._bih_l1
        inputs_h1 = self.lrelu(inputs_h1)
        
        Matrix.matmul_vectorized(inputs_h2, inputs_h1, self._whh)
        inputs_h2 = inputs_h2 + self._bih_l2
        inputs_h2 = self.tanh(inputs_h2)
        
        Matrix.matmul_vectorized(self._outputs, inputs_h2, self._who)
        self._outputs = self._outputs + self._bho
        self._outputs = self.softmax_1d(self._outputs)
        
        output_error = targets.transpose() - self._outputs
        output_error_gradient = output_error
        
        Matrix.matmul_vectorized(hidden_errors_2, output_error_gradient, self._who.transpose())
        Matrix.matmul_vectorized(hidden_errors_1, (hidden_errors_2 * self.dtanh(inputs_h2)), self._whh.transpose())
        
        if train:
            self._update(inputs, inputs_h1, inputs_h2, hidden_errors_1, hidden_errors_2, output_error_gradient)
            let end_time = Float64(now() - time_now)
            if peval:
                return end_time
            else:
                return self.mse(output_error)
        else:
            let end_time = Float64(now() - time_now)
            if peval:
                return end_time
        
        return self.mse(output_error)

    fn _update(inout self, inputs: Matrix, inputs_h1: Matrix, inputs_h2: Matrix, hidden_errors_1: Matrix, hidden_errors_2: Matrix, output_error_gradient: Matrix):
        let ho2_drelu: Matrix = hidden_errors_2 * self.dtanh(inputs_h2)
        let ho1_drelu: Matrix = hidden_errors_1 * self.dlrelu(inputs_h1)

        var ih2_o: Matrix = Matrix(inputs_h2.width, output_error_gradient.width)
        var ih1_ho2: Matrix = Matrix(inputs_h1.width, ho2_drelu.width)
        var i_ho1: Matrix = Matrix(inputs.height, ho1_drelu.width)
        
        # updating weights and biases
        Matrix.update(self._who, ih2_o, self.lr)
        Matrix.update(self._whh, ih1_ho2, self.lr)
        Matrix.update(self._wih, i_ho1, self.lr)

        Matrix.update(self._bho, output_error_gradient, self.lr)
        Matrix.update(self._bih_l1, ho2_drelu, self.lr)
        Matrix.update(self._bih_l2, ho1_drelu, self.lr)

        Matrix.matmul_vectorized(ih2_o, inputs_h2.transpose(), output_error_gradient)
        Matrix.matmul_vectorized(ih1_ho2, inputs_h1.transpose(), ho2_drelu)
        Matrix.matmul_vectorized(i_ho1, inputs, ho1_drelu)

        '''
        var who_: Matrix = Matrix(self._who.height, self._who.width)
        var whh_: Matrix = Matrix(self._whh.height, self._whh.width)
        var wih_: Matrix = Matrix(self._wih.height, self._wih.width)

        var bho_: Matrix = Matrix(self._bho.height, output_error_gradient.width)
        var bih_l1_: Matrix = Matrix(self._bih_l1.height, ho2_drelu.width)
        var bih_l2_: Matrix = Matrix(self._bih_l2.height, ho1_drelu.height)
        
        # for reference
        Matrix.update(who_, ih2_o, self.lr)
        Matrix.update(whh_, ih1_ho2, self.lr)
        Matrix.update(wih_, i_ho1, self.lr)
        
        Matrix.update(bho_, output_error_gradient, self.lr)
        Matrix.update(bih_l1_, ho2_drelu, self.lr)
        Matrix.update(bih_l2_, ho1_drelu, self.lr)

        if self._who != who_:
            print("Weights updated")
        else:
            print("Weights not updated")
        '''
        
    