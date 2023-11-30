from Network import Network
from numjo import Matrix
from python import Python
from time import now
import numjo as nj
from random import randn, rand

fn main() raises:
    let input_nodes = 784
    let hidden_nodes_1 = 150
    let hidden_nodes_2 = 80
    let output_nodes = 10
    let learning_rate = 1e-4
    var outputs = Matrix(1, output_nodes, True)
    
    var peval_nn = Network(input_nodes=input_nodes, hidden_nodes_l1=hidden_nodes_1, hidden_nodes_l2=hidden_nodes_2, output_nodes=output_nodes, learning_rate=learning_rate)
    # var train_nn = Network(input_nodes=input_nodes, hidden_nodes_l1=hidden_nodes_1, hidden_nodes_l2=hidden_nodes_2, output_nodes=output_nodes, learning_rate=learning_rate)

    # download dataset first - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    Python.add_to_path("./")
    let DataLoader = Python.import_module("DataLoader")
    let np = Python.import_module("numpy")

    print("\nStarting python mnist data loader")
    var mnist_train_inputs = np.array
    var mnist_test_inputs = np.array

    mnist_train_inputs  = DataLoader.mnist_inputs("train")
    mnist_test_inputs = DataLoader.mnist_inputs("")

    var mnist_train_labels = np.array
    var mnist_test_labels = np.array

    mnist_train_labels  = DataLoader.mnist_labels("train", output_nodes)
    mnist_test_labels = DataLoader.mnist_labels("", output_nodes)
    
    let test_inputs: Matrix = Matrix(10000, 784, True)
    let test_labels: Matrix = Matrix(10000, 10, True)

    let train_inputs: Matrix = Matrix(60000, 784, True)
    let train_labels: Matrix = Matrix(60000, 10, True)
    
    print("Starting train data converter")
    for i in range(mnist_test_inputs.shape[0]):
        for j in range(mnist_test_inputs.shape[1]):
            train_inputs[i, j] = mnist_train_inputs[i][j].to_float64().cast[DType.float32]()
            if train_inputs[i, j] <= 0.01:
                train_inputs[i,j] = 0.0
    
    for i in range(mnist_test_labels.shape[0]):
        for j in range(mnist_test_labels.shape[1]):
            train_labels[i, j] = mnist_train_labels[i][j].to_float64().cast[DType.float32]()
    
    print("Starting test data converter")
    for i in range(mnist_test_inputs.shape[0]):
        for j in range(mnist_test_inputs.shape[1]):
            test_inputs[i, j] = mnist_test_inputs[i][j].to_float64().cast[DType.float32]()
            if test_inputs[i, j] <= 0.01:
                test_inputs[i,j] = 0.0
    
    for i in range(mnist_test_labels.shape[0]):
        for j in range(mnist_test_labels.shape[1]):
            test_labels[i, j] = mnist_test_labels[i][j].to_float64().cast[DType.float32]()
    

    var mse: Matrix = Matrix(1, 1, True)

    print("Start evaluating performance")
    var iter_time: Matrix = Matrix(1, 1, True)
    var time_sum: Float32 = 0.0
    var iter: Int = 0
    
    var new_input: Matrix = Matrix(1, input_nodes, True)
    var new_label: Matrix = Matrix(1, output_nodes, True)
    
    var time_now = now()
    for i in range(test_inputs.rows):
            for j in range(test_inputs.cols):
                new_input[0, j] = test_inputs[i, j]
                if j <= 9:
                    new_label[0, j] = test_labels[i, j]
        
        # new_input.print_all()
        # new_label.print_all()
        iter_time = peval_nn.train(new_input, new_label, peval=True)
        time_sum += iter_time[0, 0]
        iter += 1

    var avg_time: Float32 = time_sum / test_inputs.rows
    print("verify iterations: " + String(iter))
    print("avg mat duration/iter: " + String(avg_time / 1e3) + " microseconds")
    print("Runtime (Forward Pass + Backward Pass): " + String((now() - time_now) / 1e9) + " seconds")
    
    iter = 0
    time_sum = 0.0
    time_now = now()
    for i in range(test_inputs.rows):
            for j in range(test_inputs.cols):
                new_input[0, j] = test_inputs[i, j]
                if j <= 9:
                    new_label[0, j] = test_labels[i, j]
        iter_time = peval_nn.query(new_input, new_label, peval=True)
        time_sum += iter_time[0, 0]
        iter += 1

    avg_time = time_sum / test_inputs.rows
    print("\nverify iterations: " + String(iter))
    print("avg mat duration/iter: " + String(avg_time / 1e3) + " microseconds")
    print("Runtime (Forward Pass): " + String((now() - time_now) / 1e9) + " seconds")
    '''
    
    var loss: Float32 = 0.0
    let epochs: Int = 10
    print("Start training")
    for epoch in range(epochs):
        for i in range(50000, train_inputs.rows):
                for j in range(train_inputs.cols):
                    new_input[0, j] = train_inputs[i, j]
                    if j <= 9:
                        new_label[0, j] = train_labels[i, j]
            mse = train_nn.train(new_input, new_label)
            # loss += mse[0, 0]
            new_input.zero()
            new_label.zero()
        print("Epoch: " + String(epoch) +  " / Loss: " + String(mse[0,0]))
    print("avg train loss: " + String(loss / train_inputs.rows))
    loss = 0.0
    
    var scores: Int = 0

    for i in range(test_inputs.rows):
            for j in range(test_inputs.cols):
                new_input[0, j] = test_inputs[i, j]
                if j <= 9:
                    new_label[0, j] = test_labels[i, j]
        outputs = train_nn.query(new_input, new_label)
        # loss += mse[0, 0]
        
        # print(nj.argmax(outputs), nj.argmax(new_label))
        if nj.argmax(outputs) == nj.argmax(new_label):
            scores += 1
        # print("Test")
        # new_label.print_all()
        # outputs.print_all()
        # print(outputs.argmax(), new_label.argmax())
        new_input.zero()
        new_label.zero()
    
    print("Accuracy on test set: " + String(scores / test_inputs.rows))
    '''
    
