from Network import Network
from types import Matrix
from python import Python
from time import now

fn main() raises:
    let input_nodes = 784
    let hidden_nodes_1 = 150
    let hidden_nodes_2 = 80
    let output_nodes = 10
    let learning_rate = 1e-4
    let outputs = Matrix(1, output_nodes)
    
    var peval_nn = Network(input_nodes=input_nodes, hidden_nodes_l1=hidden_nodes_1, hidden_nodes_l2=hidden_nodes_2, output_nodes=output_nodes, learning_rate=learning_rate, outputs=outputs)
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
    
    mnist_train_inputs = mnist_train_inputs.T
    mnist_train_labels = mnist_train_labels.T

    mnist_test_inputs = mnist_test_inputs.T
    mnist_test_labels = mnist_test_labels.T

    let test_inputs: Matrix = Matrix(784, 10000)
    let test_labels: Matrix = Matrix(10, 10000)

    let train_inputs: Matrix = Matrix(784, 60000)
    let train_labels: Matrix = Matrix(10, 60000)
    
    print("Starting train data converter")
    for i in range(mnist_test_inputs.shape[0]):
        for j in range(mnist_test_inputs.shape[1]):
            train_inputs[i, j] = mnist_train_inputs[i][j].to_float64().cast[DType.float64]()
            if train_inputs[i, j] <= 0.01:
                train_inputs[i,j] = 0.0
    
    for i in range(mnist_test_labels.shape[0]):
        for j in range(mnist_test_labels.shape[1]):
            train_labels[i, j] = mnist_train_labels[i][j].to_float64().cast[DType.float64]()
    
    print("Starting test data converter")
    for i in range(mnist_test_inputs.shape[0]):
        for j in range(mnist_test_inputs.shape[1]):
            test_inputs[i, j] = mnist_test_inputs[i][j].to_float64().cast[DType.float64]()
            if test_inputs[i, j] <= 0.01:
                test_inputs[i,j] = 0.0
    
    for i in range(mnist_test_labels.shape[0]):
        for j in range(mnist_test_labels.shape[1]):
            test_labels[i, j] = mnist_test_labels[i][j].to_float64().cast[DType.float64]()
    

    var mse: Float64 = 0.0

    var new_input: Matrix = Matrix(input_nodes, 1)
    var new_label: Matrix = Matrix(output_nodes, 1)

    print("Start training")
    var iter_time: Float64 = 0.0
    var time_sum: Float64 = 0.0
    var iter: Int = 0
    
    # for now the loops only evaluate the time taken for the operations of the network
    # it outputs a float that represents the time taken for 1 iteration of the network
    # can be changed to output either the error or the outputs but need to be adjusted
    # in the train and query function of the network
    # training inputs and labels and test inputs and labels are ready to go
    
    var time_now = now()
    for i in range(test_inputs.width):
            for j in range(test_inputs.height):
                new_input[j, 0] = test_inputs[j, i]
                if j <= 9:
                    new_label[j, 0] = test_labels[j, i]
        iter_time = peval_nn.train(new_input, new_label, peval=True)
        time_sum += iter_time
        iter += 1

    var avg_time: Float64 = time_sum / test_inputs.width
    print("verify iterations: " + String(iter))
    print("avg mat duration/iter: " + String(avg_time / 1e3) + " microseconds")
    print("Runtime (Forward Pass + Backward Pass): " + String((now() - time_now) / 1e9) + " seconds")

    iter = 0
    time_sum = 0.0
    time_now = now()
    for i in range(test_inputs.width):
            for j in range(test_inputs.height):
                new_input[j, 0] = test_inputs[j, i]
                if j <= 9:
                    new_label[j, 0] = test_labels[j, i]
        iter_time = peval_nn.query(new_input, new_label, peval=True)
        time_sum += iter_time
        iter += 1

    avg_time = time_sum / test_inputs.width
    print("\nverify iterations: " + String(iter))
    print("avg mat duration/iter: " + String(avg_time / 1e3) + " microseconds")
    print("Runtime (Forward Pass): " + String((now() - time_now) / 1e9) + " seconds")
    '''
    var train_nn = Network(input_nodes=input_nodes, hidden_nodes_l1=hidden_nodes_1, hidden_nodes_l2=hidden_nodes_2, output_nodes=output_nodes, learning_rate=learning_rate, outputs=outputs)
    
    var loss: Float64 = 0.0
    var epochs: Int = 20
    
    for epoch in range(epochs):
        for i in range(train_inputs.width):
                for j in range(train_inputs.height):
                    new_input[j, 0] = train_inputs[j, i]
                    if j <= 9:
                        new_label[j, 0] = train_labels[j, i]
            mse = train_nn.train(new_input, new_label)
            loss += mse
        print("Loss: " + String(mse))
        print("Last Output:\n")
        train_nn._outputs.print_all()
        # new_input.print_all()
        train_nn._who.print_all()
    print("avg train loss: " + String(loss / train_inputs.width))
    loss = 0.0

    for i in range(test_inputs.width):
            for j in range(test_inputs.height):
                new_input[j, 0] = test_inputs[j, i]
                if j <= 9:
                    new_label[j, 0] = test_labels[j, i]
        mse = train_nn.query(new_input, new_label)
        loss += mse
        if i % 10000 == 0:
            print("Loss: " + String(mse))
            print("Last Output:\n")
            train_nn._outputs.print_all()
            # new_input.print_all()
            train_nn._who.print_all()
    print("avg train loss: " + String(loss / test_inputs.width))
    '''
