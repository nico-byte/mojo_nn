from Network import Network
from types import Matrix
from python import Python
from time import now

fn main() raises:
    let input_nodes = 784
    let hidden_nodes_1 = 150
    let hidden_nodes_2 = 80
    let output_nodes = 10
    let learning_rate = 1e-5

    var nn = Network(input_nodes=input_nodes, hidden_nodes_l1=hidden_nodes_1, hidden_nodes_l2=hidden_nodes_2, output_nodes=output_nodes, learning_rate=learning_rate)
    # download dataset first - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    Python.add_to_path("./")
    let DataLoader = Python.import_module("DataLoader")
    let np = Python.import_module("numpy")

    var mnist_training_inputs = np.array
    var mnist_test_inputs = np.array

    mnist_training_inputs  = DataLoader.mnist_inputs("train")
    mnist_test_inputs = DataLoader.mnist_inputs("")

    var mnist_training_labels = np.array
    var mnist_test_labels = np.array

    mnist_training_labels  = DataLoader.mnist_labels("train", output_nodes)
    mnist_test_labels = DataLoader.mnist_labels("", output_nodes)
    mnist_test_inputs = mnist_test_inputs.T
    mnist_test_labels = mnist_test_labels.T
    print(mnist_test_inputs.shape)
    print(mnist_test_labels.shape)

    var inputs: Matrix = Matrix(784, 10000)
    var labels: Matrix = Matrix(10, 10000)

    print("Starting test data loader")
    for i in range(mnist_test_labels.shape[0]):
        for j in range(mnist_test_labels.shape[1]):
            labels[i, j] = mnist_test_labels[i][j].to_float64().cast[DType.float64]()            
    
    print("Starting train data loader")
    for i in range(mnist_test_inputs.shape[0]):
        for j in range(mnist_test_inputs.shape[1]):
            inputs[i, j] = mnist_test_inputs[i][j].to_float64().cast[DType.float64]()
    

    var output_error: Matrix = Matrix(10, 1)

    var new_input: Matrix = Matrix(input_nodes, 1)
    var new_label: Matrix = Matrix(output_nodes, 1)

    print("Start training")
    var iter_time: Float64 = 0.0
    var time_sum: Float64 = 0.0
    var iter: Int = 0
    print(labels.height, labels.width)
    print(new_label.height, new_label.width)
    
    # for now the loops only evaluate the time taken for the operations of the network
    # it outputs a float that represents the time taken for 1 iteration of the network
    # can be changed to output either the error or the outputs but need to be adjusted
    # in the train and query function of the network
    # training inputs and labels and test inputs and labels are ready to go
    
    var time_now = now()
    for i in range(inputs.width):
            for j in range(inputs.height):
                new_input[j, 0] = inputs[j, i]
                if j <= 9:
                    new_label[j, 0] = labels[j, i]
        iter_time = nn.train(new_input, new_label)
        time_sum += iter_time
        iter += 1

    var avg_time: Float64 = time_sum / inputs.width
    print("verify iterations: " + String(iter))
    print("avg mat duration/iter: " + String(avg_time / 1e3) + " microseconds")
    print("Runtime (Forward Pass + Backward Pass): " + String((now() - time_now) / 1e9) + " seconds")

    iter = 0
    time_sum = 0.0
    time_now = now()
    for i in range(inputs.width):
            for j in range(inputs.height):
                new_input[j, 0] = inputs[j, i]
                if j <= 9:
                    new_label[j, 0] = labels[j, i]
        iter_time = nn.query(new_input, new_label)
        time_sum += iter_time
        iter += 1

    avg_time = time_sum / inputs.width
    print("\nverify iterations: " + String(iter))
    print("avg mat duration/iter: " + String(avg_time / 1e3) + " microseconds")
    print("Runtime (Forward Pass): " + String((now() - time_now) / 1e9) + " seconds")
