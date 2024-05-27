import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from params import ModelHyperParameters
from forward import L_model_forward
from backward import L_model_backward
import copy


def initialize_parameters_deep(model_hyper_parameters):
    model_parameters = {}
    L = len(model_hyper_parameters.layer_dims)
    layer_dims = model_hyper_parameters.layer_dims
    for l in range(1, L):
        model_parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        model_parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (model_parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (model_parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return model_parameters


def print_layer_sizes(model_hyper_parameters: ModelHyperParameters):
    print("The size of layer[0] (input layer) is: " + str(model_hyper_parameters.layer_dims[0]))
    for i in range(1, len(model_hyper_parameters.layer_dims)):
        print("The size of layer[{}] is: {}".format(i, model_hyper_parameters.layer_dims[i]))

    return model_hyper_parameters


def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    cost = - np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL))) / m
    cost = np.squeeze(cost)
    return cost


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    params -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

    return parameters


def load_data():
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('data//test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def model(X, Y, model_hyper_parameters, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(model_hyper_parameters)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, model_hyper_parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, model_hyper_parameters)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs


def predict(parameters, model_hyper_parameters, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    AL, caches = L_model_forward(X, parameters, model_hyper_parameters)
    print(AL)

    for i in range(m):
        if AL[0, i] > 0.5:
            Y_prediction[0, i] = 1.0
        else:
            Y_prediction[0, i] = 0.0

    return Y_prediction


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    hyper_parameters = ModelHyperParameters(L=4, layer_dims=[12288, 20, 7, 5, 1])
    parameters, costs = model(train_x, train_y, hyper_parameters, num_iterations=2500, print_cost=True)
    print(predict(parameters, hyper_parameters, test_x))
    print(predict(parameters, hyper_parameters, train_x))
