# 
#
#

# Use verbose to control the output of the perceptron and test functions
# verbose = "perceptron"  outputs the perceptron calculations
# verbose = "test"  # outputs the test results
verbose = "" # verbose disabled 

import numpy as np
import random

test_function_training_data = [
        ([1, 1, 1], 1),
        ([1, 0, 1], 1),
        ([0, 1, 1], 0),
        ([0, 0, 1], 0),
        ([1, 1, 1], 1),
        ([1, 0, 1], 1),
        ([0, 1, 1], 0),
        ([0, 0, 1], 0)
    ]

xor_training_data = [
        ([0, 0, 1], 0),
        ([0, 1, 1], 1),
        ([1, 0, 1], 1),
        ([1, 1, 1], 0)
    ]


#------------------------------------------------------------------------
# TODO: Code a linear activation function by completing the TODOs
# Using a linear activation function, the output of the perceptron will not
# be limited to binary values (0 or 1), but will instead be a continuous value.
# The perceptron will get close to 1 and close to 0, but not exactly 1 or 0.
#------------------------------------------------------------------------
def activation_function(weighted_sum):
    # TODO - implement the linear activation function
    return weighted_sum


#------------------------------------------------------------------------
# TODO: Code a perceptron by completing the TODOs
#------------------------------------------------------------------------
def perceptron(weights, inputs): 
    """
    Computes the output of a perceptron given weights and inputs.
    
    :param weights: List of weights for the perceptron.
    :param inputs: List of inputs to the perceptron.
    :return: Output of the perceptron (1 or 0).
    """
    if len(weights) != len(inputs):
        raise ValueError("Weights and inputs must have the same length.")
    
    # TODO - impliment the perceptron function
    weighted_sum = sum(weight * input_value
                       for weight, input_value in zip(weights, inputs))

    activate = activation_function(weighted_sum)

    if "perceptron" in verbose:
        print(f"Perceptron: Inputs: {inputs}, Weights: {weights}, Weighted sum: {weighted_sum}, Activate: {activate}")    

    return activate


#------------------------------------------------------------------------
# TODO: Calculate the Squared Error part of the Mean Squared Error (MSE)
#------------------------------------------------------------------------
def squared_error(expected, actual):    
    # TODO: implement the squared error function
    se = (expected - actual) ** 2
    return se


#------------------------------------------------------------------------
# TODO: Test the weights on a single sample of the training set
#------------------------------------------------------------------------
def test_perceptron(weights, inputs, expected):
    """
    Tests the perceptron with given weights and inputs, returning the Mean Squared Error (MSE).

    :param weights: List of weights for the perceptron.
    :param inputs: List of inputs to the perceptron.
    :param expected: Expected output for the given inputs.
    :return: Squared Error (SE) between expected and actual output.
    """
    actual = perceptron(weights, inputs)
  
    # TODO: Calculate the squared error between expected and actual output
    se = squared_error(expected, actual)

    if "test" in verbose:
        print(f"Test: Expected: {expected}, Actual: {actual}, SE: {se}")

    return se


#------------------------------------------------------------------------
# TODO: Test the weights across the entire training set
#------------------------------------------------------------------------
def batch_test_perceptron(weights, training_data = test_function_training_data):
    results = []

    for inputs, expected in training_data:
        squared_error_value = test_perceptron(weights, inputs, expected)
        results.append((inputs, expected, squared_error_value))

    # TODO: return Mean Squared Error (MSE) - average of all squared errors
    batch_mse = sum(result[2] for result in results) / len(results)

    return batch_mse


#------------------------------------------------------------------------
# Use this function to experiment with brute force training
#------------------------------------------------------------------------
def find_weights_using_brute_force(
        weight_range,
        learning_rate,
        training_data = test_function_training_data):

    epochs = 0

    for w1 in np.arange(
            weight_range[0],
            weight_range[1],
            learning_rate):

        for w2 in np.arange(
                weight_range[0],
                weight_range[1],
                learning_rate):

            for w3 in np.arange(
                    weight_range[0],
                    weight_range[1],
                    learning_rate):

                epochs += 1

                weights = [w1, w2, w3]

                mse = batch_test_perceptron(
                    weights,
                    training_data=training_data)

                if "brute_force" in verbose:
                    print(
                        f"Epoch {epochs}: "
                        f"Weights: {weights}, MSE: {mse}"
                    )

                if mse < 0.005:
                    print(
                        f"Found weights: {weights} "
                        f"after {epochs} epochs "
                        f"with MSE: {mse}"
                    )

                    return weights, epochs

    print(f"No suitable weights found after {epochs} epochs.")

    return None, epochs


#------------------------------------------------------------------------
# Use this function to experiment with random search for weights
#------------------------------------------------------------------------
def find_weights_using_random_search(max_epochs=10000):

    epochs = 0
    num_inputs = 3  # Number of inputs to the perceptron

    while True:

        epochs += 1

        weights = [
            random.uniform(-1, 1)
            for _ in range(num_inputs)
        ]

        mse = batch_test_perceptron(weights)

        if "random_search" in verbose:
            print(
                f"Epoch {epochs}: "
                f"Weights: {weights}, MSE: {mse}"
            )

        if mse < 0.005:
            print(
                f"Found weights after {epochs} epochs: "
                f"{weights}"
            )

            return weights

        if epochs >= max_epochs:
            print(
                f"Reached maximum epochs ({max_epochs}) "
                f"without finding suitable weights."
            )

            return None


#------------------------------------------------------------------------
# no derivative batch gradient estimation
# are we walking uphill or downhill
# adjust the offset slightly up and down, compute the loss,
# and see which direction (up or down) reduces the loss.
#------------------------------------------------------------------------
def batch_gradient_estimation(weights, training_rate):
    """
    Estimates the gradient for each weight by adjusting the weights slightly up and down.
    
    :param weights: List of weights for the perceptron.
    :param training_data: Training data to test the perceptron.
    :param training_rate: Step size for adjusting weights.
    :return: List of gradients for each weight.
    """

    gradients = []

    if "gradient1" in verbose:
        print(
            f"Calculating gradients for weights: "
            f"{weights} with training rate: {training_rate}"
        )

    for i in range(len(weights)):

        # Adjust the weight slightly up and down
        weights_up = weights[:]
        weights_down = weights[:]

        weights_up[i] += training_rate
        weights_down[i] -= training_rate
        
        mse_up = batch_test_perceptron(weights_up)
        mse_down = batch_test_perceptron(weights_down)

        gradient = (
            (mse_up - mse_down)
            / (2 * training_rate)
        )

        gradients.append(gradient)

        if "gradient2" in verbose:
            print(
                f"Weight {i}: "
                f"Up MSE: {mse_up}, "
                f"Down MSE: {mse_down}, "
                f"Gradient: {gradient}"
            )
    
    return gradients


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# Test the perceptron and activation function with some known good weights
if 1:  # Set to one to test your perceptron and activation function

    print()
    print("Testing perceptron and activation function")

    verbose = "test"

    weights = [
        0.8999999999999995,
        -2.220446049250313e-16,
        0.09999999999999964
    ]

    mse = batch_test_perceptron(weights)

    print("Batch MSE:", mse)
    print()

    weights = [
        0.9498386580831919,
        -0.04052028630978577,
        0.004611405977543281
    ]

    mse = batch_test_perceptron(weights)

    print("Batch MSE:", mse)
    print()

    weights = [
        0.92758834748299,
        0.052563750418611166,
        -0.002519118520321495
    ]

    mse = batch_test_perceptron(weights)

    print("Batch MSE:", mse)
    print()


#------------------------------------------------------------------------
# TODO: Experiment with step size and brute force search for weights
# What is the impact of training size when searching for weights?
#------------------------------------------------------------------------

print()
print("Experiment with step size and brute force search for weights")

verbose = ""

# Experiment 1
print()
print("Brute Force Experiment 1")
print("Weight range: (-1, 1), Learning rate: 0.1")

weights, epochs = find_weights_using_brute_force(
    weight_range=(-1, 1),
    learning_rate=0.1
)

print("Result weights:", weights)
print("Epochs:", epochs)

if weights is not None:
    print("Final MSE:", batch_test_perceptron(weights))


# Experiment 2
print()
print("Brute Force Experiment 2")
print("Weight range: (-1, 1), Learning rate: 0.05")

weights, epochs = find_weights_using_brute_force(
    weight_range=(-1, 1),
    learning_rate=0.05
)

print("Result weights:", weights)
print("Epochs:", epochs)

if weights is not None:
    print("Final MSE:", batch_test_perceptron(weights))


# Experiment 3
print()
print("Brute Force Experiment 3")
print("Weight range: (-2, 2), Learning rate: 0.1")

weights, epochs = find_weights_using_brute_force(
    weight_range=(-2, 2),
    learning_rate=0.1
)

print("Result weights:", weights)
print("Epochs:", epochs)

if weights is not None:
    print("Final MSE:", batch_test_perceptron(weights))


#------------------------------------------------------------------------
# XOR experiment required by the rubric
#------------------------------------------------------------------------

print()
print("XOR Brute Force Experiment")
print("Weight range: (-1, 1), Learning rate: 0.1")

xor_weights, xor_epochs = find_weights_using_brute_force(
    weight_range=(-1, 1),
    learning_rate=0.1,
    training_data=xor_training_data
)

print("XOR Result weights:", xor_weights)
print("XOR Epochs:", xor_epochs)

if xor_weights is not None:
    print(
        "XOR Final MSE:",
        batch_test_perceptron(
            xor_weights,
            training_data=xor_training_data
        )
    )


#------------------------------------------------------------------------
# TODO: Can you get the perceptron to train if we just try random weights?
# The rubric requires calling this function 10 times.
#------------------------------------------------------------------------

print()
print("Experiment with random weights")

verbose = ""

random_results = []

for run in range(1, 11):

    print()
    print(f"Random Search Run {run}")

    random_weights = find_weights_using_random_search(
        max_epochs=10000
    )

    if random_weights is not None:

        final_mse = batch_test_perceptron(random_weights)

        random_results.append(final_mse)

        print("Final weights:", random_weights)
        print("Final MSE:", final_mse)

    else:

        print("No suitable weights found.")


print()
print("Random Search Summary")

if len(random_results) > 0:

    print("Lowest MSE:", min(random_results))

else:

    print("No successful random search runs.")


#------------------------------------------------------------------------
# TODO: Calculate the weights for the perceptron using Batch Gradient Descent
# Batch Gradient Descent: Updates weights after computing the gradient
# over the entire training set.
#
# Experiment with multiple learning rates for the rubric.
#------------------------------------------------------------------------

print()
print("Calculate the weights for the perceptron using Batch Gradient Descent")

verbose = ""

learning_rates = [0.01, 0.05, 0.1]

for training_rate in learning_rates:

    print()
    print("------------------------------------------")
    print(
        f"Batch Gradient Descent "
        f"with learning rate: {training_rate}"
    )
    print("------------------------------------------")

    # TODO: Initialize weights with random values
    weights = [
        random.uniform(-1, 1)
        for _ in range(3)
    ]

    epochs = 0

    max_epochs = 10000

    while True:

        epochs += 1

        # Compute the Loss
        mse_of_all_samples_using_current_weights = (
            batch_test_perceptron(weights)
        )

        print(
            f"Epoch {epochs}: "
            f"Weights: {weights}, "
            f"MSE: {mse_of_all_samples_using_current_weights}"
        )

        # TODO: check if the MSE is low enough
        # or if a maximum number of iterations is reached.

        if mse_of_all_samples_using_current_weights < 0.01:

            print("Target MSE reached.")
            break

        if epochs >= max_epochs:

            print("Maximum number of epochs reached.")
            break

        # TODO: measure gradient for each weight
        # by adjusting the offset slightly up and down

        gradients = batch_gradient_estimation(
            weights,
            training_rate
        )

        # Update weights:
        # weight = weight - learning_rate * gradient

        # TODO: Update weights[]
        for i in range(len(weights)):

            weights[i] = (
                weights[i]
                - training_rate * gradients[i]
            )

    # The final parameters are your optimized values.
    print()
    print(
        f"Current Weights: {weights}, "
        f"MSE: {mse_of_all_samples_using_current_weights}, "
        f"training epochs: {epochs}"
    )