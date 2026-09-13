import sys # required for command line arguments
from sklearn.neural_network import MLPClassifier  # required for MLPClassifier

def perceptron(weights, inputs):
    """
    Computes the output of a perceptron given weights and inputs.
    
    :param weights: List of weights for the perceptron.
    :param inputs: List of inputs to the perceptron.
    :return: Output of the perceptron (1 or 0).
    """
    if len(weights) != len(inputs):
        raise ValueError("Weights and inputs must have the same length.")

    # TODO - implement the perceptron function

    # ADDED: calculate weighted sum
    weighted_sum = sum(w * x for w, x in zip(weights, inputs))

    return 1 if weighted_sum > 0 else 0


def train_perceptron(training_data, learning_rate=0.1, epochs=100):
    """
    Trains a perceptron using the provided training data.
    
    :param training_data: List of tuples (inputs, expected_output).
    :param learning_rate: Learning rate for weight updates.
    :param epochs: Number of epochs to train the perceptron.
    :return: List of weights after training.
    """
    if not training_data:
        raise ValueError("Training data cannot be empty.")

    num_inputs = len(training_data[0][0])
    weights = [0.0] * num_inputs

    for epoch in range(epochs):
        for inputs, expected in training_data:
            output = perceptron(weights, inputs)

            # TODO - implement your algorithm here

            # ADDED: calculate error
            error = expected - output

            # TODO - Update weights

            # ADDED: update weights using perceptron learning rule
            for i in range(num_inputs):
                weights[i] += learning_rate * error * inputs[i]

    return weights


#------------------------------------------------------------------------
# Training Data
#------------------------------------------------------------------------
function1_training_data = [
        ([1, 0, 1], 0), 
        ([0, 1, 1], 0),  
        ([0, 0, 1], 0),  
        ([1, 1, 1], 1)   
    ]

function2_training_data = [
        ([1, 0, 1], 1),  
        ([0, 1, 1], 1), 
        ([0, 0, 1], 0),  
        ([1, 1, 1], 1)   
    ]

function3_training_data = [
        ([1, 0, 1], 1),  
        ([0, 1, 1], 1),  
        ([0, 0, 1], 0),  
        ([1, 1, 1], 0)   
    ]


#------------------------------------------------------------------------
# Create a MLP and train to solve function 3 
#------------------------------------------------------------------------
def test_function3_on_mlp(p_verbose, p_solver, p_hidden_layer_nodes, p_activation):
    inputs = [data[0] for data in function3_training_data]
    outputs = [data[1] for data in function3_training_data]
    
    mlp = MLPClassifier(verbose=p_verbose, solver=p_solver, hidden_layer_sizes=(p_hidden_layer_nodes), activation=p_activation, max_iter=10000) 
    mlp.fit(inputs, outputs)

    # Get the structure of the MLP model
    if p_verbose:
        print("Number of layers (including input and output):", mlp.n_layers_)
        print("Number of outputs:", mlp.n_outputs_)
        print("Hidden layer sizes:", mlp.hidden_layer_sizes)

        # Access weights and biases
        print("Weights for each layer:", [coef.shape for coef in mlp.coefs_])
        print("Biases for each layer:", [bias.shape for bias in mlp.intercepts_])

        # Print actual numerical weights and biases
        print("Actual weights:", mlp.coefs_)
        print("Actual biases:", mlp.intercepts_)

    # Test the trained MLP
    if p_verbose:
        print()
        print("Test the trained MLP for Function 3")
    passed = True
    for inputs, expected in function3_training_data: 
        output = mlp.predict([inputs])[0]
        if output != expected:
            passed = False
        if p_verbose:  
            print(f"Function 3 (MLP): Input {inputs} classified as: {output} (Expected: {expected})")  
    return passed


#------------------------------------------------------------------------
# TODO: Train the perceptron manually from the command line
#------------------------------------------------------------------------
weights = list(map(float, sys.argv[1:]))

if 0: # set to 1 to test weights from command line inputs for function 1
    for inputs, expected in function1_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 1: Input {inputs} classified as: {output} (Expected: {expected})")

if 0: # set to 1 to test weights from command line inputs for function 2
    for inputs, expected in function2_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 2: Input {inputs} classified as: {output} (Expected: {expected})")

if 0: # set to 1 to test weights from command line inputs for function 3
    for inputs, expected in function3_training_data:
        output = perceptron(weights, inputs)
        print(f"Function 3: Input {inputs} classified as: {output} (Expected: {expected})")


#------------------------------------------------------------------------
# TODO: Test your algorithm to train the perceptron
#------------------------------------------------------------------------
if 1:  # ADDED: enable the required training tests
    function1_weights = train_perceptron(function1_training_data)
    function2_weights = train_perceptron(function2_training_data)
    function3_weights = train_perceptron(function3_training_data)

    print(f"Trained weights for Function 1: {function1_weights}")
    print(f"Trained weights for Function 2: {function2_weights}")
    print(f"Trained weights for Function 3: {function3_weights}")

    print()
    print("Test the trained perceptron")

    for inputs, expected in function1_training_data:
        output = perceptron(function1_weights, inputs)
        print(f"Function 1: Input {inputs} classified as: {output} (Expected: {expected})")

    print()

    for inputs, expected in function2_training_data:
        output = perceptron(function2_weights, inputs)
        print(f"Function 2: Input {inputs} classified as: {output} (Expected: {expected})")

    print()

    for inputs, expected in function3_training_data:
        output = perceptron(function3_weights, inputs)
        print(f"Function 3: Input {inputs} classified as: {output} (Expected: {expected})")


#------------------------------------------------------------------------
# Test the MLPClassifier on function 3 by calling test_function3_on_mlp()
#------------------------------------------------------------------------
# ADDED: MLP experiment 1
print()
print("MLP Test 1: lbfgs solver, 2 hidden nodes, logistic activation")

result1 = test_function3_on_mlp(
    True,
    "lbfgs",
    2,
    "logistic"
)

print("MLP successfully classified all inputs:", result1)

# ADDED: MLP experiment 2
print()
print("MLP Test 2: adam solver, 5 hidden nodes, relu activation")

result2 = test_function3_on_mlp(
    True,
    "adam",
    5,
    "relu"
)

print("MLP successfully classified all inputs:", result2)
