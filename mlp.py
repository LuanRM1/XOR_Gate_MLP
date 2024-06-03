import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-1, 1, hidden_size)

        self.weights_hidden_output = np.random.uniform(
            -1, 1, (output_size, hidden_size)
        )
        self.bias_output = np.random.uniform(-1, 1, output_size)

        self.learning_rate = lr

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _devsigmoid(self, x):
        return x * (1 - x)

    def foward_pass(self, inputs):
        self.hidden_input = (
            np.dot(inputs, self.weights_input_hidden.T) + self.bias_hidden
        )
        self.hidden_output = self._sigmoid(self.hidden_input)

        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output.T) + self.bias_output
        )
        self.final_output = self._sigmoid(self.final_input)

        return self.final_output

    def backward_pass(self, inputs, expected_output):
        output_error = expected_output - self.final_output
        output_delta = output_error * self._devsigmoid(self.final_output)

        hidden_error = output_delta.dot(self.weights_hidden_output)
        hidden_delta = hidden_error * self._devsigmoid(self.hidden_output)

        self.weights_hidden_output += self.learning_rate * np.outer(
            output_delta, self.hidden_output
        )
        self.bias_output += self.learning_rate * output_delta

        self.weights_input_hidden += self.learning_rate * np.outer(hidden_delta, inputs)
        self.bias_hidden += self.learning_rate * hidden_delta

    def train(self, training_inputs, training_outputs, epochs):
        for epoch in range(epochs):
            for inputs, outputs in zip(training_inputs, training_outputs):
                self.foward_pass(inputs)
                self.backward_pass(inputs, outputs)


training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_outputs = np.array([[0], [1], [1], [0]])

mlp = MLP(input_size=2, hidden_size=2, output_size=1, lr=0.1)

epochs = 10000
mlp.train(training_inputs, training_outputs, epochs)

# teste apos o treinamento
print(mlp.foward_pass(np.array([0, 0])))  # esperado: 0
print(mlp.foward_pass(np.array([0, 1])))  # esperado: 1
print(mlp.foward_pass(np.array([1, 0])))  # esperado: 1
print(mlp.foward_pass(np.array([1, 1])))  # esperado: 0
