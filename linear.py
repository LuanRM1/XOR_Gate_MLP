import numpy as np
import math


class Perceptron:
    def __init__(self, weights=None, bias=-1, activation_threshold=0.5):
        if weights == None:
            self.weights = np.array([1, 1])
        else:
            self.weights = np.array(weights)
        self.bias = bias
        self.activation_threshold = activation_threshold

    def _heaviside(self, x):
        """
        Implementa a função delta de heaviside (famoso degrau)
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1 if x >= self.activation_threshold else 0

    def _sigmoid(self, x):
        """
        Implementa a função sigmoide
        Essa é uma função de ativação possível para os nós da rede neural.
        """
        return 1 / (1 + math.exp(-x))

    def _activation(self, perceptron_output):
        """
        Implementação da função de ativação do perceptron
        Escolha uma das funções de ativação possíveis
        """
        return self._heaviside(perceptron_output)

    def forward_pass(self, data):
        """
        Implementa a etapa de inferência (feedforward) do perceptron.
        """
        weighted_sum = self.bias + np.dot(self.weights, data)
        return self._activation(weighted_sum)

    def train(self, data, result, epoch, lr):
        for i in range(epoch):
            for inputs, expectec_output in zip(data, result):
                predicted_output = self.forward_pass(inputs)
                error = expectec_output - predicted_output
                print(error)
                for j in range(len(self.weights)):
                    self.weights[j] += error * lr * inputs[j]
                self.bias += lr * error


perceptron = Perceptron()


training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
training_outputs = np.array([0, 1, 1, 1])

learning_rate = 0.1
epochs = 100
perceptron.train(training_inputs, training_outputs, epochs, learning_rate)

print()
print(perceptron.forward_pass(np.array([0, 0])))
print(perceptron.forward_pass(np.array([0, 1])))
print(perceptron.forward_pass(np.array([1, 0])))
print(perceptron.forward_pass(np.array([1, 1])))
