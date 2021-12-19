import numpy as np
from Layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def adam(self):
        pass

    def nesterov(self):
        pass

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate, iter):
        return np.multiply(output_gradient, self.activation_prime(self.input))