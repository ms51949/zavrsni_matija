from Layer import Layer
import numpy as np
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def adam(self,beta1=0.9, beta2=0.999, eps=1e-8):
        self.adam = True
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.Vdw = np.zeros((self.output_size, self.input_size))
        self.Vdb = np.zeros((self.output_size, 1))
        self.Sdw = np.zeros((self.output_size, self.input_size))
        self.Sdb = np.zeros((self.output_size, 1))

    def nesterov(self, beta=0.9):
        self.nesterov = True
        self.beta = beta
        self.Vdw = np.zeros((self.output_size, self.input_size))
        self.Vdb = np.zeros((self.output_size, 1))

    def backward(self, output_gradient, learning_rate, iter):
        if self.nesterov == True:
            weights_gradient = np.dot(output_gradient, self.input.T)
            input_gradient = np.dot(self.weights.T, output_gradient)
            
            Vdw_prev = self.Vdw
            Vdb_prev = self.Vdb

            self.Vdw = self.Vdw*self.beta - learning_rate*weights_gradient
            self.Vdb = self.Vdb*self.beta - learning_rate*output_gradient

            self.weights += -self.beta * Vdw_prev + self.Vdw + self.beta*self.Vdw
            self.bias += -self.beta * Vdb_prev + self.Vdb + self.beta*self.Vdb
            return input_gradient
        if self.adam == True:
            weights_gradient = np.dot(output_gradient, self.input.T)
            input_gradient = np.dot(self.weights.T, output_gradient)
            

            self.Vdw = self.Vdw*self.beta1 + (1-self.beta1)*weights_gradient
            self.Sdw = self.Sdw*self.beta2 + (1-self.beta2)*weights_gradient**2

            self.Vdb = self.Vdb*self.beta1 + (1-self.beta1)*output_gradient
            self.Sdb = self.Sdb*self.beta2 + (1-self.beta2)*output_gradient**2
            
            Vdw_c = self.Vdw /(1-(np.power(self.beta1,2)))
            Vdb_c = self.Vdb /(1-(np.power(self.beta1,2)))
            Sdw_c = self.Sdw /(1-(np.power(self.beta2,2)))
            Sdb_c = self.Sdb /(1-(np.power(self.beta2,2)))
            

            self.weights -= learning_rate * (Vdw_c/(np.sqrt(Sdw_c) + self.eps))
            self.bias -= learning_rate * (Vdb_c/(np.sqrt(Sdb_c)+self.eps))
            return input_gradient
        else:
            weights_gradient = np.dot(output_gradient, self.input.T)
            input_gradient = np.dot(self.weights.T, output_gradient)
            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * output_gradient
            return input_gradient
