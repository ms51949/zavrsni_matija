from Dense import Dense
from Tanh import Tanh
from LossFun import LossFun
import numpy as np
import pandas as pa

data = pa.read_csv("toy4.csv")
X_train = np.array(data[['x1', 'x2']])
Y_train = np.array(data[['y']])
LOCATION_train = np.array(data[['lon', 'lat']])

size = LOCATION_train.shape[0]
rowNums = np.array(range(size))
rowNums = np.reshape(rowNums, (size, 1))
X_train = np.append(X_train, rowNums, axis=-1)

lossFun = LossFun(LOCATION_train, 1.801)

NETWORK = [
    Dense(2,5),
    Tanh(),
    Dense(5, size),
]


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def update_network(grad, network, learning_rate, iter):
    for layer in reversed(network):
        grad = layer.backward(grad, learning_rate, iter)

def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0
  
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches

def gradient(X, y, network):
    error = 0
    grad = 0
    for x, y_target in zip(X, y):
        row = x[-1]
        x = np.delete(x, -1, axis=-1)

        output = np.reshape(x, (2,1))
        output = predict(network, output)

        y_target = np.full((size, 1), y_target)

        error += lossFun.mse(y_target, output, row)
        grad += lossFun.mse_prime(y_target, output, row)
    
    grad = grad/len(X)
    error = error/len(X)

    return error, grad

def mini_batch(X, y, learning_rate, network, iter, batch_size):
    pomNum = 0
    errorPom=float('inf')
    model = network
    
    for i in range(iter):
        error = 0
        pomNum += 1
        mini_batches = create_mini_batches(X, y, batch_size)

        for mini_batch in mini_batches:
            X_batch, y_batch = mini_batch
            if(len(X_batch)==0):
                continue
            error_batch, grad_batch= gradient(X_batch, y_batch, network)
            error = error + error_batch
            update_network(grad_batch, network, learning_rate, i+1)
        
        error = error/len(mini_batches)
        if pomNum >= 1000:
            break
        if(error < errorPom):
            pomNum=0
            model = network
            errorPom = error
        print(" %d/%d, error = %f, pomNum = %d" % (i + 1, iter, error, pomNum))
    
    return model


def adam(X, y, learning_rate, network, iter, batch_size):
    errorPom=float('inf')
    pomNum = 0
    model = network
    for layer in network:
        layer.adam()

    for i in range(iter):
        error = 0
        mini_batches = create_mini_batches(X, y, batch_size)
        pomNum += 1
        for mini_batch in mini_batches:
            X_batch, y_batch = mini_batch
            if(len(X_batch)==0):
                continue
            error_batch, grad_batch= gradient(X_batch, y_batch, network)
            update_network(grad_batch, network, learning_rate, i+1)
            error = error + error_batch
        
        error = error/len(mini_batches)

        if pomNum >= 1000:
            break
        if(error < errorPom):
            pomNum=0
            model = network
            errorPom = error
        print(" %d/%d, error = %f, pomNum = %d" % (i + 1, iter, error, pomNum))
    
    return model

def nesterov(X, y, learning_rate, network, iter, batch_size):
    errorPom=float('inf')
    pomNum=0
    model = network
    for layer in network:
        layer.nesterov()
    for i in range(iter):
        error = 0
        pomNum += 1
        mini_batches = create_mini_batches(X, y, batch_size)

        for mini_batch in mini_batches:
            X_batch, y_batch = mini_batch
            if(len(X_batch)==0):
                continue
            error_batch, grad_batch= gradient(X_batch, y_batch, network)
            update_network(grad_batch, network, learning_rate, i+1)
            error = error + error_batch
        
        error = error/len(mini_batches)
        if pomNum >= 1000:
            break
        if(error < errorPom):
            pomNum=0
            model = network
            errorPom = error
        print(" %d/%d, error = %f, pomNum = %d" % (i + 1, iter, error, pomNum))
    
    return model



NETWORK = adam(X_train, Y_train, 0.01, NETWORK, 5600, 2)


W = NETWORK[2]
for i in range(5):
    w = W.weights.T[i]
    w = np.reshape(w, (25,25))
    np.savetxt("Neuron%d.txt" % (i+1), w)

w = W.bias.T[0]
w = np.reshape(w, (25,25))    
np.savetxt("Neuron%d.txt" % (6), w)

prediction = np.zeros((size, size))

X = np.delete(X_train, -1, axis=-1)
for i in range(size):
    output = np.reshape(X[i], (2,1))
    output = predict(NETWORK, output)
    for j in range(size):
        prediction[i][j] = output[j][0]

np.savetxt("PredictY.txt", prediction)

