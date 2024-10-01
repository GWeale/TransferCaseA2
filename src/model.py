import numpy as np
from src.transformer import Transformer

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes)-1):
            self.layers.append({'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]),
                                'biases': np.zeros((1, layer_sizes[i+1]))})
        self.learning_rate = learning_rate

    def forward(self, X):
        activations = [X]
        for layer in self.layers[:-1]:
            Z = activations[-1].dot(layer['weights']) + layer['biases']
            A = self.relu(Z)
            activations.append(A)
        Z = activations[-1].dot(self.layers[-1]['weights']) + self.layers[-1]['biases']
        A = self.softmax(Z)
        activations.append(A)
        return activations

    def backward(self, activations, Y):
        grads = []
        m = Y.shape[0]
        delta = activations[-1] - Y
        for i in reversed(range(len(self.layers))):
            A_prev = activations[i]
            dW = A_prev.T.dot(delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            grads.insert(0, {'dW': dW, 'db': db})
            if i != 0:
                delta = delta.dot(self.layers[i]['weights'].T) * self.relu_derivative(activations[i])
        return grads

    def update_parameters(self, grads):
        for i, layer in enumerate(self.layers):
            layer['weights'] -= self.learning_rate * grads[i]['dW']
            layer['biases'] -= self.learning_rate * grads[i]['db']

    def train(self, X, Y, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                activations = self.forward(X_batch)
                grads = self.backward(activations, Y_batch)
                self.update_parameters(grads)

    def predict(self, X):
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, A):
        return (A > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
