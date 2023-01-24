import random
import numpy as np


class NeuralNetwork:

    def __init__(self, neuronsPerLayer):
        self.layers = len(neuronsPerLayer)
        self.neuronsPerLayer = neuronsPerLayer
        self.biases = [np.random.randn(n, 1) for n in neuronsPerLayer[1:]]
        # x is the number of neurons in the previous layer and y is the number of neurons in the current layer starting
        # in the second layer:
        self.weights = [np.random.randn(curLayerNeurons, prevLayerNeurons)  # (cols, rows)
                        for prevLayerNeurons, curLayerNeurons in zip(neuronsPerLayer[:-1], neuronsPerLayer[1:])]

    # Just for evaluating purposes. Or to be used after the learning process.
    def forward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def train(self, trainingData, repetitions, miniBatchSize, learningRate,
              testData=None):

        trainingData = list(trainingData)  # typecasting from zip list to list
        n = len(trainingData)

        if testData:
            testData = list(testData)
            nTest = len(testData)

        for repetition in range(repetitions):
            random.shuffle(trainingData)
            miniBatches = [
                trainingData[k:k + miniBatchSize]  # taking partitions of size miniBatchSize
                for k in range(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.adjustNeurons(miniBatch, learningRate)

            # Printing repetition result:
            if testData:
                print("Training Cycle {0}: {1} / {2}".format(
                    repetition, self.evaluate(testData), nTest))
            else:
                print("Training Cycle {0} complete".format(repetition))

    def adjustNeurons(self, miniBatch, learningRate):

        # Accumulators. Basically implementing a reducer.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        """For each sample in miniBatch we calculate the partial derivatives dC_s_j/dw_k and dC_s_j/db_n
         and sum it to the accumulator. Remember we want the gradient, in other words, we want the sum(dC_s_j/dw_k)
         for all j between 0 and len(miniBatch) for each weight k. Same thing for the biases. """
        for x, y in miniBatch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learningRate / len(miniBatch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learningRate / len(miniBatch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        curLayerError = (activations[-1] - y) * sigmoid_prime(zs[-1])  # eq 1

        nabla_b[-1] = curLayerError  # eq 3
        nabla_w[-1] = np.dot(curLayerError, activations[-2].transpose())  # eq 4

        for layer in range(2, self.layers):  # Applying eq 2
            z = zs[-layer]
            spz = sigmoid_prime(z)
            curLayerError = np.dot(self.weights[-layer + 1].transpose(), curLayerError) * spz
            nabla_b[-layer] = curLayerError
            nabla_w[-layer] = np.dot(curLayerError, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
