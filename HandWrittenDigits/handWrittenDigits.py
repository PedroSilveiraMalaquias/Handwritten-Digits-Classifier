from MNISTDataLoader import load_data_wrapper
from NeuralNetwork import NeuralNetwork

training_data, validation_data, test_data = load_data_wrapper()

#print(list(training_data)[:5])
network = NeuralNetwork([784, 30, 30, 10]) # 28 x 28 pixels
network.train(training_data, 30, 10, 1, test_data)


