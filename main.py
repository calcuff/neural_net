from neural_network.neural_network import NeuralNetwork
import numpy as np


def main():
    x = np.array([0.13])
    y = np.array([0.9])
    
    params = {
    'Theta1': np.array([[0.4, 0.1],
                        [0.3, 0.2]]),
    'Theta2': np.array([[0.7, 0.5, 0.6]])
    }
    
    nn = NeuralNetwork(layer_dims=[1,2,1], params=params)
    nn.loss(x, y)
    pass
if __name__ == '__main__':
    main()