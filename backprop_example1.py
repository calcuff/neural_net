from neural_network.neural_network import NeuralNetwork
import numpy as np


def main():
    x = np.array([[0.13], [0.42]])
    y = np.array([[0.9], [0.23]])

    
    params = {
    'Theta1': np.array([[0.4, 0.1],
                        [0.3, 0.2]]),
    'Theta2': np.array([[0.7, 0.5, 0.6]])
    }
    
    nn = NeuralNetwork(layer_dims=[1,2,1], params=params)
    print("X1", x, "y1", y)
    loss, grads = nn.loss(x, y)
    print("Regulaized Loss over all training samples", loss)
    print("Final grads of theta1", grads["Theta1"])
    print("Final grads of theta2", grads["Theta2"])

if __name__ == '__main__':
    main()