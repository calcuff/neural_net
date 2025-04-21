from neural_network.neural_network import NeuralNetwork
import numpy as np


def main():
    x = np.array([[0.32, 0.68], [0.83, 0.02]])
    y = np.array([[0.75, 0.98], [0.75, 0.28]])

    
    params = {
    'Theta1': np.array([[0.42, 0.15, 0.40],
                        [0.72, 0.10, 0.54],
                        [0.01, 0.19, 0.42],
                        [0.30, 0.35, 0.68]]),
    'Theta2': np.array([[0.21000,  0.67000,  0.14000,  0.96000,  0.87000],
                        [0.87000, 0.42000,  0.20000,  0.32000,  0.89000],
                        [0.03000,  0.56000,  0.80000 , 0.69000 , 0.09000]]),
    'Theta3': np.array([[0.04000,  0.87000,  0.42000,  0.53000 ],
                        [0.17000,  0.10000,  0.95000,  0.69000  ]])
    }
    
    nn = NeuralNetwork(input_dim=2, hidden_layer_dims=[4,3], output_dim=2, params=params, reg=0.250)
    print("X", x)
    print("y", y)
    loss, grads = nn.loss(x, y, verbose=True)
    print("\nRegulaized Loss over all training samples", loss)
    print("\nFinal grads of theta1\n", grads["Theta1"])
    print("\nFinal grads of theta2\n", grads["Theta2"])
    print("\nFinal grads of theta3\n", grads["Theta3"])

if __name__ == '__main__':
    main()