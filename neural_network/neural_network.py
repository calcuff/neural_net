import numpy as np


class NeuralNetwork():
    def __init__(self, layer_dims, params, weight_scale=1e-2):
        self.params = {}
        self.num_layers = len(layer_dims)
        
        if params:
            self.params = params
        else:
            for i in range(1, self.num_layers):
                in_dim = layer_dims[i-1] + 1 # add 1 for bias input
                out_dim = layer_dims[i]
                self.params['Theta' + str(i)] = weight_scale * np.random.randn(out_dim, in_dim)
    
    def loss(self, X, y):
        print("x", X)
        print("y", y)
        cache = {}
        
        # Insert bias
        a = np.insert(X, 0, 1.0)
        print('a1', a)
        cache['a1'] = a

        # Forward pass through all hidden layers
        for i in range(1, self.num_layers):
            theta = self.params["Theta"+str(i)]
            z = np.dot(theta,a)
            print("z"+str(i+1), z)
            cache["z"+str(i+1)] = z
            a = sigmoid(z)
            if i != self.num_layers-1:
                a = np.insert(a, 0, 1.0)
            print("a"+str(i+1), a)
            cache["a"+str(i)] = a

        print("f(x)", a)
        
        J = cost(X.shape[0], y, a)
        print("Cost", J)
            
                           
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(n, y, fx):
    return (1/n) * np.sum((-y * np.log(fx)) - (1-y)*(np.log(1-fx)))