import numpy as np
import logging

logger = logging.getLogger(__name__)

class NeuralNetwork():
    def __init__(self, layer_dims, params, weight_scale=1e-2, reg = 0.0):
        self.num_layers = len(layer_dims)
        self.reg = reg
        
        if params:
            self.params = params
        else:
            self.params = {}
            self.grads = {}
            for i in range(1, self.num_layers):
                in_dim = layer_dims[i-1] + 1 # add 1 for bias input
                out_dim = layer_dims[i]
                self.params['Theta' + str(i)] = weight_scale * np.random.randn(out_dim, in_dim)
                self.grads['Theta' + str(i)] = np.zeros_like(self.params['Theta' + str(i)])
    
    def loss(self, X, Y, verbose=False):
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            
        # Number of training samples
        m = X.shape[0]
        # Total loss
        J = 0.0
        
        # Setup grads
        grads = {}
        for i in range(1, self.num_layers):
            grads["Theta"+str(i)] = np.zeros_like(self.params["Theta"+str(i)])
        
        # Range training samples
        for i in range(m):
            x = X[i]
            y = Y[i]
            logger.debug("x: %s", x)
            logger.debug("y: %s", y)
            
            # Cache for backprop
            cache = {}
            
            # Insert bias
            a = np.insert(x, 0, 1.0)
            logger.debug('a1: %s', a)
            cache['a1'] = a

            # Forward pass through all hidden layers
            for i in range(1, self.num_layers):
                # Get theta
                theta = self.params["Theta"+str(i)]
                z = np.dot(theta,a)
                logger.debug("z"+str(i+1)+ ": %s", z)
                cache["z"+str(i+1)] = z
                a = sigmoid(z)
                if i != self.num_layers-1:
                    a = np.insert(a, 0, 1.0)
                logger.debug("a"+str(i+1)+ ": %s", a)
                cache["a"+str(i+1)] = a

            logger.debug("f(x): %s", a)
            
            # Cost function for this sample
            j = cost(y, a)
            # Add to total loss
            J += j
            logger.debug("Cost: %s\n", J)
            
            # Delta of last layer
            delta = a-y
            logger.debug("delta"+str(self.num_layers)+": %s", delta)
            
            # Reverse loop for backpropagation
            for i in range(self.num_layers-1, 0, -1):
                # Get theta,a from cache
                theta = self.params["Theta"+str(i)]
                a = cache["a"+str(i)]
                
                # Compute grad
                g = np.outer(delta, a)
                logger.debug("grads" + str(i)+": %s",  g)
                # Accumulate
                grads["Theta"+str(i)] += g
                
                if i > 1:
                    # Strip bias
                    theta_no_bias = theta[:,1:]
                    a_no_bias = a[1:]
                    # Calculate delta
                    delta = (theta_no_bias.T @ delta)*(a_no_bias*(1-a_no_bias))
                    logger.debug("delta"+str(i)+": %s", delta)
            
        # Average cost across training examples
        J /= m
        
        # Add regularization penalty to loss
        for i in range(1, self.num_layers):
            # get theta
            theta = self.params["Theta" + str(i)]
            # skip bias
            theta_no_bias = theta[:,1:]
            # L2 Regularization
            S = 1/(2*m) * self.reg * np.sum(theta_no_bias ** 2)
            J += S
            
            # Average gradients over training examples + regularization
            grads["Theta" + str(i)] /= m
            grads["Theta" + str(i)][:, 1:] += (self.reg / m) * theta_no_bias
        
        return J, grads
                           
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(y, fx):
    return np.sum((-y * np.log(fx)) - (1-y)*(np.log(1-fx)))