import numpy as np
import logging

logger = logging.getLogger(__name__)

class NeuralNetwork():
    def __init__(self, input_dim, hidden_layer_dims, output_dim, params=None, reg = 0.0, lr=1e-2):
        self.layer_dims = [input_dim] + hidden_layer_dims + [output_dim]
        self.num_layers = len(self.layer_dims) 
        self.reg = reg
        self.learning_rate=lr
        
        
        if params:
            self.params = params
        else:
            self.params = {}
            for i in range(1, len(self.layer_dims)):
                in_dim = self.layer_dims[i-1] + 1 # add 1 for bias input
                out_dim = self.layer_dims[i]
                self.params['Theta' + str(i)] = np.random.randn(out_dim, in_dim) * np.sqrt(1 / in_dim)
                
                
    def train(self, X, y, num_iters=1,verbose=False):
        loss_history = []
        for it in range(1,num_iters+1):
            loss, grads = self.loss(X, y)
            if (it % 500 == 0):
                if verbose:
                    print("Iteration", it, "loss", loss)
            loss_history.append(loss)
            for k in self.params:
                self.params[k] += -grads[k] * self.learning_rate
                
    def predict(self, X):
        scores = self.loss(X)
        scores = np.array(scores)
        # Convert score to 1 or 0 prediction
        predictions = (scores > 0.5).astype(int)
        return predictions
        
    
    def loss(self, X, Y=None, verbose=False):
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        mode = 'test' if Y is None else 'train'
        scores = []
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
            logger.debug("x: %s", x)
            
            # Cache for backprop
            cache = {}
            
            # Insert bias
            a = np.insert(x, 0, 1.0)
            logger.debug('a1: %s', a)
            cache['a1'] = a

            # Forward pass through all hidden layers
            for j in range(1, self.num_layers):
                # Get theta
                theta = self.params["Theta"+str(j)]
                z = np.dot(theta,a)
                logger.debug("z"+str(j+1)+ ": %s", z)
                cache["z"+str(j+1)] = z
                a = sigmoid(z)
                if j != self.num_layers-1:
                    a = np.insert(a, 0, 1.0)
                logger.debug("a"+str(j+1)+ ": %s", a)
                cache["a"+str(j+1)] = a


            logger.debug("f(x): %s", a)
            # If test mode return early
            if mode == 'test':
                scores.append(a[0])
                continue
            
            # Cost function for this sample
            j = cost(Y[i], a)
            logger.debug("Cost: %s\n", j)
            # Add to total loss
            J += j
            
            # Delta of last layer
            delta = a - Y[i]
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
            
        if mode == 'test':
            return scores
        
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
    eps = 1e-8
    return np.sum((-y * np.log(fx + eps)) - (1-y)*(np.log(1-fx + eps)))