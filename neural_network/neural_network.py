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
            if (it % 500 == 0) and verbose:
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
        m = X.shape[0]  # number of samples

        # Forward pass
        A = np.insert(X, 0, 1.0, axis=1)  # a1 with bias
        cache = {'a1': A}
        for j in range(1, self.num_layers):
            Theta = self.params[f'Theta{j}']
            Z = A @ Theta.T
            A = sigmoid(Z)
            if j != self.num_layers - 1:
                # add bias except last layer
                A = np.insert(A, 0, 1.0, axis=1)
            cache[f'z{j+1}'] = Z
            cache[f'a{j+1}'] = A

        if mode == 'test':
           return A[:, 0]  # final output layer predictions

        # Compute loss
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        j = cost(Y, A)
        J = np.mean(np.sum(j, axis=1))

        # Add regularization to cost
        for l in range(1, self.num_layers):
            # get theta
            theta = self.params[f'Theta{l}']
            # skip bias term
            theta_no_bias = theta[:, 1:]
            S = (self.reg / (2 * m)) * np.sum(theta_no_bias ** 2)
            J += S

        # Backward pass
        grads = {}
        # Delta of last layer
        delta = A - Y
        # Reverse loop for backpropagation
        for l in range(self.num_layers - 1, 0, -1):
            # Get theta,a from cache
            A = cache[f'a{l}'] 
            theta = self.params[f'Theta{l}']

            grads[f'Theta{l}'] = (delta.T @ A) / m
            # Regularize (exclude bias column)
            grads[f'Theta{l}'][:, 1:] += (self.reg / m) * theta[:, 1:]

            if l > 1:
                theta_no_bias = theta[:, 1:]
                A_no_bias = A[:, 1:]
                delta = (delta @ theta_no_bias) * (A_no_bias * (1 - A_no_bias))

        return J, grads

    def loss_non_vectorized(self, X, Y=None, verbose=False):
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
            for j in range(self.num_layers-1, 0, -1):
                # Get theta,a from cache
                theta = self.params["Theta"+str(j)]
                a = cache["a"+str(j)]
                
                # Compute grad
                g = np.outer(delta, a)
                logger.debug("grads" + str(j)+": %s",  g)
                # Accumulate
                grads["Theta"+str(j)] += g
                
                if j > 1:
                    # Strip bias
                    theta_no_bias = theta[:,1:]
                    a_no_bias = a[1:]
                    # Calculate delta
                    delta = (theta_no_bias.T @ delta)*(a_no_bias*(1-a_no_bias))
                    logger.debug("delta"+str(j)+": %s", delta)
            
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


def cost(Y, fx):
    eps = 1e-8
    return -Y * np.log(fx + eps) - (1-Y)*np.log(1-fx + eps)