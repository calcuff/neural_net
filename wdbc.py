from neural_network.neural_network import NeuralNetwork
import numpy as np
from utils import normalize, shuffle_and_split, stratified_folds


def main():
    data = np.genfromtxt('./resources/datasets/wdbc.csv', delimiter=',', skip_header=1, dtype=float)
    print(data.shape)
    X = data[:, :-1]
    y = data[:, -1]
    y = y.flatten().astype(int)
    print("X", X.shape)
    print("Y", y.shape)
    
    X = normalize(X)
    
    X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
    
    X_train_folds, y_train_folds = stratified_folds(X_train, y_train, 5)
    
    for i, f in enumerate(X_train_folds):
        print("Fold:", i)
        x_train_data = np.concatenate([X_train_folds[j] for j in range(len(X_train_folds)) if j != i])
        y_train_data = np.concatenate([y_train_folds[j] for j in range(len(y_train_folds)) if j != i])
        validation_fold_x = X_train_folds[i]
        validation_fold_y = y_train_folds[i]
    
        nn = NeuralNetwork(input_dim=30, hidden_layer_dims=[40], output_dim=1, reg=0.0)
        nn.train(x_train_data, y_train_data, learning_rate=5e-2, num_iters=3000, verbose=True)
        
        y_pred = nn.predict(validation_fold_x)
        acc = np.mean(y_pred == validation_fold_y)
        print("Validation Accuracy", acc)
        
    nn = NeuralNetwork(input_dim=30, hidden_layer_dims=[40], output_dim=1, reg=0.0)
    nn.train(X_train, y_train, learning_rate=5e-2, num_iters=3000, verbose=True)
    y_pred = nn.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print("Testiing Accuracy", acc)



if __name__ == '__main__':
    main()