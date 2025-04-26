from neural_network.neural_network import NeuralNetwork
import numpy as np
from utils import *
import time

def main():
    data = np.genfromtxt('./resources/datasets/loan.csv', delimiter=',',dtype=object, encoding=None, skip_header=1)
    print(data.shape)
    
    categorical_cols = [0, 1, 2, 3, 4, 9, 10]
    numerical_cols = [5, 6, 7, 8]
    
    # Extract numerical features
    X_numerical = data[:, numerical_cols].astype(float)
    X_numerical_norm = normalize(X_numerical)

    # Process categorical columns 
    X_categorical_encoded = one_hot_encode(data, categorical_cols)
    print("ENCODED", X_categorical_encoded.shape)
    
    # Stack cols together
    X = np.hstack((X_numerical_norm, X_categorical_encoded))

    # Labels
    y = data[:, -1].astype(int)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = shuffle_and_split(X, y)
    X_train_folds, y_train_folds = stratified_folds(X_train, y_train, 5)
    
    regs = [1e-1]#[0.0]#, 1e-3, 1e-2, 1e-1]
    hidden_dims = [[2], [4], [10], [20], [40], [20,10], [30,15], [40,20,10]]
    
    best_acc = 0.0
    best_model = None
    
    if False:
        for reg in regs:
            for hd in hidden_dims:
                accs = []
                f1s = []
                for i, _ in enumerate(X_train_folds):
                    x_train_data = np.concatenate([X_train_folds[j] for j in range(len(X_train_folds)) if j != i])
                    y_train_data = np.concatenate([y_train_folds[j] for j in range(len(y_train_folds)) if j != i])
                    validation_fold_x = X_train_folds[i]
                    validation_fold_y = y_train_folds[i]
                    
                    nn = NeuralNetwork(input_dim=21, hidden_layer_dims=hd, output_dim=1, reg=reg, lr=5e-1)
                    nn.train(x_train_data, y_train_data, num_iters=3000, verbose=True)

                    y_pred = nn.predict(validation_fold_x)
                    
                    tp, fp, tn, fn = confusion_matrix(y_pred, validation_fold_y)
                    acc = calc_accuracy(tp, tn, validation_fold_y.shape[0])
                    f1 = calc_f1_score(calc_precision(tp,fp), calc_recall(tp,fn))
                    accs.append(acc)
                    f1s.append(f1)
                    print("Fold", i, "Reg:", reg, "HD:", hd, "Validation Accuracy", f'{acc:.4}', "F1", f'{f1:.4}')
                avg_acc = np.mean(accs)
                avg_f1 = np.mean(f1s)
                print("AVERAGE Reg:", reg, "HD:", hd, "Validation Accuracy", f'{avg_acc:.4}', "F1", f'{avg_f1:.4}')
                if avg_acc > best_acc:
                    print("Using best model @ Reg:", reg, "HD:", hd)
                    best_acc = avg_acc
                    best_model = nn
                    
    best_model = NeuralNetwork(input_dim=21, hidden_layer_dims=[40, 20, 10], output_dim=1, reg=1e-1, lr=7e-1)
    best_model.train(X_train, y_train, num_iters=20000, verbose=True)
    y_pred = best_model.predict(X_test)
    tp, fp, tn, fn = confusion_matrix(y_pred, y_test)
    acc = calc_accuracy(tp, tn, y_test.shape[0])
    print("Testiing Accuracy", acc)



if __name__ == '__main__':
    main()