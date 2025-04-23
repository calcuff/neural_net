from neural_network.neural_network import NeuralNetwork
import numpy as np
from utils import normalize, shuffle_and_split, stratified_folds, confusion_matrix, calc_accuracy, calc_f1_score, calc_precision, calc_recall


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
    
    regs = [0.0, 1e-3, 1e-2, 1e-1]
    hidden_dims = [[10], [20], [40], [20,10], [30,15], [40,20,10]]
    
    best_acc = 0.0
    best_model = None
    
    if False:
        # for lr in lrs:
        for reg in regs:
            for hd in hidden_dims:
                accs = []
                f1s = []
                for i, _ in enumerate(X_train_folds):
                    x_train_data = np.concatenate([X_train_folds[j] for j in range(len(X_train_folds)) if j != i])
                    y_train_data = np.concatenate([y_train_folds[j] for j in range(len(y_train_folds)) if j != i])
                    validation_fold_x = X_train_folds[i]
                    validation_fold_y = y_train_folds[i]
                
                    nn = NeuralNetwork(input_dim=30, hidden_layer_dims=hd, output_dim=1, reg=reg, lr=1e-1)
                    nn.train(x_train_data, y_train_data,num_iters=500, verbose=False)
                    
                    y_pred = nn.predict(validation_fold_x)
                    tp, fp, tn, fn = confusion_matrix(y_pred, validation_fold_y)
                    acc = calc_accuracy(tp, tn, validation_fold_y.shape[0])
                    f1 = calc_f1_score(calc_precision(tp,fp), calc_recall(tp,fn))
                    accs.append(acc)
                    f1s.append(f1)
                avg_acc = np.mean(accs)
                avg_f1 = np.mean(f1s)
                print("Reg:", reg, "HD:", hd, "Validation Accuracy", f'{avg_acc:.4}', "F1", f'{avg_f1:.4}')
                if avg_acc > best_acc:
                    print("Using best model @ Reg:", reg, "HD:", hd)
                    best_acc = acc
                    best_model = nn
    best_model = NeuralNetwork(input_dim=30, hidden_layer_dims=[40], output_dim=1, reg=1e-2, lr=1e-1)
    best_model.train(X_train, y_train, num_iters=1000, verbose=True)
    y_pred = best_model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print("Testiing Accuracy", acc)


if __name__ == '__main__':
    main()