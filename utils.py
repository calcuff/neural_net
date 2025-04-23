import numpy as np
import sklearn.utils, sklearn.model_selection
from typing import Tuple

# min-max normalization
def normalize(X):
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    X = (X - x_min)/(x_max - x_min)
    return X

# Shuffle data and split into training + test
def shuffle_and_split(X, y):
    # Shuffle data set
    X, y = sklearn.utils.shuffle(X,y)
    # Partition data into train + test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, train_size=0.8, test_size=0.2)
    return X_train, X_test, y_train, y_test


# Split data into num stratified folds
def stratified_folds(X:np.ndarray, y:np.ndarray, num_folds:int)-> Tuple[list, list]:
    # Shuffle data set
    X, y = sklearn.utils.shuffle(X,y)

    X_folds = []
    y_folds = []
    
    remaining_indices = np.arange(len(X))
    
    for i in range(num_folds, 1, -1):
        current_X = X[remaining_indices]
        current_y = y[remaining_indices]
        
        # Split into train of size 1/i. Proportion will grow each iteration as fold data is removed from original      
        fold_size = int(len(remaining_indices) / i)
        X_train, _, y_train, _ , train_idx, _ = sklearn.model_selection.train_test_split(current_X, current_y, remaining_indices, train_size=fold_size, stratify=current_y)
        
        # Store data for this fold
        X_folds.append(X_train)
        y_folds.append(y_train)
        
        # Remove this fold from remaining dataset so data is not used across folds
        remaining_indices = np.setdiff1d(remaining_indices, train_idx)
        
    # We have removed n-1 folds from the original, our last fold is what's left
    X_folds.append(X[remaining_indices])
    y_folds.append(y[remaining_indices])
    
    return X_folds, y_folds


def confusion_matrix(y_predict, y):
    positive_mask = y == 1
    y_positive = y[positive_mask]
    y_predict_positive = y_predict[positive_mask]

    y_negative = y[~positive_mask]
    y_predict_negative = y_predict[~positive_mask]

    true_positive = np.sum(y_predict_positive == y_positive)
    false_negative = np.size(y_predict_positive) - true_positive

    true_negative = np.sum(y_predict_negative == y_negative)
    false_positive =  np.size(y_predict_negative) - true_negative

    # accuracy = float(num_correct) / y_positive.shape[0]
    # print("------")
    # print("true_positive", true_positive, "false_negative", false_negative, "shape", y_predict_positive.shape[0])
    # print("true_negative", true_negative, "false_positive", false_positive, "shape", y_predict_negative.shape[0])
    return true_positive, false_positive, true_negative, false_negative


def calc_accuracy(tp, tn, total_test_count):
    return (tp+tn)/total_test_count

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_f1_score(precision, recall, beta=1):
    return (1 + beta**2)*(precision * recall)/(beta**2 * precision + recall)