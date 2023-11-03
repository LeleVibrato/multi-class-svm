# -*- coding: utf-8 -*-
# @Time: February 4, 2022
# @Author: sam255
# @School: SCNU (South China Normal University)
# @Filename: nlsvm.py
# @Software: SVM Multiclass Classifier
# @License: MIT

import numpy as np
from scipy import sparse
import osqp
from functools import partial
from typing import Callable, Literal, List


def nlsvm_solve(X: np.ndarray, y: np.ndarray, classes: tuple[str, str], cost: float, kernel: Callable) -> Callable:
    """
    Train a binary SVM classifier.

    :param X: Feature matrix of training data, with 'n' rows and 'dim' columns, 
              where 'dim' is the number of features. Each row represents a training sample, 
              each column represents a feature.
    :param y: Label matrix for training data, 'n' rows and 1 column, with values -1 or 1.
    :param classes: A tuple of two strings, representing the classes for the binary classification.
    :param cost: SVM hyperparameter, the upper bound of alpha_i.
    :param kernel: Kernel function.
    :return: A callable SVM binary classifier function.
    """
    n = X.shape[0]  # Number of samples

    # Calculate the inner product
    inprod = np.ones((n, n))
    for i in range(n):
        for j in range(i+1):
            inprod[i, j] = inprod[j, i] = kernel(X[j, :], X[i, :])

    # Define the quadratic programming parameter matrices
    q = -np.ones(n)
    P = sparse.csc_matrix(0.5 * y * y.T * inprod)
    P[np.diag_indices_from(P)] = 2 * P[np.diag_indices_from(P)]
    A = sparse.csc_matrix(np.vstack((np.eye(n), y.T)))
    lb = np.zeros(n+1)
    ub = np.hstack([cost * np.ones(n), 0.0])

    # Solve the quadratic programming problem
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=False)
    res = prob.solve()
    alpha = res.x

    # Find the indices of the support vectors
    svs_ind = np.where((alpha > 1e-5) & (alpha < cost - 1e-5)
                       )[0]  # Epsilon for numerical stability

    # # Calculate beta0 (by averaging)
    # support_vectors = X[svs_ind, :]
    support_labels = y[svs_ind]
    beta0 = np.mean(1.0 / support_labels -
                    np.sum((alpha * y * inprod)[svs_ind], axis=1))

    def nlsvm_classifer(x: np.ndarray) -> str:
        """
        SVM binary classifier, returned as a closure by the parent function.

        :param x: A 1D np.ndarray of length 'dim'.
        :return: Predicted class for sample x.
        """
        result = beta0 + sum(alpha[i] * y[i, 0] *
                             kernel(x, X[i, :]) for i in range(n))
        return classes[0] if np.sign(result) == 1 else classes[1]

    return nlsvm_classifer


def ovo(y: np.ndarray) -> list[tuple[str, str]]:
    """
    Split classes into multiple one-vs-one tuples.

    :param y: A one-dimensional array of strings representing the classification of each sample.
    :return: A list of tuples, each representing the two classes for binary classification.
    """
    unique_class = np.unique(y)
    unique_class.sort()
    num_of_classes = len(unique_class)   # Number of unique classes
    print("classes: ", unique_class)
    return [(unique_class[i], unique_class[j])
            for i in range(num_of_classes) for j in range(i+1, num_of_classes)]


def get_kernel(kernel: str, kargs: dict) -> Callable:
    """
    Generate a kernel function based on the given parameters.

    :param kernel: Type of the kernel function.
    :param kargs: Hyperparameters for the kernel function.
    :return: A callable representing the kernel function.
    """

    if kernel == "poly":  # Polynomial kernel function
        def poly_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            return (1+np.dot(x1, x2))**kargs["degree"]
        return poly_kernel

    elif kernel == "linear":  # Linear kernel function
        def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            return np.inner(x1, x2)
        return linear_kernel

    elif kernel == "rbf":  # Gaussian (RBF) kernel function
        # Pre-compute the inverse of sigma squared
        sigma_squared_inv = 1 / (2 * (kargs["sigma"] ** 2))

        def gaussian_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            return np.exp(-np.linalg.norm(x1-x2)**2 * sigma_squared_inv)
        return gaussian_kernel

    else:
        raise ValueError("Unsupported kernel function")


def nlsvm(X: np.ndarray, y: np.ndarray, cost: float,
          kernel: Literal["linear", "poly", "rbf"], **kargs) -> List[Callable]:
    """
    Interface function for model training.

    :param X: Feature matrix of the training data, with n rows and dim columns, 
              where dim is the number of features, each row represents a training sample.
    :param y: Label array of the training data, a one-dimensional np.ndarray of strings.
    :param cost: A positive float, the SVM hyperparameter, the upper bound for alpha_i.
    :param kernel: Type of kernel function.
    :param **kargs: Hyperparameters for the kernel function.
    :return: List of SVM classifiers for multi-classification.
    """
    kernelf: Callable = get_kernel(kernel, kargs)
    n, dim = X.shape
    nclass = len(np.unique(y))
    classes = ovo(y)  # Split into multiple one-vs-one tuples
    models = []
    print(
        f"A total of {len(classes)} classifiers are required for this task...")
    for i, (class1, class2) in enumerate(classes):
        A = X[y == class1, :]
        B = X[y == class2, :]
        yA = np.ones((A.shape[0], 1))
        yB = -np.ones((B.shape[0], 1))
        X_tmp = np.vstack((A, B))
        y_tmp = np.vstack((yA, yB))
        print(
            f"Training classifier: {i + 1}/{len(classes)} ['{class1}', '{class2}']")
        models.append(nlsvm_solve(
            X_tmp, y_tmp, (class1, class2), cost, kernelf))
    print("Training complete!")
    print(
        f"Number of training samples: {n}; Dimension: {dim}; Number of classes: {nclass}")
    return models


def predict_1dim(models: List[Callable], x: np.ndarray) -> str:
    """
    Predict the class of a single sample.

    :param models: A list of SVM classifiers, each as a function closure.
    :param x: A 1-dimensional np.ndarray representing the sample to predict.
    :return: The predicted class for sample x.
    """
    # Predict the class for the sample using each submodel
    class_predictions = list(map(lambda submodel: submodel(x), models))

    # Count the occurrences of each predicted class
    unique_classes, counts = np.unique(class_predictions, return_counts=True)

    # Return the class that appears the most times
    return unique_classes[np.argmax(counts)]


def predict(models: List[Callable], X: np.ndarray) -> np.ndarray:
    """
    Predict the classes for multiple samples.

    :param models: A list of SVM classifiers, each as a function closure.
    :param X: Feature matrix of the data to predict, as an np.ndarray.
    :return: An np.ndarray of the predicted classes for each sample.
    """
    num_samples = X.shape[0]  # Number of samples
    print(f"Number of test samples: {num_samples}")
    predict_func = partial(predict_1dim, models)

    # Generate predictions for all samples
    predictions = list(map(predict_func, X))

    return np.array(predictions)


def accuracy(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions for test samples.

    :param y_pred: 1-dimensional array of predicted labels, as np.ndarray[str].
    :param y_test: 1-dimensional array of actual test labels, as np.ndarray[str].
    :return: The accuracy as a float.
    """
    correct_predictions = sum(y_pred == y_test)
    total_predictions = len(y_pred)
    return correct_predictions / total_predictions


if __name__ == '__main__':
    from datetime import datetime
    import os
    os.chdir(os.path.dirname(__file__))
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    starttime = datetime.now()
    iris_df = datasets.load_iris()  # Load the Iris dataset
    X = iris_df.data
    y = iris_df.target.astype(int).astype(str)

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)

    print("Training on the Iris dataset...")
    # Use the Gaussian kernel function with sigma=1.0
    model = nlsvm(X_train, y_train, cost=10.0, kernel="rbf", sigma=1.0)

    print("Testing the classifier...")
    y_pred = predict(model, X_test)
    print(f"Test set accuracy: {accuracy(y_pred, y_test)*100:.2f}%")
    endtime = datetime.now()
    print("Total time: ", (endtime - starttime).seconds, "seconds")

    # Code for another dataset (commented out)
    # import pandas as pd
    # data_train = pd.read_csv(
    #     "zip.train", delimiter=" ", header=None).to_numpy()
    # data_test = pd.read_csv("zip.test", delimiter=" ",
    #                         header=None).to_numpy()
    # X_train = data_train[:, 1:].astype(float)
    # y_train = data_train[:, 0].astype(int).astype(str)
    # X_test = data_test[:, 1:].astype(float)
    # y_test = data_test[:, 0].astype(int).astype(str)

    # starttime = datetime.now()
    # print("Training on the handwritten digits dataset...")
    # model = nlsvm(X_train, y_train, cost=1.0, kernel="poly", degree=3)
    # print("Testing the classifier...")
    # y_pred = predict(model, X_test)
    # print(f"Test set accuracy: {accuracy(y_pred, y_test)*100:.2f}%")
    # endtime = datetime.now()
    # print("Total time: ", (endtime - starttime).seconds, "seconds")
