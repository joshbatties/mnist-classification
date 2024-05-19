import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from visualisation.plot_functions import plot_digit, plot_digits

def train_knn(X_train, y_multilabel):
    """Train a K-Nearest Neighbors (KNN) classifier"""
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    return knn_clf

def evaluate_knn(X_train, y_train, X_test, y_test):
    """Evaluate the KNN classifier"""
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = train_knn(X_train, y_multilabel)
    some_digit = X_train[0]
    print("KNN prediction for first digit:", knn_clf.predict([some_digit]))

    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    f1 = f1_score(y_multilabel, y_train_knn_pred, average="macro")
    print("KNN classifier F1 score:", f1)

    # Multioutput Classification with noise
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test

    some_index = 0
    plt.subplot(121); plot_digit(X_test_mod[some_index])
    plt.subplot(122); plot_digit(y_test_mod[some_index])
    plt.show()

    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digit(clean_digit)
