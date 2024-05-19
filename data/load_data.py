import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist_data():
    """Fetch and load the MNIST dataset"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]
    y = y.astype(np.uint8)
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test
