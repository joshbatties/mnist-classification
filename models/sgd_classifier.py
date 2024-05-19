import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

def train_sgd_classifier(X_train, y_train):
    """Train an SGD classifier on the MNIST dataset"""
    y_train_5 = (y_train == 5)
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    return sgd_clf, y_train_5, None

def predict_digit(clf, digit):
    """Predict if a digit is a '5' using the SGD classifier"""
    return clf.predict([digit])

def cross_validate_sgd(clf, X_train, y_train_5):
    """Perform cross-validation and return decision scores"""
    y_scores = cross_val_predict(clf, X_train, y_train_5, cv=3, method="decision_function")
    return y_scores
