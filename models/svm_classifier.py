from sklearn.svm import SVC
import numpy as np

def train_svm(X_train, y_train):
    """Train an SVM classifier on a subset of the MNIST dataset"""
    svm_clf = SVC(gamma="auto", random_state=42)
    svm_clf.fit(X_train[:1000], y_train[:1000])  # Train on a subset for speed
    return svm_clf

def evaluate_svm(X_train, y_train):
    """Evaluate the SVM classifier"""
    svm_clf = train_svm(X_train, y_train)
    some_digit = X_train[0]
    print("SVM prediction for first digit:", svm_clf.predict([some_digit]))
    some_digit_scores = svm_clf.decision_function([some_digit])
    print("SVM decision function scores for first digit:", some_digit_scores)
