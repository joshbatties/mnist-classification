from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
import numpy as np

class Never5Classifier(BaseEstimator):
    """A dummy classifier that always predicts 'not 5'"""
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

def evaluate_dummy_classifier(X_train, y_train_5):
    """Evaluate the dummy classifier using cross-validation"""
    never_5_clf = Never5Classifier()
    scores = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    print("Dummy classifier accuracy:", scores)
