from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np

def stratified_k_fold_cross_validation(clf, X_train, y_train_5, n_splits=3):
    """Perform Stratified K-Fold cross-validation"""
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print("Stratified K-Fold accuracy:", n_correct / len(y_pred))
