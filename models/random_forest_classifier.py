from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, auc

def train_random_forest(X_train, y_train_5):
    """Train a Random Forest classifier on the MNIST dataset"""
    forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]  # Score = probability of positive class
    return y_scores_forest

def evaluate_random_forest(X_train, y_train_5):
    """Evaluate the Random Forest classifier"""
    y_scores_forest = train_random_forest(X_train, y_train_5)
    roc_auc = roc_auc_score(y_train_5, y_scores_forest)
    fpr_forest, tpr_forest, _ = roc_curve(y_train_5, y_scores_forest)
    print("Random Forest ROC AUC:", roc_auc)
    return fpr_forest, tpr_forest, roc_auc
