from data.load_data import load_mnist_data
from visualisation.plot_functions import plot_digit, plot_digits
from models.sgd_classifier import train_sgd_classifier, predict_digit, cross_validate_sgd
from models.dummy_classifier import evaluate_dummy_classifier
from evaluation.metrics import calculate_metrics, plot_precision_recall_vs_threshold, plot_precision_vs_recall, plot_roc_curve
from models.random_forest_classifier import evaluate_random_forest
from models.svm_classifier import evaluate_svm
from models.knn_classifier import evaluate_knn
from evaluation.cross_validation import stratified_k_fold_cross_validation

import matplotlib.pyplot as plt

# Load the MNIST data
X_train, X_test, y_train, y_test = load_mnist_data()

# Visualize a digit
some_digit = X_train[0]
plot_digit(some_digit)
plt.show()

# Train and evaluate the SGD classifier
sgd_clf, y_train_5, y_test_5 = train_sgd_classifier(X_train, y_train)
y_scores = cross_validate_sgd(sgd_clf, X_train, y_train_5)
calculate_metrics(y_train_5, y_scores)

# Perform Stratified K-Fold cross-validation
stratified_k_fold_cross_validation(sgd_clf, X_train, y_train_5)

# Train and evaluate the Dummy classifier
evaluate_dummy_classifier(X_train, y_train_5)

# Plot precision-recall and ROC curves
plot_precision_recall_vs_threshold(y_train_5, y_scores)
plt.show()
plot_precision_vs_recall(y_train_5, y_scores)
plt.show()
plot_roc_curve(y_train_5, y_scores)
plt.show()

# Train and evaluate the Random Forest classifier
evaluate_random_forest(X_train, y_train_5)

# Train and evaluate the SVM classifier
evaluate_svm(X_train, y_train)

# Train and evaluate the KNN classifier
evaluate_knn(X_train, y_train, X_test, y_test)
