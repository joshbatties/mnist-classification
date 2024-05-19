import numpy as np
from sklearn.datasets import fetch_openml, load_digits
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score

# Fetch the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Display the keys of the dataset
mnist.keys()

# Split the dataset into features (X) and labels (y)
X, y = mnist["data"], mnist["target"]

# Display the shape of the data and labels
X.shape
y.shape

# Select the first digit in the dataset
some_digit = X[0]

# Reshape the digit to a 28x28 image
some_digit_image = some_digit.reshape(28, 28)

# Display the digit using matplotlib
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

# Convert the labels to integers
y = y.astype(np.uint8)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Create binary labels for detecting the digit '5'
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Train a Stochastic Gradient Descent (SGD) classifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Predict if the first digit is a '5'
sgd_clf.predict([some_digit])

# Perform Stratified K-Fold cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# Define a dummy classifier that always predicts 'not 5'
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

# Evaluate the dummy classifier using cross-validation
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Make cross-validated predictions for the SGD classifier
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Compute and display the confusion matrix
confusion_matrix(y_train_5, y_train_pred)

# Compute precision, recall, and F1 score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)

# Obtain decision scores for the first digit
y_scores = sgd_clf.decision_function([some_digit])

# Apply different thresholds to the decision scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)
threshold = 8000
y_some_digit_pred = (y_scores > threshold)

# Obtain decision scores for all training data using cross-validation
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# Compute precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# Plot precision-recall vs. threshold graph
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])

# Find the threshold for 90% precision
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

# Plot the precision-recall vs. threshold graph with the 90% precision threshold
plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")
plt.plot([threshold_90_precision], [recall_90_precision], "ro")
plt.show()

# Plot precision vs. recall graph
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
plt.show()

# Make predictions using the 90% precision threshold
y_train_pred_90 = (y_scores >= threshold_90_precision)

# Compute precision and recall for the new predictions
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)

# Compute ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# Plot ROC curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)

plot_roc_curve(fpr, tpr)
plt.show()

# Compute the ROC AUC score
roc_auc_score(y_train_5, y_scores)

# Train a Random Forest classifier and compute ROC curve
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# Plot ROC curves for both classifiers
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
plt.show()

# Compute the ROC AUC score for the Random Forest classifier
roc_auc_score(y_train_5, y_scores_forest)

# Train and evaluate an SVM classifier
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # y_train, not y_train_5
svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
np.argmax(some_digit_scores)
svm_clf.classes_
svm_clf.classes_[5]

# Train a One-vs-Rest classifier using SVM
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)

# Train and evaluate an SGD classifier on the original problem
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# Standardize the input data and evaluate the SGD classifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# Make cross-validated predictions and compute confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

# Plot the confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# Normalize the confusion matrix
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# Plot the normalized confusion matrix
def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# Analyze misclassified instances
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)  # plot_digits not defined in provided code
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)  # plot_digits not defined in provided code
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)  # plot_digits not defined in provided code
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)  # plot_digits not defined in provided code
plt.show()

# Multi-label classification using K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# Predict the multi-labels for the first digit
knn_clf.predict([some_digit])

# Evaluate the KNN classifier using cross-validation
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# Multioutput classification with added noise
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# Display an example of noisy data and its target
some_index = 0
plt.subplot(121); plot_digit(X_test_mod[some_index])  # plot_digit not defined in provided code
plt.subplot(122); plot_digit(y_test_mod[some_index])  # plot_digit not defined in provided code
plt.show()

# Train the KNN classifier on the noisy data and predict a clean image
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)  # plot_digit not defined in provided code
