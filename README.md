# MNIST Classification Project
This project demonstrates the application of various machine learning classifiers to recognize handwritten digits from the MNIST dataset. 

## Overview

The MNIST Classification Project aims to provide a comprehensive example of how to preprocess data, train different machine learning models, evaluate their performance, and visualize the results. 

The models used include:
- Stochastic Gradient Descent (SGD) Classifier
- Dummy Classifier
- Random Forest Classifier
- Support Vector Machine (SVM) Classifier
- K-Nearest Neighbors (KNN) Classifier

### Description of Files and Directories

- **README.md**: Provides an overview and usage instructions for the project.
- **requirements.txt**: Lists all the dependencies required for the project.
- **main.py**: The main script to run the entire pipeline, integrating all functionalities.
- **data/**:
  - **load_data.py**: Script to load the MNIST dataset.
- **visualization/**:
  - **plot_functions.py**: Contains functions for plotting and visualizing the digit images.
- **models/**:
  - **sgd_classifier.py**: Functions to train and evaluate the SGD classifier.
  - **dummy_classifier.py**: Functions to evaluate a dummy classifier.
  - **random_forest_classifier.py**: Functions to train and evaluate the Random Forest classifier.
  - **svm_classifier.py**: Functions to train and evaluate the SVM classifier.
  - **knn_classifier.py**: Functions to train and evaluate the KNN classifier.
- **evaluation/**:
  - **metrics.py**: Contains functions to calculate evaluation metrics and plot precision-recall and ROC curves.
  - **cross_validation.py**: Contains functions to perform cross-validation.

## Setup
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/joshbatties/mnist-classification.git
   cd classification

2. Create a virtual environment and activate it:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages using pip:
   ```bash
    pip install -r requirements.txt

### Detailed Description of Each Module

#### Data Loading (`data/load_data.py`)

- `load_mnist_data()`: Fetches and loads the MNIST dataset, splitting it into training and test sets.

#### Visualization (`visualization/plot_functions.py`)

- `plot_digit(data)`: Plots a single digit image.
- `plot_digits(instances, images_per_row=10, **options)`: Plots multiple digit images in a grid format.

#### Models

**SGD Classifier (`models/sgd_classifier.py`)**:

- `train_sgd_classifier(X_train, y_train)`: Trains an SGD classifier.
- `predict_digit(clf, digit)`: Predicts if a digit is '5' using the trained SGD classifier.
- `cross_validate_sgd(clf, X_train, y_train_5)`: Performs cross-validation and returns decision scores.

**Dummy Classifier (`models/dummy_classifier.py`)**:

- `Never5Classifier(BaseEstimator)`: A dummy classifier that always predicts 'not 5'.
- `evaluate_dummy_classifier(X_train, y_train_5)`: Evaluates the dummy classifier using cross-validation.

**Random Forest Classifier (`models/random_forest_classifier.py`)**:

- `train_random_forest(X_train, y_train_5)`: Trains a Random Forest classifier.
- `evaluate_random_forest(X_train, y_train_5)`: Evaluates the Random Forest classifier and returns ROC curve values.

**SVM Classifier (`models/svm_classifier.py`)**:

- `train_svm(X_train, y_train)`: Trains an SVM classifier on a subset of the MNIST dataset.
- `evaluate_svm(X_train, y_train)`: Evaluates the SVM classifier.

**KNN Classifier (`models/knn_classifier.py`)**:

- `train_knn(X_train, y_multilabel)`: Trains a K-Nearest Neighbors classifier.
- `evaluate_knn(X_train, y_train, X_test, y_test)`: Evaluates the KNN classifier, including multi-label and multi-output classification with noise.

#### Evaluation

**Metrics (`evaluation/metrics.py`)**:

- `calculate_metrics(y_train_5, y_scores)`: Calculates precision, recall, and F1 score.
- `plot_precision_recall_vs_threshold(y_train_5, y_scores)`: Plots precision-recall vs. threshold curve.
- `plot_precision_vs_recall(y_train_5, y_scores)`: Plots precision vs. recall curve.
- `plot_roc_curve(y_train_5, y_scores)`: Plots ROC curve.

**Cross-Validation (`evaluation/cross_validation.py`)**:

- `stratified_k_fold_cross_validation(clf, X_train, y_train_5, n_splits=3)`: Performs Stratified K-Fold cross-validation.

### Model Evaluation

The models are evaluated using various metrics and visualization techniques:

#### Cross-Validation

Each model undergoes Stratified K-Fold cross-validation to ensure the evaluation is robust and not biased due to a single train-test split. The cross-validation process involves splitting the training data into `n_splits` (e.g., 3), training the model on `n_splits-1` folds, and validating it on the remaining fold. This process repeats `n_splits` times, providing a comprehensive evaluation of the model's performance.

#### Metrics

- **Precision, Recall, and F1 Score**: These metrics provide insights into the model's performance, especially in handling imbalanced datasets. Precision measures the accuracy of positive predictions, recall measures the model's ability to find all relevant cases, and the F1 score provides a balance between precision and recall.
- **ROC Curve**: The Receiver Operating Characteristic (ROC) curve illustrates the true positive rate against the false positive rate. It helps in visualizing the performance of the classifier at different threshold levels. The Area Under the ROC Curve (AUC) is a single scalar value that summarizes the overall performance.

#### Visualization

- **Precision-Recall vs. Threshold Curve**: This plot helps in understanding the trade-off between precision and recall for different decision thresholds. It is particularly useful for selecting an optimal threshold for classification.
- **Precision vs. Recall Curve**: This curve provides a direct comparison between precision and recall, helping to identify the threshold where both are balanced.
- **ROC Curve**: The ROC curve visualization allows for the comparison of different models based on their true positive and false positive rates. It helps in identifying the model that best separates the classes.
