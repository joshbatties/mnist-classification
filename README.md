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
   git clone https://github.com/jbatties/classification.git
   cd classification
