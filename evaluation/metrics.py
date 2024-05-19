import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve

def calculate_metrics(y_train_5, y_scores):
    """Calculate precision, recall, and F1 score"""
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
    y_train_pred_90 = (y_scores >= threshold_90_precision)
    print("Precision:", precision_score(y_train_5, y_train_pred_90))
    print("Recall:", recall_score(y_train_5, y_train_pred_90))
    print("F1 Score:", f1_score(y_train_5, y_train_pred_90))

def plot_precision_recall_vs_threshold(y_train_5, y_scores):
    """Plot precision-recall vs. threshold"""
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])

def plot_precision_vs_recall(y_train_5, y_scores):
    """Plot precision vs. recall"""
    precisions, recalls, _ = precision_recall_curve(y_train_5, y_scores)
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

def plot_roc_curve(y_train_5, y_scores):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_train_5, y_scores)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)
