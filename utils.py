"""Shared helpers used by both train.py and app.py."""
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "Class"
MODEL_DIR = "model"

# display name -> saved file
MODEL_FILES = {
    "Logistic Regression": f"{MODEL_DIR}/logistic_regression.joblib",
    "Decision Tree": f"{MODEL_DIR}/decision_tree.joblib",
    "KNN": f"{MODEL_DIR}/knn.joblib",
    "Naive Bayes": f"{MODEL_DIR}/naive_bayes.joblib",
    "Support Vector Machine": f"{MODEL_DIR}/svm.joblib",
}


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Return the six assignment metrics for a multi-class problem.

    Precision / Recall / F1 use weighted averaging (accounts for class imbalance).
    AUC is one-vs-rest, weighted. AUC is NaN if the uploaded data is missing a class.
    """
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                            average="weighted", labels=list(range(n_classes)))
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
