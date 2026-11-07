# Dry Bean Classifier

## a. Problem Statement
Predict the variety of a dry bean (one of 7 classes) from 16 shape and size measurements
extracted from images of the grains. This is a **multi-class classification** problem.
Five classifiers are trained and compared on the same dataset, and the results are
demonstrated in an interactive Streamlit web application.

## b. Dataset Description
- **Source:** Dry Bean Dataset, UCI Machine Learning Repository
  (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), Koklu & Ozkan (2020)
- **Instances:** 13,611 | **Features:** 16 numeric | **Target:** `Class` (7 classes)
- **Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
  (imbalanced — DERMASON has 3,546 instances, BOMBAY only 522)
- **Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation,
  Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness,
  ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
- **Missing values:** None
- **Preprocessing:** Stratified 75/25 train/test split (random seed 2026); StandardScaler
  applied inside pipelines of Logistic Regression, KNN, and SVM; target label-encoded
  with `LabelEncoder`.

## c. GitHub Repository
https://github.com/Shivani1041/bean-classifier

Live Streamlit App: <PASTE YOUR STREAMLIT APP LINK HERE>

## d. Models Used

All metrics are computed on the held-out test set (`test_data.csv`, 3,403 rows).
Precision, Recall, and F1 use **weighted averaging** to account for class imbalance.
AUC is computed using one-vs-rest (OVR) strategy with weighted averaging.

### Comparison Table

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9245   | 0.9933 | 0.9248    | 0.9245 | 0.9246 | 0.9088 |
| Decision Tree          | 0.9104   | 0.9720 | 0.9108    | 0.9104 | 0.9106 | 0.8916 |
| KNN                    | 0.9224   | 0.9875 | 0.9225    | 0.9224 | 0.9225 | 0.9062 |
| Naive Bayes            | 0.7617   | 0.9639 | 0.7619    | 0.7617 | 0.7608 | 0.7123 |
| Support Vector Machine | 0.9301   | 0.9941 | 0.9303    | 0.9301 | 0.9301 | 0.9154 |

### Observations

| ML Model Name          | Observation about model performance |
|------------------------|--------------------------------------|
| Logistic Regression    | Strong all-round performer. Accuracy (92.45%) and AUC (0.9933) are both high, and the small gap between precision and recall shows it handles the class imbalance well. Regularisation via the `C` hyperparameter prevents overfitting, making it a reliable baseline for this dataset. |
| Decision Tree          | Good accuracy (91.04%) but the lowest AUC (0.9720) among all models, indicating weaker probability calibration. Trees tend to overfit to training data and the drop in AUC suggests it is less confident near decision boundaries. Still interpretable and fast to predict. |
| KNN                    | Very close to Logistic Regression on all metrics (accuracy 92.24%, AUC 0.9875). Performance depends on feature scaling — without StandardScaler the results are noticeably worse. The main downside is higher prediction latency since it searches the full training set at inference time. |
| Naive Bayes            | Lowest accuracy (76.17%) and MCC (0.7123), but a surprisingly high AUC (0.9639), meaning its class probability estimates are reasonable even though its hard predictions are less accurate. The independence assumption between features likely hurts here because bean shape features are highly correlated (e.g., Area and ConvexArea). |
| Support Vector Machine | Best overall model — highest accuracy (93.01%), highest AUC (0.9941), and highest MCC (0.9154). The RBF kernel captures non-linear boundaries effectively, and the scaled feature space helps it separate the 7 classes cleanly. The trade-off is longer training time compared to the other models. |
| **Overall Winner**     | **Support Vector Machine** — highest on all six metrics: Accuracy (0.9301), AUC (0.9941), Precision (0.9303), Recall (0.9301), F1 (0.9301), and MCC (0.9154). Logistic Regression is a close second across all metrics and may be preferred when training speed or interpretability matters. |

## e. Streamlit App Features
- **CSV upload** — upload `test_data.csv` (or any compatible test file) from the sidebar
- **Model selection dropdown** — choose any of the 5 trained classifiers
- **Evaluation metrics** — displays Accuracy, AUC, Precision, Recall, F1, and MCC
- **Confusion matrix** — visual heatmap for the selected model
- **Classification report** — per-class precision, recall, and F1
- **Model comparison tab** — compare all 5 models on the uploaded data simultaneously
- No retraining is performed; all models are loaded from pre-saved `.joblib` files

## f. How to Run Locally
```bash
pip install -r requirements.txt
python train.py          # trains models, saves model/ and test_data.csv
streamlit run app.py
```

## g. Project Structure
```
app.py              Streamlit application
train.py            Trains and saves all five models
utils.py            Shared metric computation helper
data/               Original dataset (Dry_Bean_Dataset.xlsx)
model/              Saved .joblib models, label encoder, feature names, results.csv
test_data.csv       Held-out test set (25% of data, 3,403 rows)
requirements.txt    Python dependencies
```
