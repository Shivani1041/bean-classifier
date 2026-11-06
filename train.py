"""Train the five classifiers on the Dry Bean dataset and save everything the app needs.

Run once:  python train.py
Creates:   test_data.csv, model/*.joblib, model/results.csv, model/best_params.txt
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from utils import TARGET, MODEL_FILES, MODEL_DIR, compute_metrics

SEED = 2026
TEST_SIZE = 0.25
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")
print("Dataset shape:", df.shape)
print(df[TARGET].value_counts(), "\n")

encoder = LabelEncoder()
y = encoder.fit_transform(df[TARGET])
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

# Test file keeps the readable class names (the app encodes them again)
test_out = X_test.copy()
test_out[TARGET] = encoder.inverse_transform(y_test)
test_out.to_csv("test_data.csv", index=False)
joblib.dump(encoder, f"{MODEL_DIR}/label_encoder.joblib")
joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.joblib")
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

# ---------------------------------------------------------------- models
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

def tuned(pipe, grid):
    """Small grid search (cross-validation on TRAIN data only)."""
    return GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=-1)

candidates = {
    "Logistic Regression": tuned(
        Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000))]),
        {"clf__C": [0.5, 1, 5, 20]}),
    "Decision Tree": tuned(
        Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__max_depth": [6, 10, 14, None], "clf__min_samples_leaf": [2, 5, 10]}),
    "KNN": tuned(
        Pipeline([("scale", StandardScaler()),
                  ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [5, 7, 11, 21, 31]}),
    "Naive Bayes": tuned(
        Pipeline([("clf", GaussianNB())]),
        {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]}),
    "Support Vector Machine": tuned(
        Pipeline([("scale", StandardScaler()),
                  ("clf", SVC(probability=True, random_state=SEED))]),
        {"clf__C": [5, 10, 20], "clf__kernel": ["rbf"]}),
}

results, notes = {}, []
for name, model in candidates.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    best = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if hasattr(model, "best_params_"):
        notes.append(f"{name}: {model.best_params_}")
    joblib.dump(best, MODEL_FILES[name])

    pred, proba = best.predict(X_test), best.predict_proba(X_test)
    results[name] = compute_metrics(y_test, pred, proba, len(encoder.classes_))

table = pd.DataFrame(results).T.round(4)
table.index.name = "ML Model Name"
table.to_csv(f"{MODEL_DIR}/results.csv")
with open(f"{MODEL_DIR}/best_params.txt", "w") as f:
    f.write("\n".join(notes))

print("\n=== Test-set results ===")
print(table.to_string())
print("\nBest hyperparameters:\n" + "\n".join(notes))
