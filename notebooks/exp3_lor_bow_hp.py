import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")


# MLflow + DAGsHub setup
MLFLOW_TRACKING_URI = "https://dagshub.com/somg0703/MLOps-Capstone-Project.mlflow"

dagshub.init(
    repo_owner="somg0703",
    repo_name="MLOps-Capstone-Project-2",
    mlflow=True
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoR Hyperparameter Tuning")


# ==========================
# TEXT PREPROCESSING
# ==========================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )

    return text.strip()


# ==========================
# LOAD + PREP DATA
# ==========================
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df["review"] = df["review"].astype(str).apply(preprocess_text)

    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# TRAIN + LOG MODELS
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    # 1) Train GridSearch normally
    grid_search = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # 2) Log each hyperparameter run separately
    for params, mean_score, std_score in zip(
        grid_search.cv_results_["params"],
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"]
    ):
        with mlflow.start_run(run_name=f"LR params: {params}", nested=True):

            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_params(params)
            mlflow.log_metrics({
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "mean_cv_score": mean_score,
                "std_cv_score": std_score
            })

    # 3) Log BEST model in a clean parent run
    with mlflow.start_run(run_name="Best Logistic Regression Model"):

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)

        # ---- FIX: DagsHub-supported model logging ----
        model_path = "best_model.pkl"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        print("\nBEST MODEL LOGGED:")
        print(f"Best Params: {best_params}")
        print(f"Best F1 Score: {best_f1:.4f}")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)