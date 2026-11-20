
import os
import zipfile
import urllib.request
from io import BytesIO

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ----------------------------------
# 1. Dataset Configuration
# ----------------------------------

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")


# ----------------------------------
# 2. Download and Load Dataset
# ----------------------------------

def load_dataset():

    print("Downloading SMS Spam Collection dataset...")

    response = urllib.request.urlopen(DATASET_URL, timeout=60)

    dataset_zip = zipfile.ZipFile(BytesIO(response.read()))

    with dataset_zip.open("SMSSpamCollection") as file:

        df = pd.read_csv(
            file,
            sep="\t",
            names=["label", "message"]
        )

    print("Dataset loaded successfully!")

    return df


# ----------------------------------
# 3. Prepare Dataset
# ----------------------------------

def prepare_data(df):

    df = df.dropna(subset=["message", "label"])

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df = df.dropna(subset=["label"])

    X = df["message"].astype(str)
    y = df["label"].astype(int)

    return X, y


# ----------------------------------
# 4. Build Machine Learning Pipeline
# ----------------------------------

def create_model():

    model = Pipeline([

        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )),

        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))

    ])

    return model


# ----------------------------------
# 5. Train and Evaluate Model
# ----------------------------------

def train_model():

    df = load_dataset()

    print("\nDataset Information")
    print("-------------------")

    print("Total messages:", len(df))
    print("\nCategory distribution:")
    print(df["label"].value_counts())

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining messages:", len(X_train))
    print("Testing messages:", len(X_test))

    model = create_model()

    print("\nTraining Logistic Regression model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ----------------------------------
    # 6. Model Evaluation
    # ----------------------------------

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print("\nMODEL PERFORMANCE")
    print("======================")

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Spam"],
        zero_division=0
    ))

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print(cm)

    # ----------------------------------
    # 7. Save Evaluation Results
    # ----------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    evaluation = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_size": int(len(X_test)),
        "training_size": int(len(X_train))
    }

    joblib.dump(evaluation, os.path.join(
        MODEL_DIR, "evaluation.pkl"
    ))

    # ----------------------------------
    # 8. Train Final Model
    # ----------------------------------

    print("\nTraining final model on full dataset...")

    final_model = create_model()

    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_PATH)

    print("\nFinal model saved successfully!")
    print("Model location:", MODEL_PATH)


# ----------------------------------
# 9. Main Execution
# ----------------------------------

if __name__ == "__main__":

    train_model()
