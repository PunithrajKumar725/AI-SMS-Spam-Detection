
import os
import zipfile
import urllib.request

import pandas as pd
import joblib

from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# Dataset URL
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)


def load_dataset():
    print("Downloading SMS Spam Collection dataset...")

    response = urllib.request.urlopen(DATASET_URL)
    dataset_zip = zipfile.ZipFile(BytesIO(response.read()))

    with dataset_zip.open("SMSSpamCollection") as file:
        df = pd.read_csv(
            file,
            sep="\t",
            names=["label", "message"]
        )

    return df


def train_model():

    # Load dataset
    df = load_dataset()

    print("\nDataset loaded successfully!")
    print(df.head())

    # Convert labels
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # Create ML pipeline
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    # Train model
    print("\nTraining Logistic Regression model...")
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Evaluation")
    print("-----------------------")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Spam"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save model
    joblib.dump(model, "spam_model.pkl")

    print("\nModel saved successfully as spam_model.pkl")


if __name__ == "__main__":
    train_model()
