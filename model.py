import os
import re
import json

import joblib
import numpy as np
import pandas as pd
from loguru import logger as log
from collections import Counter
from typing import Dict, List, Tuple, Type, Union, Any
from scipy.sparse import spmatrix

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Dropout
from keras.api.preprocessing.sequence import pad_sequences
from keras_hub.api.tokenizers import Tokenizer

from data import save_data_npbin


class Model:
    """
    Class for using pre-trained models.
    Can either be instanted with an already loaded model, or load one from a given file
    """

    def __init__(self, loaded_model: Type = None, model_file: str = None) -> None:
        self.model: Type = loaded_model
        self.model_file: str = model_file
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.trained: bool = False

        if self.model_file and not self.model:
            self.load_model()

    def load_model(self) -> None:
        """
        Loads a trained model and TF-IDF vectorizer from a file.
        Raises an error if the file does not exist.
        """
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"❌ Model file '{self.model_file}' not found.")

        # Load model and vectorizer
        try:
            model_data = joblib.load(self.model_file)
            self.model = model_data["model"]
            self.vectorizer = model_data["vectorizer"]
            self.trained = True

            log.info(f"✅ Model loaded successfully from {self.model_file}")
        except Exception as e:
            log.error(f"❌ {e}")
            exit(1)

    def predict(self, text: Union[str, List[str], np.ndarray[Any]]) -> np.ndarray:
        """Interpret the given text and try to determine whether it
        possibly matches our author"""
        if not self.model:
            raise RuntimeError("❌ Model must be loaded before making predictions.")

        text = [text] if isinstance(text, str) else text
        X_transformed = self.vectorizer.transform(text).toarray()
        return self.model.predict(X_transformed)


def load_trained_model(model_path: str, tokenizer_path: str = None):
    """
    Loads a trained ML model and optional tokenizer.

    :param model_path: Path to the saved model file (.pkl for ML models, .h5 for deep learning models).
    :param tokenizer_path: Path to the tokenizer file (only for deep learning models).
    :return: Tuple (model, tokenizer or None)
    """

    # Check if the model is a deep learning model (.h5) or a traditional ML model (.pkl)
    if model_path.endswith(".h5"):
        model = load_model(model_path)
        tokenizer = None

        # Load tokenizer if provided
        if tokenizer_path:
            if not os.path.exists(tokenizer_path):
                log.error(f"❌ {tokenizer_path} not found")
                raise FileNotFoundError(f"{tokenizer_path} not found")

            with open(tokenizer_path, "rb") as file:
                tokenizer = joblib.load(file)

        log.info(f"✅ Loaded deep learning model from {model_path}")
        return model, tokenizer

    elif model_path.endswith(".pkl"):
        model = joblib.load(model_path)
        log.info(f"✅ Loaded ML model from {model_path}")
        return model, None  # No tokenizer for traditional ML models

    else:
        raise ValueError(
            "❌ Unsupported model format. Use '.h5' for deep learning or '.pkl' for ML models."
        )


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Trains a Random Forest classifier to detect an author's writing style."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    log.info(f"🧠 Model Accuracy: {accuracy:.2f}")

    return model


def train_lstm(texts: list[str], labels: list[str]) -> tuple[Sequential, Any]:
    """Trains an LSTM deep learning model for author detection."""
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=100)
    y = np.array(labels.map({"Author": 1, "Other": 0}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model: Sequential = Sequential(
        layers=[
            Embedding(input_dim=5000, output_dim=128, input_length=100),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    log.info("Compiling model...")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    log.info("Training...")
    model.fit(
        X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test)
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    log.info(f"🧠 LSTM Accuracy: {accuracy:.2f}")

    return model, tokenizer


class ModelTrainer:
    """
    Picks and trains a Model based on a given data set to
    recognize an author's writing style.
    """

    def __init__(self) -> None:
        """
        Initializes the training pipeline.
        """
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english"
        )
        self.trained_model: Any = None

    def load_and_prepare_data(
        self, training_data: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Loads, analyzes, and preprocesses the dataset.

        :param file_path: Path to the CSV dataset.
        :return: Tuple (feature matrix, labels, dataset information).
        """
        df: pd.DataFrame = pd.read_csv(training_data)
        if not {"author", "post_content"}.issubset(df.columns):
            raise ValueError("❌ CSV must contain 'author' and 'post_content' columns.")

        # Extract TF-IDF features
        X: np.ndarray = self.vectorizer.fit_transform(df["post_content"]).toarray()
        y: np.ndarray = df["author"].values

        # Analyze dataset characteristics
        dataset_info: Dict[str, Any] = self.analyze_dataset(y, X)
        return X, y, dataset_info

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans and normalizes text by removing unnecessary characters."""
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = re.sub(r"[^a-zA-Z0-9.,!?;'\"]", " ", text)  # Remove special characters
        return text.strip().lower()

    def extract_features(self, texts: pd.Series) -> spmatrix:
        """Extracts TF-IDF features from text."""
        return self.vectorizer.fit_transform(texts)

    def analyze_dataset(self, y: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes dataset characteristics for model selection.

        :param y: Labels (author names).
        :param X: Feature matrix.
        :return: Dictionary containing dataset metadata.
        """
        num_samples = len(y)
        num_classes = len(set(y))
        class_distribution = Counter(y)
        class_balance = min(class_distribution.values()) / max(
            class_distribution.values()
        )

        dataset_info: Dict[str, Any] = {
            "num_samples": num_samples,
            "num_classes": num_classes,
            "vocab_size": X.shape[1],
            "class_balance": class_balance,
        }

        log.info(f"📊 Dataset Analysis: {json.dumps(dataset_info, indent=2)}")
        return dataset_info

    def select_best_model(
        self, dataset_info: Dict[str, Any]
    ) -> MultinomialNB | LogisticRegression | RandomForestClassifier | SVC:
        """
        Automatically selects the best model based on dataset characteristics.

        :param dataset_info: Metadata of dataset.
        """
        num_samples: int = dataset_info["num_samples"]
        class_balance: float = dataset_info["class_balance"]

        if num_samples < 1000:
            log.info("🔹 Small dataset detected. Choosing Naive Bayes for efficiency.")
            return MultinomialNB()
        elif class_balance < 0.5:
            log.info(
                "🔹 Imbalanced dataset detected. Choosing Logistic Regression for better generalization."
            )
            return LogisticRegression()
        elif num_samples > 5000:
            log.info(
                "🔹 Large dataset detected. Choosing Random Forest for scalability and interpretability."
            )
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            log.info(
                "🔹 Balanced and sufficient data detected. Choosing SVM for accuracy."
            )
            return SVC()

    def train(
        self,
        model: MultinomialNB | LogisticRegression | RandomForestClassifier | SVC,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Trains the selected model and evaluates performance.

        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        :param X_val: Validation feature matrix.
        :param y_val: Validation labels.
        """
        if not model:
            raise ValueError("❌ No model selected. Run 'select_best_model()' first.")

        self.trained_model = model.fit(X_train, y_train)
        accuracy = model.score(X_val, y_val)
        log.info(f"🧠 Model accuracy: {accuracy:.2f}")

    def save_model(self, model_path: str = "model.pkl") -> None:
        """
        Saves the trained model and vectorizer.

        :param model_path: Path to save the model.
        """
        if not self.trained_model:
            return

        joblib.dump(
            {"model": self.trained_model, "vectorizer": self.vectorizer}, model_path
        )
        log.info(f"✅ Model saved to {model_path}")


def run_model() -> None:
    """Example usage of running a pre-trained model using the Model class"""
    try:
        model = Model(model_file="model.pkl")
        sample_text = "This is an example post discussing AI."
        prediction = model.predict(sample_text)
        log.info(f"Prediction: {prediction[0]}")
    except FileNotFoundError as e:
        log.error(f"❌ Unexpected error: {e}")


def train_model() -> None:
    """
    Example usage of training a new model.
    Loads CSV data, automatically selects a model, trains it, and saves results.
    """
    try:
        file_path = "forum_posts.csv"
        model_trainer = ModelTrainer()

        X, y, dataset_info = model_trainer.load_and_prepare_data(file_path)
        best_model = model_trainer.select_best_model(dataset_info)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        save_data_npbin(X_train, y_train, X_val, y_val)
        model_trainer.train(best_model, X_train, y_train, X_val, y_val)
        model_trainer.save_model()

    except FileNotFoundError:
        log.error(f"❌ {file_path} file not found.")
    except Exception as e:
        log.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    train_model()
