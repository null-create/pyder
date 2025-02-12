import os
import json
import joblib
import numpy as np
import pickle
from typing import Dict, List, Tuple, Type, Union, Any

from loguru import logger as log

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, Dropout
from keras.api.preprocessing.sequence import pad_sequences
from keras_hub.api.tokenizers import Tokenizer

from data import save_data


class Model:
    def __init__(self, model_type: Type, file_name: str = None) -> None:
        self.model: Type = model_type()
        self.file_name: str = file_name or "author_model.pkl"
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.trained: bool = False
        self.metrics: str | Dict = None
        self.history: List = []

    def is_trained(self) -> bool:
        return self.trained

    def prepare_data(
        self, data: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Reformats incoming data to X, y np.ndarrays for use by the model.

        Call before running train."""
        texts: List[str] = []
        labels: List[int] = []
        label_map: Dict[str, int] = {
            author: idx for idx, author in enumerate(data.keys())
        }

        for author, samples in data.items():
            texts.extend(samples)
            labels.extend([label_map[author]] * len(samples))

        X: np.ndarray = self.vectorizer.fit_transform(texts).toarray()
        y: np.ndarray = np.array(labels)

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Train the model using the given inputs"""
        if not self.trained:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)

            accuracy = accuracy_score(y_val, y_pred)
            self.metrics = classification_report(y_val, y_pred, output_dict=True)
            self.history.append({"accuracy": accuracy, "metrics": self.metrics})
            self.trained = True

            log.info(f"🎯 Validation Accuracy: {accuracy:.4f}")
            log.info("📊 Classification Report:")
            log.info(classification_report(y_val, y_pred))
            self.save_model()
        else:
            log.warning(f"⚠️ Model already trained")

    def save_model(self) -> None:
        with open(self.file_name, "wb") as f:
            pickle.dump(self.model, f)

        log.info(f"✅ Model ({self.file_name}) saved")

    def save_vectorizer(self, filename: str = "vectorizer.json") -> None:
        with open(filename, "w") as f:
            json.dump(self.vectorizer.vocabulary_, f)

        log.info("✅ Vectorizer saved")

    def load_model(self) -> None:
        with open(self.file_name, "rb") as f:
            self.model = pickle.load(f)

        self.trained = True
        log.info("✅ Model loaded")

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        """Interpret the given text and try to determine whether it
        possibly matches our author"""
        if not self.trained or not self.model:
            raise LookupError(
                "❌ Model must be trained or loaded before making predictions."
            )

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
                log.error(f"{tokenizer_path} not found")
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
    print(f"🧠 Model Accuracy: {accuracy:.2f}")

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
        [
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
    print(f"🧠 LSTM Accuracy: {accuracy:.2f}")

    return model, tokenizer


def main() -> None:
    try:
        with open("data.json", "r") as f:
            data: Dict[str, List[str]] = json.load(f)
    except (FileNotFoundError("data.json file not found"), Exception) as e:
        log.error(e)
        return

    models = {
        "Naive Bayes": MultinomialNB,
        "Logistic Regression": LogisticRegression,
        "Support Vector Machine": SVC,
    }

    for model_name, model_type in models.items():
        log.info(f"Training {model_name}...")
        model = Model(model_type)
        X, y = model.prepare_data(data)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        save_data(X_train, y_train, X_val, y_val)
        model.train(X_train, y_train, X_val, y_val)
        model.save_vectorizer()


if __name__ == "__main__":
    main()
