import json
import joblib
import numpy as np
import pickle
from loguru import logger as log

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Type, Union

from keras.api.models import load_model

from data import save_data
from analyze import extract_features


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

    def save_model(self) -> None:
        with open(self.file_name, "wb") as f:
            pickle.dump(self.model, f)

        log.info(f"✅ Model ({self.file_name}) saved")

    def save_vectorizer(self, filename: str = "vectorizer.json") -> None:
        with open(filename, "w") as f:
            json.dump(self.vectorizer.vocabulary_, f)

        log.info("✅ vectorizer saved")

    def load_model(self) -> None:
        with open(self.file_name, "rb") as f:
            self.model = pickle.load(f)

        self.trained = True
        log.info("✅ Model loaded")

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        """Interpret the given text and try to determine whether it
        possibly matches our author"""
        if not self.trained or not self.model:
            raise ValueError(
                "❌ Model must be trained or loaded before making predictions."
            )

        text = [text] if isinstance(text, str) else text
        X_transformed = self.vectorizer.transform(text).toarray()
        return self.model.predict(X_transformed)


def load_data(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, "r") as f:
        return json.load(f)


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


def main() -> None:
    data: Dict[str, List[str]] = load_data("data.json")
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
