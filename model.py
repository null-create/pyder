import json
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


class Model:
    def __init__(self, model_type: Type) -> None:
        self.model = model_type()
        self.vectorizer = TfidfVectorizer()
        self.trained = False
        self.metrics = {}
        self.history = []

    def prepare_data(
        self, data: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        texts: List[str] = []
        labels: List[int] = []
        self.label_map: Dict[str, int] = {
            author: idx for idx, author in enumerate(data.keys())
        }

        for author, samples in data.items():
            texts.extend(samples)
            labels.extend([self.label_map[author]] * len(samples))

        X: np.ndarray = self.vectorizer.fit_transform(texts).toarray()
        y: np.ndarray = np.array(labels)

        return X, y

    def is_trained(self) -> bool:
        return self.trained

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        self.metrics = classification_report(y_val, y_pred, output_dict=True)
        self.history.append({"accuracy": accuracy, "metrics": self.metrics})
        self.is_trained = True

        log.info(f"Validation Accuracy: {accuracy:.4f}")
        log.info("Classification Report:")
        log.info(classification_report(y_val, y_pred))

        self.save_model()

    def save_model(self, filename: str = "model.pkl") -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def save_vectorizer(self, filename: str = "vectorizer.json") -> None:
        with open(filename, "w") as f:
            json.dump(self.vectorizer.vocabulary_, f)

    def load_model(self, filename: str = "model.pkl") -> None:
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        if not self.trained:
            raise ValueError(
                "Model must be trained or loaded before making predictions."
            )

        text = [text] if isinstance(text, str) else text
        X_transformed = self.vectorizer.transform(text).toarray()
        return self.model.predict(X_transformed)


def load_data(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, "r") as f:
        return json.load(f)


def save_data(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> None:
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)


def main() -> None:
    data: Dict[str, List[str]] = load_data("data.json")
    models = {
        "Naive Bayes": MultinomialNB,
        "Logistic Regression": LogisticRegression,
        "Support Vector Machine": SVC,
    }

    for model_name, model_type in models.items():
        print(f"Training {model_name}...")
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
