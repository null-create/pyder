import re
from typing import Dict
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from loguru import logger as log

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


# Preprocess text: remove special characters and tokenize
def preprocess_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"[^a-zA-Z0-9.,!?;'\"]", " ", text)  # Keep key characters
    return text.lower().strip()


# Define stopwords and intensifiers
STOPWORDS = set(stopwords.words("english"))
INTENSIFIERS = {"very", "extremely", "incredibly", "highly", "remarkably", "super"}


def flesch_kincaid_readability(text: str) -> float:
    """Computes Flesch-Kincaid readability score (higher = easier to read).
    See: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests"""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    syllables = sum(len(re.findall(r"[aeiouyAEIOUY]+", word)) for word in words)
    num_words = len(words)
    num_sentences = len(sentences)

    if num_words == 0 or num_sentences == 0:
        return 0

    return (
        206.835
        - (1.015 * (num_words / num_sentences))
        - (84.6 * (syllables / num_words))
    )


def extract_features(text: str) -> Dict[str, float]:
    """Extracts a comprehensive set of features from the given text."""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    num_words = len(words)
    num_sentences = len(sentences)

    # Basic statistics
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    avg_sentence_length = (
        np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0
    )
    lexical_diversity = len(set(words)) / num_words if num_words else 0
    punctuation_ratio = (
        sum(1 for char in text if char in ".,!?;") / len(text) if text else 0
    )
    passive_voice_count = len(
        re.findall(r"\b(is|was|were|been|being) [a-zA-Z]+ed\b", text, re.IGNORECASE)
    )

    # Word choice
    word_freq = Counter(words)
    most_common_words = [
        word for word, _ in word_freq.most_common(5)
    ]  # Top 5 frequent words
    unique_words = len(set(words))

    # Sentence phrasing: Identify common starting words
    sentence_starters = [
        sentence.split()[0].lower() for sentence in sentences if sentence
    ]
    common_sentence_starters = [
        word for word, _ in Counter(sentence_starters).most_common(3)
    ]  # Top 3 starters

    # Readability score
    readability = flesch_kincaid_readability(text)

    # Stopwords usage
    stopword_ratio = (
        sum(1 for word in words if word.lower() in STOPWORDS) / num_words
        if num_words
        else 0
    )

    # Part-of-speech (POS) tags
    pos_tags = nltk.pos_tag(words)
    num_nouns = sum(1 for _, tag in pos_tags if tag.startswith("NN"))
    num_verbs = sum(1 for _, tag in pos_tags if tag.startswith("VB"))
    num_adjectives = sum(1 for _, tag in pos_tags if tag.startswith("JJ"))

    # Emphasis usage
    emphasis_count = sum(1 for word in words if word.lower() in INTENSIFIERS)

    return {
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "punctuation_ratio": punctuation_ratio,
        "passive_voice_count": passive_voice_count,
        "unique_word_count": unique_words,
        "top_5_words": ", ".join(most_common_words),
        "common_sentence_starters": ", ".join(common_sentence_starters),
        "readability_score": readability,
        "stopword_ratio": stopword_ratio,
        "noun_ratio": num_nouns / num_words if num_words else 0,
        "verb_ratio": num_verbs / num_words if num_words else 0,
        "adjective_ratio": num_adjectives / num_words if num_words else 0,
        "emphasis_word_count": emphasis_count,
    }


# Load dataset: A CSV file containing text samples labeled as "Author" or "Other"
def load_training_data(file_path: str):
    df = pd.read_csv(file_path)
    df["text"] = df["text"].apply(preprocess_text)
    df["features"] = df["text"].apply(extract_features)
    feature_df = pd.DataFrame(df["features"].tolist())
    return feature_df, df["label"]


# Train the model
def train_author_style_model(file_path: str, model_path: str = "author_model.pkl"):
    X, y = load_training_data(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    log.info(f"Model Accuracy: {accuracy:.2f}")

    joblib.dump(model, model_path)
    print(f"Model saved as {model_path}")


# Load trained model and test new text
def predict_author(text: str, model_path: str = "author_model.pkl"):
    model = joblib.load(model_path)
    text = preprocess_text(text)
    features = extract_features(text)
    input_df = pd.DataFrame([features])

    prediction = model.predict(input_df)
    return (
        "Likely Written by the Author"
        if prediction[0] == "Author"
        else "Not Likely Written by the Author"
    )


if __name__ == "__main__":
    # Example usage
    train_author_style_model("author_data.csv")  # Train model on labeled dataset

    sample_text = """
    According to my research, cybersecurity threats evolve daily.
    AI is revolutionizing how we detect and mitigate risks in this field.
    """
    print(predict_author(sample_text))
