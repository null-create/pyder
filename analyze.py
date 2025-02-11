import re
from collections import Counter

import numpy as np
from loguru import logger as log

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


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
