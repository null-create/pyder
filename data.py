import os
import re
import csv
import json
import gzip
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger as log


def ensure_directory_exists(directory: str) -> None:
    """Creates the directory if it does not exist."""
    return os.makedirs(directory, exist_ok=True)


def get_model_and_tokenizer_filenames() -> tuple[str, str]:
    """Retrieves model and tokenizer filenames from the environment

    MODEL_FILE and TOKENIZER_FILE must be set!"""
    model_file = os.getenv("MODEL_FILE")
    tokenizer_file = os.getenv("TOKENIZER_FILE")
    if not model_file or not tokenizer_file:
        raise ValueError(
            f"❌ Missing model file (model={model_file}) or tokenizer file (tokenizer={tokenizer_file})"
        )

    return model_file, tokenizer_file


def generate_unique_filename(directory: str, filename: str) -> str:
    """
    Generates a unique filename if the file already exists in the directory.
    Example: 'data.json' → 'data_1.json', 'data_2.json', etc.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1

    return unique_filename


def get_starting_data() -> dict:
    file = "seed-data.json"
    if not os.path.exits(file):
        raise FileNotFoundError(f"❌ {file} file not found")

    with open(file, "r") as f:
        seed_data = json.load(fp=f)

    return seed_data


def preprocess_text(text: str) -> str:
    """Cleans and normalizes text for training."""
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"[^a-zA-Z0-9.,!?;'\"]", " ", text)  # Remove unnecessary characters
    return text.lower().strip()


def decompress_json_gz(file_path: str) -> str:
    """
    Decompresses a .json.gz file back into a normal .json file.

    Args:
        file_path (str): Path to the compressed .json.gz file.

    Returns:
        str: Path to the decompressed JSON file.
    """
    if not file_path.endswith(".json.gz"):
        log.error("❌ Error: File must be a .json.gz compressed file")
        return ""

    directory = os.path.dirname(file_path)
    base_filename = os.path.basename(file_path).replace(".json.gz", ".json")
    output_filename = generate_unique_filename(directory, base_filename)
    output_path = os.path.join(directory, output_filename)

    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:
            data = json.load(gz_file)

        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)

        log.info(f"✅ File successfully decompressed to: {output_path}")
        return output_path

    except (IOError, json.JSONDecodeError) as e:
        log.error(f"❌ Error decompressing file: {e}")
        return ""


def save_data_npbin(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    file_name: str = "train_data.npbin",
) -> None:
    """
    Saves training and validation data in NumPy binary format.

    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :param X_val: Validation feature matrix.
    :param y_val: Validation labels.
    :param file_name: Path to save the binary file.
    """
    np.savez(file_name, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    log.info(f"✅ Training data saved to {file_name}")


class DataHandler:
    def __init__(
        self,
        compress: bool = False,
        output_format: str = "csv",
        output_file_name: str = "output",
    ) -> None:
        """Manages caching and storing extracted data to JSON or CSV format."""
        self.compress = compress  # only works with json output
        self.output = []  # cached data to be written out

        if output_format not in ["json", "csv"]:
            raise ValueError("❌ Invalid format. Use either 'json' or 'csv'.")

        self.output_format = output_format.lower()
        self.output_file = f"{output_file_name}.{self.output_format}"

    def store(self, data: Dict[str, Any]) -> None:
        """Cache data before writing out."""
        self.output.append(data)

    def dump(self) -> None:
        """Empty cache to specified file"""
        self.export(self.output)
        self.output.clear()

    def export(self, data: Dict[str, Any] | list[Dict[str, Any]]) -> None:
        """Saves extracted data to a file in the specified format."""
        if self.output_format == "json":
            self._save_json(data)
        elif self.output_format == "csv":
            self._save_csv(data)

    def _save_json(self, data: Dict[str, Any] | list[Dict[str, Any]]) -> None:
        """Appends extracted data to a JSON file."""
        try:
            if self.compress:
                with gzip.open(
                    self.output_file + ".gz", "wt", encoding="utf-8"
                ) as gz_file:
                    json.dump(data, gz_file, ensure_ascii=False, indent=2)
            else:
                with open(self.output_file, "a", encoding="utf-8") as file:
                    json.dump(data, file, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"❌ Error saving JSON: {e}")

    def _save_csv(self, data: Dict[str, Any] | list[Dict[str, Any]]) -> None:
        """Appends extracted data to a CSV file."""
        try:
            file_exists = os.path.isfile(self.output_file)

            with open(self.output_file, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                # add initial header if we're creating the file for the first time
                if not file_exists:
                    writer.writerow(
                        [
                            "author",
                            "post_content",
                            "timestamp",
                            "thread_url",
                        ]
                    )

                for post in data:
                    writer.writerow(
                        [
                            post["author"],
                            post["post_content"],
                            post["timestamp"],
                            post["thread_url"],
                        ]
                    )
        except Exception as e:
            log.error(f"❌ Error saving CSV: {e}")
