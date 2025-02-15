import os
import re
import csv
import json
import gzip
from typing import Dict, Any, List

import numpy as np
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


def get_keywords() -> list[str]:
    file = "keywords.txt"
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ {file} file not found")

    with open(file, "r") as f:
        keywords = f.read().splitlines()

    return keywords


def preprocess_text(text: str) -> str:
    """Cleans and normalizes text for training."""
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = re.sub(r"[^a-zA-Z0-9.,!?;'\"]", " ", text)  # Remove unnecessary characters
    return text.lower().strip()


def save_author_data_for_training(
    scraped_data: List[Dict[str, str]],
    author_name: str,
    output_file: str = None,
) -> None:
    """
    Processes scraped data into a CSV file for training.

    - `scraped_data`: List of dicts with "text" and optionally "author".
    - `author_name`: Name of the author to label known writings.
    - `output_file`: File to save processed data.
    """
    if not output_file:
        output_file = "author_data.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])  # CSV header

        for entry in scraped_data:
            text = preprocess_text(entry.get("text", ""))
            if not text:
                continue  # Skip empty text

            label = (
                "Author"
                if entry.get("author", "").lower() == author_name.lower()
                else "Other"
            )
            writer.writerow([text, label])

    log.info(f"✅ Processed data saved to {output_file}")


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
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> None:
    """Save training data as numpy binary files"""
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)


def save_data_to_json(
    data: Dict[str, Any], directory: str, filename: str, compress: bool = False
) -> str:
    """
    Saves extracted data as a JSON file in the specified directory.

    Args:
        data (Dict[str, Any]): Extracted data to save.
        directory (str): Directory where the file should be saved.
        filename (str): Name of the JSON file.
        compress (bool): If True, saves as a .json.gz compressed file.

    Returns:
        str: Path to the saved JSON file.
    """
    ensure_directory_exists(directory)

    # Adjust filename extension for compression
    if compress and not filename.endswith(".gz"):
        filename += ".gz"

    unique_filename = generate_unique_filename(directory, filename)
    file_path = os.path.join(directory, unique_filename)

    try:
        if compress:
            with gzip.open(file_path, "wt", encoding="utf-8") as gz_file:
                json.dump(data, gz_file, indent=2, ensure_ascii=False)
        else:
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)

        log.info(f"✅ Data successfully saved to: {file_path}")
        return file_path

    except IOError as e:
        log.error(f"❌ IOError while saving file: {e}")
        return ""

    except Exception as e:
        log.error(f"❌ Exception while saving file: {e}")


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

        if self.output_format not in ["json", "csv"]:
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
            # Flatten nested lists/dictionaries for CSV format
            flat_data = {
                k: (",".join(v) if isinstance(v, list) else v) for k, v in data.items()
            }
            file_exists = os.path.isfile(self.output_file)

            with open(self.output_file, "a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=flat_data.keys())

                if not file_exists:  # Write header only if file is new
                    writer.writeheader()
                writer.writerow(flat_data)
        except Exception as e:
            log.error(f"❌ Error saving CSV: {e}")
