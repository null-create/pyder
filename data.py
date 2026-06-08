import os
import re
import csv
import json
import gzip
from typing import Dict, Any

from loguru import logger as log
from pydantic import BaseModel

SEED_DATA_FILE = "seed-data.json"

# Flat fields used for CSV export of page results
PAGE_COLUMNS = ["url", "title", "content", "description", "author", "timestamp"]

# Flat fields used for CSV export of forum-post results
POST_COLUMNS = ["author", "content", "timestamp", "url"]


class SeedData(BaseModel):
    home: str
    urls: list[str]
    keywords: list[str]
    search_depth: int
    outfile: str


def get_starting_data() -> SeedData:
    """
    Opens and returns the contents of seed-data.json file.
    This file is used for the web crawler's discovery mode.

    Expects something like:
    ```
    {
        "home": "",
        "urls": [],
        "keywords": [],
        "search-depth": 0,
        "outfile": ""
    }
    ```
    """
    seed_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), SEED_DATA_FILE)
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"seed file not found at {seed_file}")

    with open(seed_file, "r") as f:
        sd: dict = json.load(fp=f)

    try:
        seed_data = SeedData(**sd)
    except Exception as e:
        raise ValueError(f"Invalid seed-data.json format: {e}")

    return seed_data


def generate_unique_filename(directory: str, filename: str) -> str:
    """
    Generates a unique filename if the file already exists in the directory.
    Example: 'data.json' -> 'data_1.json', 'data_2.json', etc.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1

    return unique_filename


def preprocess_text(text: str) -> str:
    """Cleans and normalizes text for training (aggressive)."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;'\"]", " ", text)
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
        log.error("File must be a .json.gz compressed file")
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

        log.info(f"File successfully decompressed to: {output_path}")
        return output_path

    except (IOError, json.JSONDecodeError) as e:
        log.error(f"Error decompressing file: {e}")
        return ""


class DataHandler:
    def __init__(
        self,
        compress: bool = False,
        output_format: str = "json",
        output_file_name: str = "output",
    ) -> None:
        """Manages caching and storing extracted data to JSON or CSV format."""
        self.compress = compress
        self.output = []

        if output_format not in ("json", "csv"):
            raise ValueError("Invalid format. Use 'json' or 'csv'.")

        self.output_format = output_format.lower()
        self.output_file = f"{output_file_name}.{self.output_format}"

    def store(self, data: Dict[str, Any]) -> None:
        """Cache a single result dict for later export."""
        self.output.append(data)

    def dump(self) -> None:
        """Write all cached data to file and clear the cache."""
        if not self.output:
            log.warning("No data to export")
            return
        self.export(self.output)
        self.output.clear()

    def export(self, data: Dict[str, Any] | list[Dict[str, Any]]) -> None:
        """Save extracted data to a file in the configured format."""
        if self.output_format == "json":
            self.save_json(data)
        elif self.output_format == "csv":
            self.save_csv(data)

    def save_json(self, data: Dict[str, Any] | list[Dict[str, Any]]) -> None:
        """Write data as JSON (overwrites file each call)."""
        try:
            out_path = self.output_file + ".gz" if self.compress else self.output_file
            opener = gzip.open if self.compress else open
            mode = "wt" if self.compress else "w"

            with opener(out_path, mode, encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            log.error(f"Error saving JSON: {e}")

    def save_csv(self, data: list[Dict[str, Any]]) -> None:
        """Write data as CSV (appends rows).  Columns are derived from the
        union of keys across all rows; nested values are JSON-encoded."""
        if not data:
            return

        try:
            file_exists = os.path.isfile(self.output_file)

            # Union of all keys, preserving insertion order
            fieldnames = list(dict.fromkeys(k for d in data for k in d))

            with open(self.output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()

                for row in data:
                    flat = {}
                    for k, v in row.items():
                        if not isinstance(v, (str, int, float, bool)):
                            flat[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            flat[k] = v
                    writer.writerow(flat)

        except Exception as e:
            log.error(f"Error saving CSV: {e}")
