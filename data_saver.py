import os
import json
import gzip
from typing import Dict, Any

from loguru import logger as log


def ensure_directory_exists(directory: str) -> None:
    """Creates the directory if it does not exist."""
    return os.makedirs(directory, exist_ok=True)


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
            json.dump(data, json_file, indent=4, ensure_ascii=False)

        log.info(f"✅ File successfully decompressed to: {output_path}")
        return output_path

    except (IOError, json.JSONDecodeError) as e:
        log.error(f"❌ Error decompressing file: {e}")
        return ""


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
                json.dump(data, gz_file, indent=4, ensure_ascii=False)
        else:
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)

        log.info(f"✅ Data successfully saved to: {file_path}")
        return file_path

    except IOError as e:
        log.error(f"❌ Error saving file: {e}")
        return ""
