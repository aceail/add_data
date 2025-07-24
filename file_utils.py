import logging
import zipfile
from pathlib import Path
from typing import Optional


def extract_zip(zip_path: str | Path) -> Optional[Path]:
    """Extract a zip file and return the extraction directory."""

    path = Path(zip_path)
    if not path.exists():
        logging.error("File not found: %s", zip_path)
        return None

    if path.suffix.lower() != ".zip":
        logging.error("Unsupported extension: %s", path.suffix)
        return None

    output_dir = path.parent / path.stem
    if output_dir.exists():
        logging.info("Skip extraction, already exists: %s", output_dir)
        return output_dir

    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(output_dir)
        logging.info("Extracted %s to %s", path, output_dir)
        return output_dir
    except Exception as exc:  # pragma: no cover - just logging
        logging.exception("Failed to extract %s: %s", path, exc)
        return None
