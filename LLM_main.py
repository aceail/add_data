"""Utility script for handling JLK brain CT data and generating reports."""

from __future__ import annotations

from pathlib import Path

from dataset_utils import _collection_path
from file_utils import extract_zip
from image_processing import convert_all_dicom_to_png_parallel
from openai_utils import JLK_ICH


def main() -> None:
    """Example main routine."""

    zip_path = Path("path/to/archive.zip")
    extract_path = extract_zip(zip_path)
    if not extract_path:
        return

    df = _collection_path(extract_path)
    output_dir = Path("output_images")
    convert_all_dicom_to_png_parallel(df, output_dir)

    ich_result = []
    for item in output_dir.iterdir():
        if item.is_dir():
            non_mask_dir = item / "Non_mask"
            if non_mask_dir.exists() and any(non_mask_dir.iterdir()):
                ich_dir = next((p for p in item.iterdir() if p.is_dir() and "ICH" in p.name), None)
                ich_result.append(JLK_ICH(non_mask_dir, ich_dir))

    print(ich_result)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
