import logging
import re
from itertools import chain
from pathlib import Path
from typing import List

import pandas as pd


def get_jlk_summary_dirs(path: Path) -> List[str]:
    """Return child summary directories under the given path."""

    if not re.search(r"ICH|CTI|CTL|WMHC|CVL", str(path)):
        return []

    try:
        return [
            child.name
            for child in path.iterdir()
            if child.is_dir() and "summary" in child.name.lower()
        ]
    except Exception as exc:  # pragma: no cover - just logging
        logging.error("Error scanning %s: %s", path, exc)
        return []


def join_jlk_full_paths(row: pd.Series) -> List[Path]:
    """Join `row['file']` with summary subdirectories."""

    if not isinstance(row["JLK_AI"], list):
        return []
    return [row["file"] / sub for sub in row["JLK_AI"]]


def _collection_path(extract_path: Path) -> pd.DataFrame:
    """Collect metadata from extracted image directories."""

    extract_path = Path(extract_path)
    paths = list(extract_path.glob("*/*/*/*/*"))

    df = pd.DataFrame(paths, columns=["file"])
    df["patientID"] = df["file"].apply(lambda x: x.parts[-4])
    df["StudyDesc"] = df["file"].apply(lambda x: x.parts[-3])
    df["SeriesDesc"] = df["file"].apply(lambda x: x.parts[-2])
    df["modality"] = df["file"].apply(lambda x: x.parts[-1])
    df["JLK_AI"] = df["file"].apply(get_jlk_summary_dirs)
    df["JLK_AI_full"] = df.apply(join_jlk_full_paths, axis=1)
    df["JLK_AI_full_dcm"] = df["JLK_AI_full"].apply(
        lambda folders: list(chain.from_iterable(folder.rglob("*.dcm") for folder in folders))
        if isinstance(folders, list)
        else []
    )
    return df
