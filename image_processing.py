import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

from CT_Preprocessing import PreprocessingCTImage as pp
from utils import draw_overlay_cti


def normalize_to_8bit(array: np.ndarray) -> np.ndarray:
    """Normalize an array to 0-255 range and return ``uint8`` array."""

    array = array.astype(np.float32)
    array -= array.min()
    array /= array.max() + 1e-8
    array *= 255
    return array.astype(np.uint8)


def dicom_to_png(path: Path, save_path: Path) -> None:
    """Convert a DICOM file to a PNG image."""

    ds = pydicom.dcmread(str(path))
    arr = normalize_to_8bit(ds.pixel_array)
    Image.fromarray(arr).save(save_path)


def non_mask(dcm_path: Path, dir_path: Path) -> None:
    """Save a non-masked overlay image for a given DICOM."""

    image_sitk = pp.read_dicom(str(dcm_path))
    image_np = pp.to_numpy(image_sitk)
    total_region = np.zeros_like(image_np)
    draw_fig, _, _, _ = draw_overlay_cti(image_np, total_region)
    draw_fig = draw_fig.astype(np.uint8)[:, :, :3]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(draw_fig)
    plt.axis("off")
    plt.tight_layout()

    non_mask_path = Path(dir_path) / "Non_mask"
    os.makedirs(non_mask_path, exist_ok=True)
    plt.savefig(non_mask_path / "non_mask.png", bbox_inches="tight", pad_inches=0, facecolor="black")


def convert_all_dicom_to_png(grouped_df: pd.DataFrame, output_dir: Path) -> None:
    """Convert all DICOMs referenced in ``grouped_df`` into PNG images."""

    os.makedirs(output_dir, exist_ok=True)
    for _, row in tqdm(grouped_df.iterrows(), total=len(grouped_df)):
        patient_id = row["patientID"]
        study_desc = row["StudyDesc"].split(". ")[-1]
        series_desc = row["SeriesDesc"].split(". ")[-1]
        dir_path = output_dir / f"{patient_id}_{study_desc}_{series_desc}"
        os.makedirs(dir_path, exist_ok=True)

        if "NCCT" in row["modality"]:
            non_mask(row["file"], dir_path)
        else:
            for idx, f in enumerate(row["JLK_AI_full_dcm"]):
                sub_path = dir_path / row["modality"]
                os.makedirs(sub_path, exist_ok=True)
                fname = Path(f).stem + f"_{idx}.png"
                save_path = sub_path / fname
                dicom_to_png(Path(f), save_path)


def process_row(row: pd.Series, output_dir: Path) -> None:
    """Process a single dataframe row for PNG conversion."""

    patient_id = row["patientID"]
    study_desc = row["StudyDesc"].split(". ")[-1]
    series_desc = row["SeriesDesc"].split(". ")[-1]
    dir_path = output_dir / f"{patient_id}_{study_desc}_{series_desc}"
    os.makedirs(dir_path, exist_ok=True)

    if "NCCT" in row["modality"]:
        non_mask(row["file"], dir_path)
    else:
        for idx, f in enumerate(row["JLK_AI_full_dcm"]):
            sub_path = dir_path / row["modality"]
            os.makedirs(sub_path, exist_ok=True)
            fname = Path(f).stem + f"_{idx}.png"
            save_path = sub_path / fname
            dicom_to_png(Path(f), save_path)


def convert_all_dicom_to_png_parallel(grouped_df: pd.DataFrame, output_dir: Path, max_workers: int = 8) -> None:
    """Convert DICOMs to PNGs using multi-threading."""

    os.makedirs(output_dir, exist_ok=True)
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in grouped_df.iterrows():
            futures.append(executor.submit(process_row, row, output_dir))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
