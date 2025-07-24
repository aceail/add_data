"""Utility script for handling JLK brain CT data and generating reports."""

from __future__ import annotations

import base64
import logging
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm

from CT_Preprocessing import PreprocessingCTImage as pp
from utils import draw_overlay_cti

try:  # Optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - openai might not be available
    OpenAI = None

if OpenAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:  # pragma: no cover - when openai is unavailable
    client = None

# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------
# 경로 객체로 처리

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

from datetime import datetime


def pil_to_base64(image_path: str) -> str:
    """Return the base64-encoded string of the given image."""

    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class PatientInfo(BaseModel):
    patientId: str = Field(..., description="환자 고유 식별자")
    gender: str = Field(..., description="성별(Male/Female/Other 등)")
    age: int = Field(..., description="나이 (단위: 년)")
    imagingModalities: List[str] = Field(..., description="영상 촬영 모달리티 리스트 (예: ['NCCT', 'CTA', 'CTP'])")
    scanTimestamp: str = Field(..., description="스캔 시각 (ISO8601 형식)")

class ActionableRecommendation(BaseModel):
    primary: str = Field(..., description="첫 번째 주요 권고")
    secondary: str = Field(..., description="추가적 권고")
    justification: str = Field(..., description="권고의 근거")

class Interpretation(BaseModel):
    primaryDiagnosis: str = Field(..., description="주 진단 (영문)")
    pathophysiologicalSynopsis: str = Field(..., description="병태생리학적 해설(영문)")
    aiCritique: str = Field(..., description="AI 분석 결과의 비판적 평가 및 한계")
    actionableRecommendation: ActionableRecommendation = Field(..., description="권고사항")

class ImagingReport(BaseModel):
    patientInfo: PatientInfo
    interpretation: Interpretation
def safe_load_image(image_path: str) -> Optional[str]:
    if not os.path.exists(image_path):
        logging.warning(f"이미지 파일이 존재하지 않음: {image_path}")
        return None
    try:
        return f"data:image/png;base64,{pil_to_base64(image_path)}"
    except Exception as e:
        logging.error(f"이미지 변환 실패: {image_path} - {e}")
        return None
    
def JLK_ICH(mask_path: Path, path: Path) -> Optional[str]:
    try:

        content_blocks = []

        desc_masked = (
            "Summary_0000_0 : JLK ICH는 비조영 CT에서 모든 유형의 뇌출혈을 탐지하는 AI 알고리즘입니다. 뇌출혈 의심 영역은 붉은색 마스크로 표시되며, 환자 단위의 뇌출혈 확률값과 전체 뇌영역에서의 뇌출혈 부피 정보가 함께 제공됩니다."
        )
        img_masked = safe_load_image(str(path / "Summary_0000_0.png"))
        if img_masked:
            content_blocks.append({"type": "input_text", "text": desc_masked})
            content_blocks.append({"type": "input_image", "image_url": img_masked})

        desc_nomask = (
            "Summary_0000_0_non_mask : Summary_0000_0과 동일한 영상이며, 붉은색 마스크를 제거한 버전입니다. (정확한 비교를 위해 mask만 제외되었습니다.)"
        )
        img_nomask = safe_load_image(str(mask_path / "non_mask.png"))
        if img_nomask:
            content_blocks.append({"type": "input_text", "text": desc_nomask})
            content_blocks.append({"type": "input_image", "image_url": img_nomask})

        if not content_blocks:
            logging.error("유효한 이미지가 없어 요청을 수행할 수 없습니다.")
            return None

        if client is None:
            logging.error("OpenAI client not available")
            return None

        response = client.responses.parse(
            model="gpt-4.1",
            prompt={
                "id": "pmpt_687ee31179fc819091e852e25eaca6c20399e215f5db1fab",
                "version": "2"
            },
            input=[{"role": "user", "content": content_blocks}],
            reasoning={},
            max_output_tokens=8192,
            store=True,
            text_format=ImagingReport,
        )

        if not hasattr(response, 'output_parsed'):
            logging.error("output_parsed 항목이 응답에 존재하지 않음")
            return None

        return response.output_parsed.model_dump_json(indent=2)
    except Exception as e:
        logging.exception(f"JLK_ICH 실행 중 오류 발생: {e}")
        return None


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
