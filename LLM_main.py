import pydicom
from tqdm import tqdm
import numpy as np
import pydicom
from PIL import Image
import os
from pathlib import Path
from utils import draw_overlay_cti
from CT_Preprocessing import PreprocessingCTImage as pp
import numpy as np
from pathlib import Path
import pandas as pd
import re
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed

# 경로 객체로 처리

def get_jlk_summary_dirs(p: Path):
    if not re.search(r'ICH|CTI|CTL|WMHC|CVL', str(p)):
        return []
    try:
        return [
            child.name
            for child in p.iterdir()
            if child.is_dir() and 'summary' in child.name.lower()
        ]
    except Exception as e:
        print(f"[오류] {p} → {e}")
        return []

def join_jlk_full_paths(row):
    if not isinstance(row['JLK_AI'], list):
        return []
    return [row['file'] / sub for sub in row['JLK_AI']]

def normalize_to_8bit(array):
    array = array.astype(np.float32)
    array -= array.min()
    array /= (array.max() + 1e-8)
    array *= 255
    return array.astype(np.uint8)

def dicom_to_png(path, save_path):
    ds = pydicom.dcmread(path)
    arr = normalize_to_8bit(ds.pixel_array)
    image = Image.fromarray(arr)
    image.save(save_path)

def convert_all_dicom_to_png(grouped_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for num, row in tqdm(grouped_df.iterrows(), total=len(grouped_df)):
        patientID = row['patientID']
        StudyDesc = row['StudyDesc'].split('. ')[-1]
        SeriesDesc = row['SeriesDesc'].split('. ')[-1]
        dir_path = os.path.join(output_dir, f"{patientID}_{StudyDesc}_{SeriesDesc}")
        os.makedirs(dir_path, exist_ok=True)
        if "NCCT" in row['modality']: 
            non_mask(row['file'], dir_path)
        else:
            for idx, f in enumerate(row['JLK_AI_full_dcm']):
                sub_path = os.path.join(dir_path, row['modality'])
                os.makedirs(sub_path, exist_ok=True)
                fname = os.path.basename(f).replace(".dcm", f"_{idx}.png")
                save_path = os.path.join(sub_path, fname)
                dicom_to_png(f, save_path)

def non_mask(dcm_path, dir_path):
    image_sitk = pp.read_dicom(dcm_path)
    image_np = pp.to_numpy(image_sitk)
    total_region = np.zeros_like(image_np)
    draw_fig,_,_,_ = draw_overlay_cti(image_np,total_region)
    draw_fig = draw_fig.astype(np.uint8)[:, :, :3]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(draw_fig)
    plt.axis('off')  # 축 제거
    plt.tight_layout()
    non_mask_path = os.path.join(dir_path, "Non_mask")
    os.makedirs(non_mask_path, exist_ok=True)
    plt.savefig(f'{non_mask_path}\\non_mask.png', bbox_inches='tight', pad_inches=0, facecolor='black')

def _collection_path(extract_path):
    extract_path = Path(extract_path)  # 문자열이면 Path로 변환
    paths = list(extract_path.glob("*/*/*/*/*"))  # 5단계 하위 폴더 모두 수집

    # DataFrame 구성
    path_ = pd.DataFrame(paths, columns=['file'])

    # 메타 정보 추출
    path_['patientID'] = path_['file'].apply(lambda x: x.parts[-4])
    path_['StudyDesc'] = path_['file'].apply(lambda x: x.parts[-3])
    path_['SeriesDesc'] = path_['file'].apply(lambda x: x.parts[-2])
    path_['modality'] = path_['file'].apply(lambda x: x.parts[-1])
    path_['JLK_AI'] = path_['file'].apply(get_jlk_summary_dirs)
    path_['JLK_AI_full'] = path_.apply(join_jlk_full_paths, axis=1)
    path_['JLK_AI_full_dcm'] = path_['JLK_AI_full'].apply(
        lambda folders: list(
            chain.from_iterable(folder.rglob("*.dcm") for folder in folders)
        ) if isinstance(folders, list) else []
    )
    return path_



def process_row(row, output_dir):
    patientID = row['patientID']
    StudyDesc = row['StudyDesc'].split('. ')[-1]
    SeriesDesc = row['SeriesDesc'].split('. ')[-1]
    dir_path = os.path.join(output_dir, f"{patientID}_{StudyDesc}_{SeriesDesc}")
    os.makedirs(dir_path, exist_ok=True)

    if "NCCT" in row['modality']: 
        non_mask(row['file'], dir_path)
    else:
        for idx, f in enumerate(row['JLK_AI_full_dcm']):
            sub_path = os.path.join(dir_path, row['modality'])
            os.makedirs(sub_path, exist_ok=True)
            fname = os.path.basename(f).replace(".dcm", f"_{idx}.png")
            save_path = os.path.join(sub_path, fname)
            dicom_to_png(f, save_path)

def convert_all_dicom_to_png_parallel(grouped_df, output_dir, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in grouped_df.iterrows():
            futures.append(executor.submit(process_row, row, output_dir))

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass  # 결과값을 저장할 필요 없고, 완료 여부만 추적함

from openai import OpenAI
from pydantic import BaseModel

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import base64
from io import BytesIO

from PIL import Image
from io import BytesIO
import base64
from glob import glob
from tqdm import tqdm
import os
import logging
from typing import Optional
def pil_to_base64(image_path):
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
    
def JLK_ICH(mask_path: str,path: str) -> Optional[str]:
    try:

        content_blocks = []

        desc_masked = (
            "Summary_0000_0 : JLK ICH는 비조영 CT에서 모든 유형의 뇌출혈을 탐지하는 AI 알고리즘입니다. 뇌출혈 의심 영역은 붉은색 마스크로 표시되며, 환자 단위의 뇌출혈 확률값과 전체 뇌영역에서의 뇌출혈 부피 정보가 함께 제공됩니다."
        )
        img_masked = safe_load_image(f'{path}/Summary_0000_0.png')
        if img_masked:
            content_blocks.append({"type": "input_text", "text": desc_masked})
            content_blocks.append({"type": "input_image", "image_url": img_masked})

        desc_nomask = (
            "Summary_0000_0_non_mask : Summary_0000_0과 동일한 영상이며, 붉은색 마스크를 제거한 버전입니다. (정확한 비교를 위해 mask만 제외되었습니다.)"
        )
        img_nomask = safe_load_image(f'{mask_path}/non_mask.png')
        if img_nomask:
            content_blocks.append({"type": "input_text", "text": desc_nomask})
            content_blocks.append({"type": "input_image", "image_url": img_nomask})

        if not content_blocks:
            logging.error("유효한 이미지가 없어 요청을 수행할 수 없습니다.")
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


extract_path = extract_zip(r"D:\새 폴더 (2)\add_data\JLK_NCCT\DCM_REQUEST_2025-07-23-05-16-58-465027_0.zip")
collection_path = _collection_path(extract_path)
convert_all_dicom_to_png_parallel(collection_path, Path(r"D:\새 폴더 (2)\add_data\test"))

from pathlib import Path

path = Path(r'D:\새 폴더 (2)\add_data\test')
first_level = list(path.glob('*'))
ich_result = []
for item in first_level:
    if item.is_dir():
        non_mask_dir = item / 'Non_mask'
        if non_mask_dir.exists() and any(non_mask_dir.glob('*')):
            CTL_dir = next((p for p in item.glob('*') if p.is_dir() and 'CTL' in p.name), None)
            CTI_dir = next((p for p in item.glob('*') if p.is_dir() and 'CTI' in p.name), None)
            WMHC_dir = next((p for p in item.glob('*') if p.is_dir() and 'WMHC' in p.name), None)
            ICH_dir = next((p for p in item.glob('*') if p.is_dir() and 'ICH' in p.name), None)
            ich_result.append(JLK_ICH(non_mask_dir, ICH_dir))
        else:
            print("Non_mask 없음:", item)
