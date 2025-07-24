import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from report_models import ImagingReport

try:  # Optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - openai might not be available
    OpenAI = None

if OpenAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:  # pragma: no cover - when openai is unavailable
    client = None


def pil_to_base64(image_path: str) -> str:
    """Return the base64-encoded string of the given image."""

    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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
            prompt={"id": "pmpt_687ee31179fc819091e852e25eaca6c20399e215f5db1fab", "version": "2"},
            input=[{"role": "user", "content": content_blocks}],
            reasoning={},
            max_output_tokens=8192,
            store=True,
            text_format=ImagingReport,
        )

        if not hasattr(response, "output_parsed"):
            logging.error("output_parsed 항목이 응답에 존재하지 않음")
            return None

        return response.output_parsed.model_dump_json(indent=2)
    except Exception as e:
        logging.exception(f"JLK_ICH 실행 중 오류 발생: {e}")
        return None
