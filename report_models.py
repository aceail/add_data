from typing import List

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    patientId: str = Field(..., description="환자 고유 식별자")
    gender: str = Field(..., description="성별(Male/Female/Other 등)")
    age: int = Field(..., description="나이 (단위: 년)")
    imagingModalities: List[str] = Field(
        ..., description="영상 촬영 모달리티 리스트 (예: ['NCCT', 'CTA', 'CTP'])"
    )
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
