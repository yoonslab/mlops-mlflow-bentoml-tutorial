import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, List

class AdClickData(BaseModel):
    ID: str = Field(default="TRAIN_00000000", description="Unique identifier for the sample")
    F01: Optional[str] = Field(default="NSLHFNS", description="Feature 01")
    F02: Optional[str] = Field(default="AVKQTCL", description="Feature 02")
    F03: Optional[str] = Field(default="DTZFPRW", description="Feature 03")
    F04: Optional[float] = Field(default=114.0, description="Feature 04")
    F05: Optional[str] = Field(default="ISVXFVA", description="Feature 05")
    F06: int = Field(default=1, description="Feature 06")
    F07: str = Field(default="PQZBVMG", description="Feature 07")
    F08: str = Field(default="LPYPUNA", description="Feature 08")
    F09: str = Field(default="", description="Feature 09")
    F10: Optional[str] = Field(default="", description="Feature 10")
    F11: Optional[float] = Field(default=None, description="Feature 11")
    F12: Optional[str] = Field(default="", description="Feature 12")
    F13: str = Field(default="", description="Feature 13")
    F14: int = Field(default=0, description="Feature 14")
    F15: Optional[str] = Field(default="", description="Feature 15")
    F16: str = Field(default="", description="Feature 16")
    F17: str = Field(default="", description="Feature 17")
    F18: Optional[float] = Field(default=None, description="Feature 18")
    F19: Optional[float] = Field(default=None, description="Feature 19")
    F20: Optional[str] = Field(default="", description="Feature 20")
    F21: str = Field(default="", description="Feature 21")
    F22: str = Field(default="", description="Feature 22")
    F23: str = Field(default="", description="Feature 23")
    F24: Optional[float] = Field(default=None, description="Feature 24")
    F25: str = Field(default="", description="Feature 25")
    F26: Optional[str] = Field(default="", description="Feature 26")
    F27: Optional[float] = Field(default=None, description="Feature 27")
    F28: str = Field(default="", description="Feature 28")
    F29: Optional[float] = Field(default=None, description="Feature 29")
    F30: str = Field(default="NZGEZLW", description="Feature 30")
    F31: str = Field(default="GTISJWW", description="Feature 31")
    F32: Optional[float] = Field(default=380.0, description="Feature 32")
    F33: Optional[float] = Field(default=2.0, description="Feature 33")
    F34: Optional[str] = Field(default="AXQFZWC", description="Feature 34")
    F35: str = Field(default="IRUDRFB", description="Feature 35")
    F36: Optional[float] = Field(default=None, description="Feature 36")
    F37: str = Field(default="TFJMLCZ", description="Feature 37")
    F38: Optional[float] = Field(default=0.0, description="Feature 38")
    F39: str = Field(default="AURZYDY", description="Feature 39")

# BentoML 모델 러너 설정
web_ctr_runner = bentoml.sklearn.get("web_ctr:latest").to_runner()

# BentoML 서비스 정의
svc = bentoml.Service("web-ctr-predictor", runners=[web_ctr_runner])

@svc.api(input=bentoml.io.JSON(pydantic_model=AdClickData), output=bentoml.io.NumpyNdarray())
async def predict(data: AdClickData) -> np.ndarray:
    # Pydantic 모델을 pandas DataFrame으로 변환
    df = pd.DataFrame([data.dict()])
    
    # 'Click' 열이 있다면 제거 (예측에 사용되지 않음)
    if 'Click' in df.columns:
        df = df.drop('Click', axis=1)
    
    # 예측 실행
    return await web_ctr_runner.predict.async_run(df)

@svc.api(input=bentoml.io.JSON(pydantic_model=List[AdClickData]), output=bentoml.io.NumpyNdarray())
async def predict_batch(data: List[AdClickData]) -> np.ndarray:
    # Pydantic 모델 리스트를 pandas DataFrame으로 변환
    df = pd.DataFrame([item.dict() for item in data])
    
    # 'Click' 열이 있다면 제거 (예측에 사용되지 않음)
    if 'Click' in df.columns:
        df = df.drop('Click', axis=1)
    
    # 배치 예측 실행
    return await web_ctr_runner.predict.async_run(df)

# 서비스 메타데이터 설정
svc.info(
    title="Web Ad Click Predictor",
    description="click ad 서비스",
    version="1.0.0",
)

# (선택사항) health check 엔드포인트 추가
@svc.api(input=bentoml.io.Text(), output=bentoml.io.JSON())
def health_check(input_data: str = "health") -> dict:
    if input_data == "health":
        return {"status": "ok", "message": "Service is healthy"}
    else:
        return {"status": "error", "message": "Invalid health check request"}
