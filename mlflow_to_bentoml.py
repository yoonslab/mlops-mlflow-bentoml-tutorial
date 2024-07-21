import bentoml
import mlflow
import boto3
import os

# Minio 서버 설정
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'  # Minio 서버 주소 및 포트
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'  # Minio 액세스 키
os.environ['AWS_SECRET_ACCESS_KEY'] = 'miniostorage'  # Minio 시크릿 키

# MLflow 추적 서버 URI 설정
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5001'  # MLflow 서버 주소

# MLflow 모델 로드
model_name = "web-ctr"
model_alias = "Staging"

prod_model = mlflow.sklearn.load_model(f"models:/{model_name}@{model_alias}")

# BentoML에 모델 저장
saved_model = bentoml.sklearn.save_model(
    model_name,
    prod_model,
    signatures={
        "predict": {"batchable": True},
        "predict_proba": {"batchable": True}
    }
)
print(f"Model saved: {saved_model}")
