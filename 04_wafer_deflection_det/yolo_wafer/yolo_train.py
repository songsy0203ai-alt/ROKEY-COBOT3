from ultralytics import YOLO
from datetime import datetime
import os

project_folder = "yolo_wafer"
# 모델 파일이 현재 폴더에 있다고 가정
model_filename = "yolov8n.pt"
data_yaml_filename = "data.yaml"

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, project_folder)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_name = f"train_{timestamp}"

# 파일 경로 재확인 (존재 여부 체크 권장)
model_path = model_filename 
data_yaml_path = os.path.join(BASE_DIR, data_yaml_filename)

model = YOLO(model_path)

results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,         # 2의 거듭제곱 권장
    patience=20,
    project=OUTPUT_DIR,
    name=train_name,
    device=0         # GPU 사용 시 추가 (CPU 사용 시 제거 또는 'cpu')
)

print(f"학습 및 검증 완료. 결과 저장 위치: {os.path.join(OUTPUT_DIR, train_name)}")
