from ultralytics import YOLO
from datetime import datetime
import os

project_folder = "Fruits"
model_filename = "best.pt" # train 디렉토리에 있는 best.pt를 project_folder로 복사
data_yaml_filename = "data.yaml"
predict_images_path = "predict_images"

# 자동으로 현재 폴더 기준 경로 설정
BASE_DIR = os.getcwd()  # 현재 폴더 기준
OUTPUT_DIR = os.path.join(BASE_DIR, project_folder)  # 원하는 상위 폴더 이름

# 날짜/시간 기반 하위 폴더 이름 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
predict_name = f"predict_{timestamp}"

model_path = os.path.join(OUTPUT_DIR, model_filename)
data_yaml_path = os.path.join(OUTPUT_DIR, data_yaml_filename)

# Model 로드
model = YOLO(model_path)

# 예측 실행
model.predict(
    source=os.path.join(OUTPUT_DIR, predict_images_path),  # 예측 대상 이미지 폴더
    conf=0.25,
    project=OUTPUT_DIR,
    name=predict_name,
    save=True
)

# 최종 출력 경로 확인
print(f"Predict : {os.path.join(OUTPUT_DIR, predict_name)}")
