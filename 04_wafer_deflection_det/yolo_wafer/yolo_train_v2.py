from ultralytics import YOLO
from datetime import datetime
import os

project_folder = "yolo_wafer"
model_filename = "yolov8n.pt"
data_yaml_filename = "data.yaml"

BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, project_folder)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_name = f"train_{timestamp}"

model_path = model_filename 
data_yaml_path = os.path.join(BASE_DIR, data_yaml_filename)

model = YOLO(model_path)

# ── 훈련 및 데이터 증강 하이퍼파라미터 설정 ──────────────────────────────
results = model.train(
    data=data_yaml_path,
    epochs=100,              # 과적합 방지 로직이 추가되었으므로 에포크를 늘려 학습 유도
    imgsz=640,
    batch=8,
    patience=30,             # Early stopping 기준 완화
    project=OUTPUT_DIR,
    name=train_name,
    device=0,
    
    # [기하학적 변형 증강]
    degrees=180.0,           # 이미지 회전 (-180도 ~ +180도). 웨이퍼 검사에 필수적
    translate=0.1,           # 상하좌우 이동 (이미지 크기의 10%). 객체 위치 다양성 확보
    scale=0.5,               # 크기 조절 (±50%). 객체 크기 변화 학습
    shear=0.0,               # 기울임 (웨이퍼 형태가 왜곡될 수 있으므로 0.0 유지)
    perspective=0.0,         # 원근법 변화 (마찬가지로 형태 왜곡 방지를 위해 0.0)
    flipud=0.5,              # 상하 반전 확률 (50%)
    fliplr=0.5,              # 좌우 반전 확률 (50%)
    
    # [색상 및 광원 변형 증강]
    hsv_h=0.015,             # 색상(Hue) 변형 조정. 파란/노란색이 완전히 다른 색이 되지 않게 낮게 설정
    hsv_s=0.7,               # 채도(Saturation) 변형 조정
    hsv_v=0.4,               # 명도(Value) 변형 조정. 밝기 변화 시뮬레이션
    
    # [YOLO 특화 증강 기법]
    mosaic=1.0,              # 4장의 이미지를 하나로 합치는 모자이크 증강 (기본값 활성화)
    mixup=0.1,               # 두 이미지를 겹치는 MixUp 비율. 단조로운 배경 타파에 도움
    copy_paste=0.0           # 분할(Segmentation) 마스크가 없으므로 0.0으로 비활성화
)

print(f"학습 및 검증 완료. 결과 저장 위치: {os.path.join(OUTPUT_DIR, train_name)}")