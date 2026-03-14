from ultralytics import YOLO
import cv2
import os

project_folder = "Fruits"
model_filename = "best.pt" # train 디렉토리에 있는 best.pt를 project_folder로 복사

# 모델 로드
# 자동으로 현재 폴더 기준 경로 설정
BASE_DIR = os.getcwd()  # 현재 폴더 기준
OUTPUT_DIR = os.path.join(BASE_DIR, project_folder)  # 원하는 상위 폴더 이름

# model_path = os.path.join(OUTPUT_DIR, model_filename)
model_path = "/home/rokey/Tutorial/Fruits/Fruits/train_20260210_172748/weights/best.pt"
model = YOLO(model_path)

# 카메라 열기 (0: 기본 내장 카메라)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 예측 시작 (종료: Q 키 누르기)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 예측 (stream=False: 프레임 1개씩 예측)
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # 예측 결과 가져오기
    result = results[0]
    boxes = result.boxes  # bounding box 정보
    classes = result.names  # 클래스 이름들

    for box in boxes:
        cls_id = int(box.cls[0])              # 클래스 ID
        conf = float(box.conf[0]) * 100       # 신뢰도 (0~1 → 0~100%)
        label = f"{classes[cls_id]} {conf:.1f}%"

        # Bounding box 좌표
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 정수 변환

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 클래스 및 확률 표시
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("YOLO Predict", frame)

    # Q 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
print("예측 종료")
