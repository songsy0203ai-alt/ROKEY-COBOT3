#!/usr/bin/env python3
"""
[코드 기능]
- 웨이퍼 카메라 영상(/wafer_camera/image_raw)을 실시간으로 구독하여 YOLO 모델로 결함을 탐지합니다.
- 탐지된 결과(정상, 스크래치, 도넛)를 문자열로 발행하며, OpenCV 창을 통해 실시간 검사 화면을 시각화합니다.
- 터미널 로그가 너무 빠르게 올라가지 않도록 2초 간격으로 상태를 보고합니다.

[입력(Input)]
- 이미지 토픽: /wafer_camera/image_raw (sensor_msgs/Image)
- 모델 파일: wafer_best.pt (YOLOv8/v11 weight)

[출력(Output)]
- 결과 토픽: /def_det_result (std_msgs/String) -> 'none', 'scratch', 'donut'
- 시각화: OpenCV Window ("Wafer Defect Detection")
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
import cv2
import numpy as np
from ultralytics import YOLO

# 모델 가중치 파일의 절대 경로
MODEL_PATH = (
    "/home/rokey/cobot3_ws/04_wafer_deflection_det/"
    "yolo_wafer/yolo_wafer/train_20260313_173709/weights/wafer_best_v2.pt"
)

# YOLO 클래스 명칭을 프로젝트 표준 레이블로 매핑
LABEL_MAP = {
    "none":    "none",
    "scratch": "scratch",
    "donut":   "donut",
    "normal":  "none",
    "ok":      "none",
    "good":    "none",
}

VALID_LABELS = {"none", "scratch", "donut"}
CONF_THRESHOLD = 0.4      # 탐지 확신도 임계값 (40% 이상일 때만 인정)


class WaferDefectDetector(Node):
    def __init__(self):
        """
        [함수 설명] 노드 생성자. 파라미터 로드, YOLO 모델 초기화 및 ROS 통신 설정을 수행합니다.
        [Input] 없음
        [Output] WaferDefectDetector 인스턴스
        """
        super().__init__("wafer_defect_detector")

        # ── 파라미터 선언 및 설정값 가져오기 ────────────────────────────────
        self.declare_parameter("model_path",      MODEL_PATH)
        self.declare_parameter("conf_threshold",  CONF_THRESHOLD)
        self.declare_parameter("input_topic",     "/wafer_camera/image_raw")
        self.declare_parameter("output_topic",    "/def_det_result")

        model_path     = self.get_parameter("model_path").value
        self.conf_thr  = self.get_parameter("conf_threshold").value
        input_topic    = self.get_parameter("input_topic").value
        output_topic   = self.get_parameter("output_topic").value

        # ── YOLO 모델 로드 ───────────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path) # 모델 로딩
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load model: {e}")
            raise

        # ── 로그 쓰로틀링 변수 (터미널 로그 도배 방지용) ────────────────────
        self.last_log_time = 0.0
        self.log_interval  = 2.0   # 2초마다 한 번씩만 출력

        # ── ROS 인터페이스 설정 ──────────────────────────────────────────────
        self.bridge = CvBridge() # ROS 이미지 <-> OpenCV 이미지 변환기

        # 이미지 구독자(Subscriber) 설정
        self.sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10,
        )

        # 결과 발행자(Publisher) 설정
        self.pub = self.create_publisher(String, output_topic, 10)

        self.get_logger().info(
            f"Subscribing : {input_topic}\n"
            f"Publishing  : {output_topic}"
        )

    def image_callback(self, msg: Image):
        """
        [함수 설명] 카메라 이미지를 수신할 때마다 호출되어 탐지 및 시각화, 결과를 발행합니다.
        [Input] msg: sensor_msgs/Image (입력 이미지 데이터)
        [Output] 없음 (주제 발행 및 화면 표시)
        """
        # 1. ROS 이미지를 OpenCV BGR 포맷으로 변환
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # 2. YOLO 추론 수행 (결과 레이블과 박스가 그려진 이미지 반환받음)
        result_label, annotated = self._detect(cv_image)

        # 3. 시각화 텍스트 오버레이 (정상은 녹색, 불량은 적색)
        color = (0, 200, 0) if result_label == "none" else (0, 0, 255)
        label_text = (
            "NORMAL" if result_label == "none"
            else f"DEFECT: {result_label.upper()}"
        )
        # 이미지 좌상단에 큼직하게 텍스트 출력
        cv2.putText(
            annotated, label_text,
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, color, 2, cv2.LINE_AA,
        )
        # OpenCV 윈도우 창 업데이트
        cv2.imshow("Wafer Defect Detection", annotated)
        cv2.waitKey(1) # 창 갱신을 위한 짧은 대기

        # 4. 탐지 결과 발행
        out_msg = String()
        out_msg.data = result_label
        self.pub.publish(out_msg)

        # 5. 주기적 터미널 로그 출력 (2초 간격)
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            self.last_log_time = now
            if result_label in ("donut", "scratch"):
                # 불량 탐지 시 경고(WARN) 레벨 로그
                self.get_logger().warn(f"🔴 [REPORT] 불량 웨이퍼 탐지됨! ({result_label.upper()})")
            else:
                # 정상 탐지 시 정보(INFO) 레벨 로그
                self.get_logger().info(f"🟢 [REPORT] 정상 웨이퍼 (NONE)")

    def _detect(self, image: np.ndarray):
        """
        [함수 설명] 입력된 이미지에 대해 YOLO 추론을 수행하고 결과와 시각화된 이미지를 반환합니다.
        [Input] image: np.ndarray (OpenCV BGR 이미지)
        [Output] (str, np.ndarray): (최종 레이블, 박스/마스크가 그려진 결과 이미지)
        """
        results = self.model(image, verbose=False) # 추론 수행

        best_label    = "none"
        best_conf     = 0.0
        annotated     = image.copy() # 원본 복사본 생성

        for result in results:
            # ── A. 분류(Classification) 결과 처리 ──────────────────────────
            if result.probs is not None:
                probs     = result.probs.data.cpu().numpy()
                class_idx = int(np.argmax(probs))
                conf      = float(probs[class_idx])
                raw_name  = result.names[class_idx].lower()

                if conf >= self.conf_thr:
                    label = LABEL_MAP.get(raw_name, raw_name)
                    if label in VALID_LABELS and conf > best_conf:
                        best_conf  = conf
                        best_label = label

            # ── B. 객체 탐지(Detection) 결과 처리 ────────────────────────────
            elif result.boxes is not None and len(result.boxes) > 0:
                # YOLO 자체 제공 시각화 도구로 박스 그리기
                annotated = result.plot()

                boxes = result.boxes
                for i in range(len(boxes)):
                    conf      = float(boxes.conf[i].cpu())
                    class_idx = int(boxes.cls[i].cpu())
                    raw_name  = result.names[class_idx].lower()

                    label = LABEL_MAP.get(raw_name, raw_name)
                    if label not in VALID_LABELS:
                        continue

                    # 여러 개의 객체 중 가장 확신도가 높은 것의 레이블 선택
                    if conf >= self.conf_thr and conf > best_conf:
                        best_conf  = conf
                        best_label = label

        return best_label, annotated


def main(args=None):
    """
    [함수 설명] rclpy를 초기화하고 노드를 실행하며, 종료 시 리소스를 해제합니다.
    """
    rclpy.init(args=args)
    node = WaferDefectDetector()
    try:
        rclpy.spin(node) # 콜백 실행 무한 루프
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows() # 열려있는 OpenCV 창 닫기
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()