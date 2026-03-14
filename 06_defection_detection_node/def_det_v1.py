#!/usr/bin/env python3
"""
[코드 기능]
- 웨이퍼 이미지에서 결함(scratch, donut) 또는 정상(none) 상태를 실시간으로 탐지하는 ROS 2 노드입니다.
- Ultralytics YOLO 모델을 사용하여 객체 탐지(Detection) 또는 분류(Classification)를 수행합니다.

[입력(Input)]
- 이미지 데이터: /wafer_camera/image_raw (sensor_msgs/Image)
- 학습된 모델: wafer_best.pt (YOLOv8/v11 weight file)

[출력(Output)]
- 탐지 결과: /def_det_result (std_msgs/String) - 'none', 'scratch', 'donut' 중 하나를 발행
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

# 기본 모델 경로 설정
MODEL_PATH = (
    "/home/rokey/cobot3_ws/04_wafer_deflection_det/"
    "yolo_wafer/yolo_wafer/train_20260310_152505/weights/wafer_best.pt"
)

# YOLO 클래스 이름을 시스템 내부 레이블로 매핑
LABEL_MAP = {
    "none":    "none",
    "scratch": "scratch",
    "donut":   "donut",
    "normal":  "none",
    "ok":      "none",
    "good":    "none",
}

# 유효한 레이블 집합 및 신뢰도 임계값
VALID_LABELS = {"none", "scratch", "donut"}
CONF_THRESHOLD = 0.5


class WaferDefectDetector(Node):
    def __init__(self):
        """
        [함수 설명] 노드 초기화 및 파라미터, 모델, ROS 인터페이스 설정
        [Input] 없음
        [Output] WaferDefectDetector 인스턴스
        """
        super().__init__("wafer_defect_detector")

        # ── 1. 파라미터 선언 및 가져오기 ────────────────────────────────────
        self.declare_parameter("model_path",      MODEL_PATH)
        self.declare_parameter("conf_threshold",  CONF_THRESHOLD)
        self.declare_parameter("input_topic",     "/wafer_camera/image_raw")
        self.declare_parameter("output_topic",    "/def_det_result")

        model_path     = self.get_parameter("model_path").value
        self.conf_thr  = self.get_parameter("conf_threshold").value
        input_topic    = self.get_parameter("input_topic").value
        output_topic   = self.get_parameter("output_topic").value

        # ── 2. YOLO 모델 로드 ───────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path) # 지정된 경로에서 YOLO 가중치 파일 로드
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load model: {e}")
            raise

        # ── 3. ROS 인터페이스 및 유틸리티 설정 ───────────────────────────────
        self.bridge = CvBridge() # ROS Image 메시지를 OpenCV 포맷으로 변환하는 브릿지

        # 이미지 구독(Subscribe) 설정
        self.sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10,
        )

        # 결과 문자열 발행(Publish) 설정
        self.pub = self.create_publisher(String, output_topic, 10)

        self.get_logger().info(
            f"Subscribing : {input_topic}\n"
            f"Publishing  : {output_topic}"
        )

    def image_callback(self, msg: Image):
        """
        [함수 설명] 구독한 이미지 메시지를 처리하여 결함 여부를 판단하고 결과를 발행함
        [Input] msg: sensor_msgs/Image (카메라로부터 받은 원본 이미지 데이터)
        [Output] 없음 (결과를 Topic으로 발행)
        """
        # ROS Image 메시지를 OpenCV BGR 이미지 포맷으로 변환
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # 추론 함수 호출하여 결과 레이블 획득
        result_label = self._detect(cv_image)

        if result_label in ("donut", "scratch"):
            # 불량 탐지 시: 경고 이모지와 함께 레이블 명시
            self.get_logger().warn(f"🔴 [REPORT] 불량 웨이퍼 탐지됨! ({result_label.upper()})")
        else:
            # 정상 탐지 시: 체크 이모지와 함께 출력
            self.get_logger().info(f"🟢 [REPORT] 정상 웨이퍼 (NONE)")

        # 결과를 String 메시지에 담아 발행
        out_msg = String()
        out_msg.data = result_label
        self.pub.publish(out_msg)

        self.get_logger().info(f"Detection result: {result_label}")

    def _detect(self, image: np.ndarray) -> str:
        """
        [함수 설명] OpenCV 이미지에 대해 YOLO 추론을 수행하고 가장 신뢰도가 높은 결함 레이블을 반환
        [Input] image: np.ndarray (OpenCV BGR 이미지)
        [Output] str: 탐지된 결함 레이블 ('none', 'scratch', 'donut')
        """
        # YOLO 모델 추론 수행 (로그 출력을 끔)
        results = self.model(image, verbose=False)

        best_label = "none" # 기본값 설정
        best_conf  = 0.0

        for result in results:
            # ── A. Classification 모델 결과 처리 (이미지 전체 분류) ──────────
            if result.probs is not None:
                probs     = result.probs.data.cpu().numpy() # 확률 데이터 추출
                class_idx = int(np.argmax(probs))           # 가장 높은 확률의 인덱스
                conf      = float(probs[class_idx])         # 해당 클래스의 신뢰도
                raw_name  = result.names[class_idx].lower() # 클래스 이름

                # 임계값 및 유효성 검사 후 최적 레이블 업데이트
                if conf >= self.conf_thr:
                    label = LABEL_MAP.get(raw_name, raw_name)
                    if label in VALID_LABELS and conf > best_conf:
                        best_conf  = conf
                        best_label = label

                self.get_logger().debug(
                    f"[cls] top={raw_name} conf={conf:.3f} → {best_label}"
                )

            # ── B. Detection 모델 결과 처리 (경계 상자 탐지) ─────────────────
            elif result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    conf      = float(boxes.conf[i].cpu())  # 탐지된 박스의 신뢰도
                    class_idx = int(boxes.cls[i].cpu())     # 탐지된 클래스 인덱스
                    raw_name  = result.names[class_idx].lower()

                    label = LABEL_MAP.get(raw_name, raw_name)
                    if label not in VALID_LABELS:
                        continue

                    # 가장 신뢰도가 높은 박스의 레이블을 결과로 선택
                    if conf >= self.conf_thr and conf > best_conf:
                        best_conf  = conf
                        best_label = label

                    self.get_logger().debug(
                        f"[det] cls={raw_name} conf={conf:.3f} → {label}"
                    )

        return best_label


def main(args=None):
    """
    [함수 설명] rclpy 초기화 및 노드 실행 메인 루프
    [Input] args: 커맨드라인 인자
    [Output] 없음
    """
    rclpy.init(args=args)
    node = WaferDefectDetector()
    try:
        rclpy.spin(node) # 노드가 종료될 때까지 대기 및 콜백 처리
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()