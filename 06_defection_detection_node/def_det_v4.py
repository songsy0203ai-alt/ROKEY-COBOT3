#!/usr/bin/env python3
"""
[코드 기능]
- 웨이퍼 카메라 영상(/wafer_camera/image_raw)을 실시간으로 구독하여 YOLO 모델로 결함을 탐지합니다.
- empty / none / scratch / donut 4개 클래스를 YOLO가 직접 판정합니다.
- OpenCV 창을 통해 실시간 검사 화면을 시각화합니다.
- 터미널 로그는 2초 간격으로 출력합니다.

[클래스 정의]
  empty   : 카메라 구역에 웨이퍼가 아직 없음
  none    : 정상 웨이퍼 (결함 없음)
  scratch : 불량 웨이퍼 (스크래치 결함)
  donut   : 불량 웨이퍼 (도넛형 결함)

[입력(Input)]
- 이미지 토픽: /wafer_camera/image_raw (sensor_msgs/Image)
- 모델 파일: wafer_best.pt (YOLOv8/v11 weight, 4-class)

[출력(Output)]
- 결과 토픽: /def_det_result (std_msgs/String) → 'empty' | 'none' | 'scratch' | 'donut'
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
    "yolo_wafer/yolo_wafer/train_20260310_152505/weights/wafer_best.pt"
)

# YOLO 클래스 명칭 → 프로젝트 표준 레이블 매핑
# 모델 학습 시 사용한 클래스명이 다를 경우 여기에 추가
LABEL_MAP = {
    "empty":   "empty",
    "none":    "none",
    "scratch": "scratch",
    "donut":   "donut",
    # 학습 데이터에서 다른 이름을 썼을 경우 대비
    "normal":  "none",
    "ok":      "none",
    "good":    "none",
    "background": "empty",
    "bg":      "empty",
}

# YOLO가 판정할 수 있는 모든 유효 레이블
VALID_LABELS = {"empty", "none", "scratch", "donut"}

CONF_THRESHOLD = 0.7  # 이 확신도 미만이면 판정 무효 → "empty"로 처리


class WaferDefectDetector(Node):
    def __init__(self):
        super().__init__("wafer_defect_detector")

        # ── 파라미터 선언 ────────────────────────────────────────────────────
        self.declare_parameter("model_path",     MODEL_PATH)
        self.declare_parameter("conf_threshold", CONF_THRESHOLD)
        self.declare_parameter("input_topic",    "/wafer_camera/Compressed")
        self.declare_parameter("output_topic",   "/def_det_result")

        model_path    = self.get_parameter("model_path").value
        self.conf_thr = self.get_parameter("conf_threshold").value
        input_topic   = self.get_parameter("input_topic").value
        output_topic  = self.get_parameter("output_topic").value

        # ── YOLO 모델 로드 ───────────────────────────────────────────────────
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load model: {e}")
            raise

        # ── 로그 쓰로틀링 ───────────────────────────────────────────────────
        self.last_log_time = 0.0
        self.log_interval  = 2.0

        # ── ROS 인터페이스 ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.pub = self.create_publisher(String, output_topic, 10)

        self.get_logger().info(
            f"Subscribing : {input_topic}\n"
            f"Publishing  : {output_topic}"
        )

    # ────────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        result_label, annotated = self._detect(cv_image)

        # ── 시각화 오버레이 ──────────────────────────────────────────────────
        if result_label == "empty":
            color      = (200, 200, 200)
            label_text = "WAITING FOR WAFER..."
        elif result_label == "none":
            color      = (0, 200, 0)
            label_text = "NORMAL"
        else:  # scratch / donut
            color      = (0, 0, 255)
            label_text = f"DEFECT: {result_label.upper()}"

        cv2.putText(
            annotated, label_text,
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1.1, color, 2, cv2.LINE_AA,
        )
        cv2.imshow("Wafer Defect Detection", annotated)
        cv2.waitKey(1)

        # ── 결과 발행 ────────────────────────────────────────────────────────
        out_msg = String()
        out_msg.data = result_label
        self.pub.publish(out_msg)

        # ── 주기적 터미널 로그 (2초 간격) ────────────────────────────────────
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            self.last_log_time = now
            if result_label == "empty":
                self.get_logger().info("⏳ 아직 웨이퍼가 분류 구역에 없습니다.")
            elif result_label in ("scratch", "donut"):
                self.get_logger().warn(f"🔴 [REPORT] 불량 웨이퍼 탐지됨! ({result_label.upper()})")
            else:
                self.get_logger().info("🟢 [REPORT] 정상 웨이퍼 (NONE)")

    # ────────────────────────────────────────────────────────────────────────
    def _detect(self, image: np.ndarray):
        """
        YOLO 추론을 수행하고 (result_label, annotated_image)를 반환한다.

        판정 기준:
          - Classification 모드(probs): 가장 높은 확신도의 클래스를 채택.
                                        conf_threshold 미만이면 "empty"로 처리.
          - Detection 모드(boxes)     : 가장 높은 conf의 박스 클래스를 채택.
                                        탐지된 박스가 없으면 "empty"로 처리.

        "empty"는 YOLO가 직접 판정하는 정식 클래스이므로,
        탐지 실패(fallback)와 구분하지 않고 동일하게 취급한다.
        """
        results   = self.model(image, verbose=False)
        best_label = "empty"   # conf_threshold 미달 또는 탐지 없음의 경우 기본값
        best_conf  = 0.0
        annotated  = image.copy()

        for result in results:

            # ── Classification 모드 ──────────────────────────────────────────
            if result.probs is not None:
                probs     = result.probs.data.cpu().numpy()
                class_idx = int(np.argmax(probs))
                conf      = float(probs[class_idx])
                raw_name  = result.names[class_idx].lower()
                label     = LABEL_MAP.get(raw_name, raw_name)

                if label not in VALID_LABELS:
                    # 알 수 없는 클래스는 empty로 간주
                    label = "empty"

                # conf_threshold 미만이면 판정 신뢰도 부족 → empty 유지
                if conf >= self.conf_thr and conf > best_conf:
                    best_conf  = conf
                    best_label = label

            # ── Detection 모드 ───────────────────────────────────────────────
            elif result.boxes is not None and len(result.boxes) > 0:
                annotated = result.plot()

                for i in range(len(result.boxes)):
                    conf      = float(result.boxes.conf[i].cpu())
                    class_idx = int(result.boxes.cls[i].cpu())
                    raw_name  = result.names[class_idx].lower()
                    label     = LABEL_MAP.get(raw_name, raw_name)

                    if label not in VALID_LABELS:
                        label = "empty"

                    if conf >= self.conf_thr and conf > best_conf:
                        best_conf  = conf
                        best_label = label
            # 박스가 없는 Detection 결과 → best_label은 "empty" 유지

        return best_label, annotated


def main(args=None):
    rclpy.init(args=args)
    node = WaferDefectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()