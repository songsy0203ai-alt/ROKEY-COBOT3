"""
[코드 기능]
wafer1 → wafer2 → wafer3 순차 처리 버전.
- /def_det_result 토픽 구독 (YOLO detection 결과: none / scratch / donut)
- /wafer_camera/image_raw 토픽 퍼블리싱 (맵 내 카메라 영상)
- detection 결과에 따라 Phase 7의 place 위치 분기
"""

import numpy as np
import asyncio
import threading

import omni.usd
import omni.kit.app
import omni.kit.viewport.utility as vp_util

from pxr import Gf, UsdGeom, UsdLux, Sdf, Usd

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation

# ── ROS2 임포트 ────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7 분기용 place 좌표 상수
# detection 결과에 따라 아래 두 좌표 중 하나를 사용한다
# ──────────────────────────────────────────────────────────────────────────────
PLACE_NORMAL   = np.array([-2.75777, -8.8769,  2.125   ])  # none → 정상 적재
PLACE_DEFECT   = np.array([-0.73097, -8.94011, 2.09002 ])  # scratch / donut → 불량 적재
# ──────────────────────────────────────────────────────────────────────────────


class _WaferProxy:
    """
    맵에 이미 존재하는 웨이퍼 Prim의 월드 좌표를 읽기 위한 USD 직접 접근 래퍼.
    RigidPrim의 버전별 생성자 차이를 우회한다.
    """
    def __init__(self, prim_path: str):
        self._prim_path = prim_path

    def get_world_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"[_WaferProxy] Invalid prim path: {self._prim_path}")

        xform = UsdGeom.Xformable(prim)
        # Usd.TimeCode.Default() → 매 스텝 최신 변환 행렬을 읽어옴
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()

        pos = np.array([translation[0], translation[1], translation[2]])
        imag = rotation.GetImaginary()
        ori = np.array([rotation.GetReal(), imag[0], imag[1], imag[2]])
        return pos, ori


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 노드: /def_det_result 구독 + /wafer_camera/image_raw 퍼블리싱
# Isaac Sim의 메인 루프와 분리하여 별도 스레드에서 spin한다.
# ──────────────────────────────────────────────────────────────────────────────
class WaferROS2Node(Node):

    def __init__(self):
        super().__init__("wafer_pickup_node")

        self._det_result = None
        self._det_result_lock = threading.Lock()

        self.create_subscription(
            String,
            "/def_det_result",
            self._det_result_callback,
            10,
        )
        self.get_logger().info("[ROS2] Subscribed to /def_det_result")

        self._image_pub = self.create_publisher(Image, "/wafer_camera/image_raw", 10)
        self.get_logger().info("[ROS2] Publisher ready: /wafer_camera/image_raw")

        # ── 별도 스레드에서 spin 실행 ────────────────────────────────
        # Isaac Sim 내부 executor와 충돌을 피하기 위해
        # 이 노드 전용 executor를 백그라운드 스레드에서 돌린다.
        # physics_step에서 spin_once를 호출하지 않아도
        # 콜백이 즉시 처리된다.
        from rclpy.executors import SingleThreadedExecutor
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,   # 메인 프로세스 종료 시 자동으로 같이 종료
        )
        self._spin_thread.start()
        self.get_logger().info("[ROS2] Background spin thread started.")
        # ────────────────────────────────────────────────────────────

    def _det_result_callback(self, msg: String):
        result = msg.data.strip().lower()
        with self._det_result_lock:
            self._det_result = result
        self.get_logger().info(f"[ROS2] /def_det_result received: '{result}'")

    def get_det_result(self):
        with self._det_result_lock:
            return self._det_result

    def reset_det_result(self):
        with self._det_result_lock:
            self._det_result = None
        self.get_logger().info("[ROS2] Detection result reset.")

    def publish_camera_image(self, rgb_array: np.ndarray):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "wafer_camera"
        msg.height = rgb_array.shape[0]
        msg.width  = rgb_array.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = rgb_array.shape[1] * 3
        msg.data = rgb_array.tobytes()
        self._image_pub.publish(msg)
# ──────────────────────────────────────────────────────────────────────────────


class RMPFlowController(mg.MotionPolicyController):

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:
        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflow"
            )

        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)

        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )


def setup_camera_and_light():
    stage = omni.usd.get_context().get_stage()

    cam_path = "/World/ScriptCamera"
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))

    transform_matrix = Gf.Matrix4d(
        -0.99978,  0.02106,  0.00066, 0.0,
        -0.01815,  -6.0,     0.48032, 0.0,
        -0.0107,  -0.4802,  -0.87709, 0.0,
        -1.26865,  -8.66478,  5.24915, 1.0,
    )
    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform_matrix)

    cam_prim.GetHorizontalApertureAttr().Set(20.955)
    cam_prim.GetFocalLengthAttr().Set(8.0)

    viewport = vp_util.get_active_viewport()
    if viewport:
        viewport.set_active_camera(cam_path)

    light_path = "/World/CameraLight"
    light = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
    light.GetRadiusAttr().Set(1.6)
    light.GetIntensityAttr().Set(10000.0)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    light_xform = UsdGeom.Xformable(light)
    light_xform.ClearXformOpOrder()
    light_xform.AddTranslateOp().Set(Gf.Vec3d(-4.95977, -8.53786, 8.14303))


class WaferPickup:

    WAFER_PRIM_PATHS = [
        "/World/Factory/smcnd_factory_v4/wafers/wafer1",
        "/World/Factory/smcnd_factory_v4/wafers/wafer2",
        "/World/Factory/smcnd_factory_v4/wafers/wafer3",
    ]

    # 맵 내 결함 분류 카메라 Prim 경로
    DEFECT_CAM_PRIM = (
        "/World/Factory/smcnd_factory_v4/defection_classification_bar"
        "/body4/defection_camera"
    )

    def __init__(self):
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])
        self.place_position = np.array([-2.9288041591644287, -8.730463027954102, 2.124993324279785])

        self._wafer_index        = 0
        self.wafer               = None
        self.task_phase          = 1
        self._wait_counter       = 0
        self._phase1_lock_counter = 0
        self._phase8_counter     = 0
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self._debug_printed      = False

        # ── detection 결과 및 카메라 촬영 상태 ────────────────────────
        # Phase 1 조건 만족 시 카메라를 1회만 촬영하도록 플래그 관리
        self._camera_shot_done   = False   # 현재 웨이퍼에 대한 촬영 완료 여부
        # Phase 7에서 사용할 최종 place 좌표 (detection 결과로 결정됨)
        self._resolved_place_pos = None
        # ────────────────────────────────────────────────────────────

    # ── Isaac Sim 카메라 → numpy RGB 배열 ────────────────────────────────
    def _capture_defect_camera(self) -> np.ndarray | None:
        """
        [기능] 맵 내 결함 분류 카메라(defection_camera)의 현재 프레임을
               numpy RGB 배열로 반환한다.
               IsaacSim의 synthetic data API를 사용한다.
        [출력] np.ndarray (H, W, 3) uint8  |  None (실패 시)
        """
        try:
            import omni.replicator.core as rep

            if not hasattr(self, '_rp'):
                self._rp = rep.create.render_product(
                    self.DEFECT_CAM_PRIM,
                    resolution=(640, 480),
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                self._rgb_annotator.attach(self._rp)

            data = self._rgb_annotator.get_data()
            
            # 배열이 None이거나 비어있거나, 이미지 차원(2D 이상)이 아닌 경우 캡처 실패로 간주
            if data is None or getattr(data, 'size', 0) == 0 or len(data.shape) < 2:
                return None

            if len(data.shape) == 3 and data.shape[-1] == 4:
                data = data[..., :3]

            return data.astype(np.uint8)

        except Exception as e:
            print(f"[WARN] _capture_defect_camera failed: {e}")
            return None
    # ────────────────────────────────────────────────────────────────────

    def _advance_wafer(self, world: World):
        if self._wafer_index >= len(self.WAFER_PRIM_PATHS):
            return False

        prim_path = self.WAFER_PRIM_PATHS[self._wafer_index]
        self.wafer = _WaferProxy(prim_path)

        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.GetMassAttr().Set(0.01)
            print(f"[WAFER] Mass set to 0.01 kg for {prim_path}")

        print(f"[WAFER] Active wafer → {prim_path} (index={self._wafer_index})")
        return True

    def _boost_gripper_force(self):
        stage = omni.usd.get_context().get_stage()
        gripper_prim = stage.GetPrimAtPath("/World/UR10/ee_link/SurfaceGripper")

        if not gripper_prim.IsValid():
            print("[WARN] _boost_gripper_force: SurfaceGripper prim not found")
            return

        attr_map = {
            "physxSurfaceGripper:forceLimit":    1.0e2,
            "physxSurfaceGripper:torqueLimit":   1.0e4,
            "physxSurfaceGripper:gripThreshold": 0.02,
            "physxSurfaceGripper:retryClose":    True,
        }
        for attr_name, value in attr_map.items():
            attr = gripper_prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(value)
                print(f"[GRIPPER] {attr_name} = {value}")
            else:
                print(f"[WARN] attribute not found: {attr_name}")

        print("[GRIPPER] All attributes on SurfaceGripper prim:")
        for attr in gripper_prim.GetAttributes():
            print(f"  {attr.GetName()} = {attr.Get()}")

    def setup_scene(self, world: World):
        world.scene.add_default_ground_plane()

        assets_root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            surface_gripper_path="/World/UR10/ee_link/SurfaceGripper",
        )

        ur10 = world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10",
                name="my_ur10",
                end_effector_prim_path="/World/UR10/ee_link",
                gripper=gripper,
            )
        )
        ur10.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )
        self._boost_gripper_force()

    def _reset_for_next_wafer(self):
        self._wait_counter            = 0
        self._phase1_lock_counter     = 0
        self._phase1_det_wait_counter = 0  # ← 추가
        self._phase7_wait_log_counter = 0  # ← 추가
        self._phase8_counter          = 0
        self._debug_printed           = False
        self._brown_wafer_position    = np.array([-1.83094, -7.92573, 1.94285])
        self._camera_shot_done        = False
        self._resolved_place_pos      = None
        self._ros2_node.reset_det_result()
        self.cspace_controller.reset()
        self.task_phase = 1
        print(f"[NEXT] State reset done. Starting wafer index={self._wafer_index}")

    def physics_step(self, step_size):
            if self.wafer is None:
                return
                
            # World 객체가 없거나 시뮬레이션의 첫 스텝인 경우 물리 뷰가 초기화되지 않았으므로 건너뜀
            if not hasattr(self, '_world') or self._world.current_time_step_index == 0:
                return

            try:
                self._physics_step_impl(step_size)
            except Exception as e:
                if "Failed to get DOF" in str(e):
                    pass
                else:
                    raise e

    def _physics_step_impl(self, step_size):

        if not self._debug_printed and self.task_phase == 1:
            robot_actual_pos, _ = self.robots.get_world_pose()
            wafer_actual_pos, _ = self.wafer.get_world_pose()
            rmpflow_base = self.cspace_controller._default_position
            print("=" * 50)
            print(f"[DEBUG] Robot actual world pose : {robot_actual_pos}")
            print(f"[DEBUG] RMPFlow base pose       : {rmpflow_base}")
            print(f"[DEBUG] Wafer actual world pose : {wafer_actual_pos}")
            print(f"[DEBUG] self.robot_position     : {self.robot_position}")
            print(f"[DEBUG] wafer - robot_position  : {wafer_actual_pos - self.robot_position}")
            print(f"[DEBUG] wafer - rmpflow_base    : {wafer_actual_pos - rmpflow_base}")
            print("=" * 50)
            self._debug_printed = True

        # Phase 1: 웨이퍼 X축 임계값 도달 대기 + detection 대기 (3초)
        if self.task_phase == 1:
            self._phase1_lock_counter = getattr(self, '_phase1_lock_counter', 0) + 1
            if self._phase1_lock_counter <= 30:
                return

            wafer_position, _ = self.wafer.get_world_pose()
            current_x_position = wafer_position[0]

            # ── 임시 진단: 매 60스텝마다 현재 X좌표 출력 ──
            if self._phase1_lock_counter % 60 == 0:
                print(f"[DIAG Phase 1] wafer_index={self._wafer_index} | X={current_x_position:.5f} | threshold=-1.625")
            # ────────────────────────────────────────────

            if current_x_position >= -1.625:

                # Phase 1 안의 카메라 촬영 블록을 아래로 교체
                if not self._camera_shot_done:
                    rgb = self._capture_defect_camera()
                    if rgb is not None:
                        self._ros2_node.publish_camera_image(rgb)
                        print(f"[Phase 1] ✅ Camera image published ({rgb.shape}) to /wafer_camera/image_raw")
                    else:
                        # ── 캡처 실패 시 더미 이미지라도 전송해서 YOLO 노드가 동작하게 함 ──
                        # _capture_defect_camera()가 replicator 초기화 타이밍 문제로
                        # 첫 호출에 None을 반환하는 경우가 있다. 더미 이미지로 대체한다.
                        print(f"[Phase 1] ⚠️ Camera capture failed — sending dummy image")
                        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                        self._ros2_node.publish_camera_image(dummy)
                    self._camera_shot_done = True

                    # ── 진단: 현재 det_result 상태 출력 ──────────────────────
                    print(f"[Phase 1] det_result at shot time = '{self._ros2_node.get_det_result()}'")

                # ── detection 대기: 조건 만족 후 3초(180스텝) 대기 ──────
                # YOLO 노드가 /def_det_result를 퍼블리싱할 시간을 확보한다.
                # _phase1_det_wait_counter로 별도 카운팅한다.
                self._phase1_det_wait_counter = getattr(self, '_phase1_det_wait_counter', 0) + 1

                if self._phase1_det_wait_counter % 60 == 0:
                    elapsed = self._phase1_det_wait_counter // 60
                    print(f"[Phase 1] Waiting for detection result... ({elapsed}/3 sec)")

                if self._phase1_det_wait_counter >= 90: # 1.5초
                    print(f"[Phase 1] Wafer X ({current_x_position:.4f}) reached & 3s elapsed → Phase 2")
                    self._phase1_lock_counter = 0
                    self._phase1_det_wait_counter = 0
                    self.task_phase = 2
                # ────────────────────────────────────────────────────────

        # Phase 2: 10스텝 대기 후 웨이퍼 위치 저장
        elif self.task_phase == 2:
            print(f"[DIAG Phase 2] _wait_counter={self._wait_counter}")  # 임시 진단

            if self._wait_counter < 10:
                self._wait_counter += 1
            else:
                self._brown_wafer_position, _ = self.wafer.get_world_pose()
                print(f"[Phase 2] Wafer stabilized at {self._brown_wafer_position} → Phase 3")
                self.task_phase = 3

        # Phase 3: 웨이퍼 위쪽 접근
        elif self.task_phase == 3:
            _target_position = self._brown_wafer_position.copy()
            print('[Debugging Log] Phase 3 )  configuring target position done')
            print(_target_position)
            _target_position[2] = 1.82

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.01
            ):
                print("[Debugging Log] Phase 3 )  Approach done → Phase 4")
                self.cspace_controller.reset()
                self.task_phase = 4

        # Phase 4: 웨이퍼 위치로 하강
        elif self.task_phase == 4:
            pick_position = self._brown_wafer_position.copy()
            _target_position = pick_position - self.robot_position

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            print("[Debugging Log] Phase 4 ) end_effector_orientation done")
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            print("[Debugging Log] Phase 4 ) action done")

            current_joint_positions = self.robots.get_joint_positions()
            print(current_joint_positions)
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.1
            ):
                print("[Phase 4] Descend done → Phase 5")
                self.cspace_controller.reset()
                self.task_phase = 5

        # Phase 5: 그리퍼 흡착
        elif self.task_phase == 5:
            print("[Phase 5] Gripper close → Phase 6")
            self.robots.gripper.close()
            self.task_phase = 6

        # Phase 6: 수직 상승
        elif self.task_phase == 6:
            _target_position = self._brown_wafer_position.copy() - self.robot_position
            _target_position[0] -= 4.0
            _target_position[1] -= 2.0
            _target_position[2] += 3.0

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            print(end_effector_orientation)
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.01
            ):
                print("[Phase 6] Lift done → Phase 7")
                self.cspace_controller.reset()
                self.task_phase = 7
                print('[Debugging Log] Now Phase 7')


        # Phase 7: detection 결과에 따라 place 위치 분기 후 이동
        elif self.task_phase == 7:

            if self._resolved_place_pos is None:
                det = self._ros2_node.get_det_result()

                # "empty" 이거나 아직 결과가 None인 경우 대기
                if det is None or det == "empty":
                    return

                # "none"인 경우 (정상 웨이퍼)
                if det == "none":
                    self._resolved_place_pos = np.array([-2.75777, -8.8769, 2.125])
                    print(f"[Phase 7] ✅ det='{det}' → PLACE_NORMAL {self._resolved_place_pos}")
                # "scratch" 또는 "donut"인 경우 (불량 웨이퍼)
                elif det in ["scratch", "donut"]:
                    self._resolved_place_pos = np.array([-0.73097, -8.94011, 2.09002])
                    print(f"[Phase 7] ✅ det='{det}' → PLACE_DEFECT {self._resolved_place_pos}")
                else:
                    # 정의되지 않은 문자열이 들어올 경우 대기
                    print(f"[Phase 7] ⚠️ 알 수 없는 결과 수신: '{det}'. 올바른 값을 대기합니다.")
                    return

            _target_position = self._resolved_place_pos

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            print('[Debugging Log] Phase 7 apply_action done')

            current_joint_positions = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.01
            ):
                print(f"[Phase 7] Place position reached ({self._resolved_place_pos}) → Phase 8")
                self.cspace_controller.reset()
                self.task_phase = 8

        # Phase 8: 그리퍼 개방 (60스텝 타임아웃)
        elif self.task_phase == 8:
            self.robots.gripper.open()
            self._phase8_counter = getattr(self, '_phase8_counter', 0) + 1

            print(f"[Phase 8] open() step={self._phase8_counter}")

            if self._phase8_counter >= 60:
                wafer_pos, _ = self.wafer.get_world_pose()
                ee_pos, _    = self.robots.end_effector.get_world_pose()
                dist = np.linalg.norm(wafer_pos - ee_pos)
                print(f"[Phase 8] Released after {self._phase8_counter} steps | dist={dist:.4f} → Phase 9")
                self._phase8_counter = 0
                self.task_phase = 9

        # Phase 9: 홈 복귀 후 다음 웨이퍼로 전환
        elif self.task_phase == 9:
            home_joint_positions = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])

            from isaacsim.core.utils.types import ArticulationAction
            action = ArticulationAction(joint_positions=home_joint_positions)
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - home_joint_positions) < 0.01):
                print(f"[Phase 9] Home reached. wafer index={self._wafer_index} done.")

                self._wafer_index += 1
                if self._wafer_index < len(self.WAFER_PRIM_PATHS):
                    self._advance_wafer(self._world)
                    self._reset_for_next_wafer()
                    print(f"[NEXT] → Now processing: {self.WAFER_PRIM_PATHS[self._wafer_index - 1]}")
                else:
                    print("[DONE] All 3 wafers processed. Simulation complete.")
                    self.wafer = None


async def main():
    # ── ROS2 초기화 ──────────────────────────────────────────────────────
    # Isaac Sim이 내부적으로 rclpy.init()을 이미 호출해두므로
    # 중복 호출하면 RuntimeError: Context.init() must only be called once 발생.
    # init() 없이 노드만 생성하면 기존 컨텍스트를 그대로 사용한다.
    if not rclpy.ok():
        rclpy.init()   # 혹시라도 초기화가 안 된 환경에서 실행될 경우를 대비한 방어 코드
    ros2_node = WaferROS2Node()
    # ────────────────────────────────────────────────────────────────────

    world = World.instance()
    if world is not None:
        world.stop()
        world.clear_instance()

    await omni.usd.get_context().new_stage_async()

    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    sim = WaferPickup()

    # ── ROS2 노드를 sim 인스턴스에 연결 ──────────────────────────────────
    sim._ros2_node = ros2_node
    # ────────────────────────────────────────────────────────────────────

    background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v17.usd"
    add_reference_to_stage(usd_path=background_usd, prim_path="/World/Factory")

    setup_camera_and_light()
    await world.reset_async()

    sim.setup_scene(world)
    await world.reset_async()

    sim.robots = world.scene.get_object("my_ur10")
    sim.robots.set_world_pose(position=sim.robot_position)

    sim._advance_wafer(world)

    sim.cspace_controller = RMPFlowController(
        name="my_ur10_cspace_controller",
        robot_articulation=sim.robots,
        attach_gripper=True,
    )
    sim._world = world

    actual_pos, _ = sim.robots.get_world_pose()
    wafer_pos, _  = sim.wafer.get_world_pose()
    print("=" * 50)
    print(f"[INIT] Robot set_world_pose target : {sim.robot_position}")
    print(f"[INIT] Robot actual world pose     : {actual_pos}")
    print(f"[INIT] RMPFlow base pose           : {sim.cspace_controller._default_position}")
    print(f"[INIT] Wafer world pose (wafer1)   : {wafer_pos}")
    print("=" * 50)

    world.add_physics_callback("sim_step", callback_fn=sim.physics_step)
    await world.play_async()
    print("Simulation started. Processing wafer1 → wafer2 → wafer3 ...")


asyncio.ensure_future(main())
