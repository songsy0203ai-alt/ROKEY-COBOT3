"""
[코드 기능]
wafer1 → wafer2 → wafer3 순차 처리 버전.
RigidPrim 버전 호환성 문제를 _WaferProxy(USD 직접 접근)로 우회.

[변경사항]
/def_det_result ROS2 토픽으로 수신되는 YOLO detection class에 따라
Phase 7의 place 목적지를 분기한다.
  - empty  : 품질 분류 구역 미진입 → Phase 1에서 계속 대기
  - none   : 정상 웨이퍼 → [-2.75777, -8.8769, 2.125]
  - scratch : 불량 웨이퍼 → [-0.73097, -8.94011, 2.09002]
  - donut  : 불량 웨이퍼 → [-0.73097, -8.94011, 2.09002]

[Latch 정책]
YOLO 노드는 프레임마다 결과를 발행하므로 none/scratch/donut 사이에
empty가 끼어들 수 있다. 한 번이라도 유효한 결과(none/scratch/donut)가
수신되면 그 값을 래치(latch)하여 이후 empty 메시지는 무시한다.
reset() 호출 시에만 latch가 해제된다.
"""

import numpy as np
import asyncio
import threading

import omni.usd
import omni.kit.app
import omni.kit.viewport.utility as vp_util

from pxr import Gf, UsdGeom, UsdLux, Sdf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 Detection Subscriber
# ──────────────────────────────────────────────────────────────────────────────
class DetectionSubscriber:
    """
    [기능] ROS2 /def_det_result 토픽을 구독하여 최신 detection class를 저장한다.
           별도 스레드에서 rclpy.spin()을 실행하므로 Isaac Sim 메인 루프를 블로킹하지 않는다.

    [Latch 정책]
    none/scratch/donut 중 하나가 한 번이라도 수신되면 래치(latch) 상태가 되어
    이후 empty 메시지를 무시한다. reset() 호출 시에만 latch가 해제된다.
    이를 통해 empty → scratch → empty → donut 패턴에서도
    가장 최근에 수신된 유효 판정이 유지된다.
    """

    VALID_CLASSES = {"empty", "none", "scratch", "donut"}
    RESULT_CLASSES = {"none", "scratch", "donut"}  # empty가 아닌 유효 판정 클래스

    def __init__(self):
        self.latest_class: str = "empty"  # 초기값: 구역 미진입
        self._latched: bool = False        # True이면 이후 empty 메시지 무시
        self._lock = threading.Lock()
        self._node = None
        self._spin_thread = None
        self._init_ros()

    def _init_ros(self):
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String

            if not rclpy.ok():
                rclpy.init()

            class _InnerNode(Node):
                def __init__(inner_self):
                    super().__init__("wafer_detection_subscriber")
                    inner_self.create_subscription(
                        String,
                        "/def_det_result",
                        inner_self._callback,
                        10,
                    )

                def _callback(inner_self, msg: "String"):
                    value = msg.data.strip().lower()
                    if value not in DetectionSubscriber.VALID_CLASSES:
                        print(f"[DetectionSubscriber] Unknown class received: '{msg.data}' → ignored")
                        return

                    with self._lock:
                        # ── Latch 로직 ──────────────────────────────────────
                        # 유효 판정(none/scratch/donut)이 한 번이라도 수신된 뒤에는
                        # empty 메시지를 무시하여 값이 empty로 되돌아가지 않게 한다.
                        if self._latched and value == "empty":
                            return

                        prev = self.latest_class
                        self.latest_class = value

                        # 유효 판정이 수신되면 래치 활성화
                        if value in DetectionSubscriber.RESULT_CLASSES:
                            self._latched = True
                        # ────────────────────────────────────────────────────

                    if prev != value:
                        latch_tag = " [LATCHED]" if self._latched else ""
                        print(f"[DetectionSubscriber] Class updated: '{prev}' → '{value}'{latch_tag}")

            self._node = _InnerNode()

            self._spin_thread = threading.Thread(
                target=self._spin_worker,
                daemon=True,
                name="ros2_detection_spin",
            )
            self._spin_thread.start()
            print("[DetectionSubscriber] ROS2 subscriber started on /def_det_result")

        except Exception as e:
            print(f"[DetectionSubscriber] ROS2 init failed: {e}")
            print("[DetectionSubscriber] Falling back to empty (no-op mode)")

    def _spin_worker(self):
        import rclpy
        try:
            rclpy.spin(self._node)
        except Exception as e:
            print(f"[DetectionSubscriber] spin error: {e}")

    def get_class(self) -> str:
        """스레드 안전하게 최신 detection class를 반환한다."""
        with self._lock:
            return self.latest_class

    def reset(self):
        """다음 웨이퍼 처리 시작 전 detection class와 latch를 초기화한다."""
        with self._lock:
            self.latest_class = "empty"
            self._latched = False  # latch 해제 → 다음 웨이퍼 판정 가능
        print("[DetectionSubscriber] Reset → 'empty' (latch cleared)")


# ──────────────────────────────────────────────────────────────────────────────
# RigidPrim 버전 호환성 우회용 USD 직접 접근 래퍼
# ──────────────────────────────────────────────────────────────────────────────
class _WaferProxy:
    """
    [기능] 맵에 이미 존재하는 웨이퍼 Prim의 월드 좌표를 읽기 위한 USD 직접 접근 래퍼.
    [입력] prim_path (str): USD 스테이지 내 Prim 경로
    [출력] get_world_pose() → (position: np.ndarray, orientation: np.ndarray)
    """
    def __init__(self, prim_path: str):
        self._prim_path = prim_path

    def get_world_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"[_WaferProxy] Invalid prim path: {self._prim_path}")

        xform = UsdGeom.Xformable(prim)

        from pxr import Usd
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()

        pos = np.array([translation[0], translation[1], translation[2]])
        imag = rotation.GetImaginary()
        ori = np.array([rotation.GetReal(), imag[0], imag[1], imag[2]])
        return pos, ori


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
    light.GetIntensityAttr().Set(70000.0)
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

    # Phase 7 목적지 좌표
    _PLACE_POS_NORMAL = np.array([-2.75777, -8.8769,  2.125  ])  # none   (정상)
    _PLACE_POS_DEFECT = np.array([-0.73097, -8.94011, 2.09002])  # scratch / donut (불량)

    def __init__(self):
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])
        self.place_position = np.array([-2.9288041591644287, -8.730463027954102, 2.124993324279785])

        self._wafer_index = 0
        self.wafer = None

        self.task_phase = 1
        self._wait_counter = 0
        self._phase1_lock_counter = 0
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self._debug_printed = False

        self._detection_class: str = "empty"
        self._place_target: np.ndarray = self._PLACE_POS_NORMAL.copy()

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
        gripper_prim_path = "/World/UR10/ee_link/SurfaceGripper"
        gripper_prim = stage.GetPrimAtPath(gripper_prim_path)

        if not gripper_prim.IsValid():
            print(f"[WARN] _boost_gripper_force: prim not found at {gripper_prim_path}")
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
        """다음 웨이퍼 처리를 위해 상태를 초기화한다."""
        self._wait_counter = 0
        self._phase1_lock_counter = 0
        self._phase8_counter = 0
        self._debug_printed = False
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])

        self._detection_class = "empty"
        self._place_target = self._PLACE_POS_NORMAL.copy()
        self._detection_subscriber.reset()  # latest_class = "empty", _latched = False

        self.cspace_controller.reset()
        self.task_phase = 1
        print(f"[NEXT] State reset done. Starting wafer index={self._wafer_index}")

    def _update_detection_class(self):
        """
        DetectionSubscriber에서 최신 detection class를 읽어 내부 상태를 갱신한다.
        래치 덕분에 empty → scratch → empty 패턴에서도 scratch가 유지된다.
        Phase 7 목적지(_place_target)도 여기서 결정한다.
        """
        cls = self._detection_subscriber.get_class()
        if cls == self._detection_class:
            return  # 변화 없음

        self._detection_class = cls
        if cls == "none":
            self._place_target = self._PLACE_POS_NORMAL.copy()
            print(f"[DETECTION] class='{cls}' → NORMAL place {self._place_target}")
        elif cls in ("scratch", "donut"):
            self._place_target = self._PLACE_POS_DEFECT.copy()
            print(f"[DETECTION] class='{cls}' → DEFECT place {self._place_target}")
        # "empty"는 목적지를 변경하지 않는다 (Phase 1 대기 상태 유지)

    def physics_step(self, step_size):
        if self.wafer is None:
            return
        self._update_detection_class()
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

        # Phase 1: 웨이퍼 X축 임계값 도달 대기 + detection 'empty' 해제 대기
        if self.task_phase == 1:
            self._phase1_lock_counter = getattr(self, '_phase1_lock_counter', 0) + 1
            if self._phase1_lock_counter <= 30:
                return

            # 래치 덕분에 한 번 none/scratch/donut이 수신되면
            # 이후 empty가 와도 _detection_class는 유효 판정값을 유지한다.
            if self._detection_class == "empty":
                return

            wafer_position, _ = self.wafer.get_world_pose()
            current_x_position = wafer_position[0]
            print(current_x_position)
            # print(current_x_position)
            if current_x_position >= -1.625: # -1.625
                print(
                    f"[Phase 1] Wafer X ({current_x_position:.4f}) reached the barrier "
                    f"| detection='{self._detection_class}' → Phase 2"
                )
                self._phase1_lock_counter = 0
                self.task_phase = 2

        # Phase 2: 10스텝 대기 후 웨이퍼 위치 저장
        elif self.task_phase == 2:
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
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
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
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                print("[Phase 6] Lift done → Phase 7")
                self.cspace_controller.reset()
                self.task_phase = 7
                print('[Debugging Log] Now Phase 7')

        # Phase 7: Place 위치로 이동 (detection class에 따라 목적지 분기)
        elif self.task_phase == 7:
            print('[Debugging Log] Phase 7 start')

            # none        → _PLACE_POS_NORMAL  [-2.75777, -8.8769,  2.125  ]
            # scratch/donut → _PLACE_POS_DEFECT  [-0.73097, -8.94011, 2.09002]
            _target_position = self._place_target.copy()
            print(f"[Phase 7] Target={_target_position} (detection='{self._detection_class}')")

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            print('[Debugging Log] Phase 7 apply_action done')

            current_joint_positions = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                print("[Phase 7] Place position reached → Phase 8")
                self.cspace_controller.reset()
                self.task_phase = 8

        # Phase 8: 그리퍼 개방
        elif self.task_phase == 8:
            self.robots.gripper.open()
            self._phase8_counter = getattr(self, '_phase8_counter', 0) + 1

            print(f"[Phase 8] open() step={self._phase8_counter}")

            if self._phase8_counter >= 60:
                wafer_pos, _ = self.wafer.get_world_pose()
                ee_pos, _ = self.robots.end_effector.get_world_pose()
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
    world = World.instance()
    if world is not None:
        world.stop()
        world.clear_instance()

    await omni.usd.get_context().new_stage_async()

    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    sim = WaferPickup()

    # DetectionSubscriber 초기화 (ROS2 토픽 구독 시작)
    sim._detection_subscriber = DetectionSubscriber()

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
    wafer_pos, _ = sim.wafer.get_world_pose()
    print("=" * 50)
    print(f"[INIT] Robot set_world_pose target : {sim.robot_position}")
    print(f"[INIT] Robot actual world pose     : {actual_pos}")
    print(f"[INIT] RMPFlow base pose           : {sim.cspace_controller._default_position}")
    print(f"[INIT] Wafer world pose (wafer1)   : {wafer_pos}")
    print("=" * 50)

    world.add_physics_callback("sim_step", callback_fn=sim.physics_step)
    await world.play_async()
    print("Simulation started. Processing wafer1 → wafer2 → wafer3 ...")
    print("Detection routing: none→NORMAL | scratch/donut→DEFECT")


asyncio.ensure_future(main())