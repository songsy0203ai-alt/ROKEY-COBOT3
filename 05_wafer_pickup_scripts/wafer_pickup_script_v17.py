"""
[코드 기능]
Isaac Sim 환경에서 UR10 매니퓰레이터와 흡착 그리퍼를 이용해 맵에 배치된 웨이퍼(wafer1~3)를
순서대로 픽업하여 지정된 위치에 배치하는 작업을 수행하는 물리 기반 시뮬레이션 스크립트입니다.
ROS2 토픽 /def_det_result를 구독하여 YOLO 검출 결과(none / scratch / donut)에 따라
배치 위치를 동적으로 결정합니다. 상태 머신(Phase 1~10)을 기반으로 모션 정책 제어기(RMPflow)를 통해
궤적을 생성합니다.

[수정사항]
1. /def_det_result ROS2 토픽 구독 추가 (yolo 검출 결과: none / scratch / donut)
2. DynamicCylinder(/World/wafer) 생성 제거 → 맵 내 기존 웨이퍼 prim 사용
3. background_usd → smcnd_factory_v12_2.usd 사용
4. wafer1~3 prim path 사용 (/World/smcnd_factory_v4/wafers/wafer1~3)
5. wafer1→wafer2→wafer3 순서로 순차 픽앤플레이스 수행
6. /def_det_result 결과에 따라 Phase 7 배치 좌표 분기
   - none    → (-2.75777, -8.8769, 2.125)
   - donut / scratch → (-0.73097, -8.94011, 2.09002)

[입력(Input)]
- 환경 리소스: UR10 로봇 USD, 공장 배경 USD (smcnd_factory_v12_2.usd)
- 초기 상태값: 로봇 월드 좌표(robot_position), 목표 배치 좌표(place_position)
- 제어 설정값: RMPflow 구성 파일 (UR10, RMPflowSuction)
- ROS2 토픽: /def_det_result (str: none / scratch / donut)

[출력(Output)]
- 로봇 동작: 매 물리 스텝마다 산출되는 관절 위치 목표값(Joint Position Action) 및 그리퍼 개폐 명령
- 콘솔 출력: 로봇 및 웨이퍼 현재 좌표 디버그 정보, 각 작업 단계(Task Phase) 전환 상태
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
# RigidPrim / XFormPrim 직접 사용 대신 USD stage API로 웨이퍼 prim 접근
# (이 Isaac Sim 버전에서는 prim_path 키워드 미지원으로 world.scene.add 불가)

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation

# ── ROS2 구독 (rclpy) ──────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False
    print("[WARN] rclpy를 불러올 수 없습니다. /def_det_result 토픽 구독이 비활성화됩니다.")
# ──────────────────────────────────────────────────────────────────────────────


# =============================================================================
# ROS2 구독 노드
# =============================================================================
class DetResultSubscriber:
    """
    [기능] /def_det_result 토픽을 구독하여 YOLO 검출 결과를 수신하는 ROS2 노드 래퍼.
           별도 스레드에서 spin하므로 Isaac Sim 메인 루프를 블로킹하지 않음.

    [입력] 없음
    [출력] self.latest_result (str | None): 가장 최근에 수신된 검출 결과 문자열
    """

    def __init__(self):
        self.latest_result: str | None = None  # 최신 검출 결과 저장
        self._node = None
        self._spin_thread = None

        if not _RCLPY_AVAILABLE:
            return

        # rclpy 초기화 (이미 초기화된 경우 무시)
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("wafer_det_result_subscriber")
        self._node.create_subscription(
            String,
            "/def_det_result",
            self._callback,
            10,
        )

        # 별도 데몬 스레드에서 spin
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        print("[ROS2] /def_det_result 구독 시작")

    def _callback(self, msg: "String"):
        """수신 콜백: 메시지 data 필드를 소문자로 정규화하여 저장"""
        self.latest_result = msg.data.strip().lower()
        print(f"[ROS2] /def_det_result 수신: '{self.latest_result}'")

    def _spin(self):
        """백그라운드 스레드에서 rclpy 이벤트 루프 실행"""
        rclpy.spin(self._node)

    def destroy(self):
        """노드 및 스레드 정리"""
        if self._node is not None:
            self._node.destroy_node()


# =============================================================================
# RMPflow 모션 제어기
# =============================================================================
class RMPFlowController(mg.MotionPolicyController):
    """충돌 회피 및 모션 생성을 담당하는 RMPflow 제어기 클래스"""

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:
        """
        [기능] RMPflow 제어기 초기화 및 정책 구성 파일 로드
        [입력]
          - name (str): 제어기 식별용 이름
          - robot_articulation (SingleArticulation): 제어할 대상 로봇 객체
          - physics_dt (float): 물리 엔진 시뮬레이션 스텝 시간
          - attach_gripper (bool): 그리퍼 장착 여부에 따른 정책 선택 플래그
        [출력] 없음 (클래스 인스턴스 생성)
        """
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
        """
        [기능] 제어기의 내부 상태를 초기화하고 기준 좌표계를 기본값으로 복구
        [입력] 없음
        [출력] 없음
        """
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )


# =============================================================================
# 카메라 및 조명 설정
# =============================================================================
def setup_camera_and_light():
    """
    [기능] 시뮬레이션 스테이지 내 스크립트 기반 카메라 및 구형 조명 생성/배치
    [입력] 없음
    [출력] 없음
    """
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


# =============================================================================
# 메인 픽앤플레이스 작업 클래스
# =============================================================================
class WaferPickup:
    """픽앤플레이스 작업의 상태 머신 및 씬 설정을 관리하는 주 작업 클래스"""

    # ── 맵 내 웨이퍼 prim 경로 정의 ──────────────────────────────────────────
    WAFER_PRIM_PATHS = [
        "/World/smcnd_factory_v4/wafers/wafer1",  # scratch
        "/World/smcnd_factory_v4/wafers/wafer2",  # none
        "/World/smcnd_factory_v4/wafers/wafer3",  # donut
    ]

    # ── Phase 7 배치 좌표 (검출 결과별) ──────────────────────────────────────
    PLACE_TARGET_NONE          = np.array([-2.75777, -8.8769,  2.125  ])
    PLACE_TARGET_DEFECT        = np.array([-0.73097, -8.94011, 2.09002])  # scratch / donut

    # ── 공통 설정값 ───────────────────────────────────────────────────────────
    # 모든 웨이퍼를 최종적으로 모아두는 컨베이어 도착 좌표
    CONVEYOR_ARRIVE_POSITION   = np.array([-5.93046331038515, -7.88888, 1.8305293321609353])

    def __init__(self):
        """
        [기능] 작업에 필요한 좌표 변수, 상태 머신 변수, 웨이퍼 인덱스 초기화
        [입력] 없음
        [출력] 없음
        """
        # ── 로봇 기준 좌표 ────────────────────────────────────────────────────
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])

        # ── 상태 머신 변수 ────────────────────────────────────────────────────
        self.task_phase    = 1      # 현재 작업 단계 (1~10)
        self._wait_counter = 0      # 지연 대기 카운터
        self._debug_printed = False # 디버그 중복 출력 방지 플래그

        # ── 웨이퍼 순서 관리 ──────────────────────────────────────────────────
        self.wafer_index = 0        # 현재 처리 중인 웨이퍼 인덱스 (0~2)
        self.wafer = None           # 현재 처리 대상 RigidPrim

        # ── 픽업/배치 위치 캐시 ───────────────────────────────────────────────
        self._brown_wafer_position = np.zeros(3)  # Phase 2에서 확정된 웨이퍼 위치

        # ── ROS2 검출 결과 구독기 ─────────────────────────────────────────────
        self._det_subscriber = DetResultSubscriber()

    # ── 씬 구성 ──────────────────────────────────────────────────────────────
    def setup_scene(self, world: World):
        """
        [기능] 로봇, 그리퍼 에셋을 스테이지에 로드하고 초기화.
               웨이퍼는 맵 USD 내 기존 prim을 사용하므로 별도 생성하지 않음.
        [입력]
          - world (World): 객체가 추가될 물리 시뮬레이션 World 인스턴스
        [출력] 없음
        """
        world.scene.add_default_ground_plane()

        # UR10 로봇 에셋 로드
        assets_root_path = (
            "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
        )
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

        # ── 맵 내 웨이퍼 prim은 world.scene.add 없이 USD stage에서 직접 접근 ───
        # RigidPrim / XFormPrim 생성자가 이 Isaac Sim 버전에서 prim_path 키워드를
        # 지원하지 않으므로, prim 참조는 _load_next_wafer()에서 stage API로 획득함.
        # (setup_scene에서는 별도 등록 불필요)
        pass  # 웨이퍼 prim은 USD 맵에 이미 존재하므로 추가 생성/등록 생략

    # ── 웨이퍼 전환 헬퍼 ─────────────────────────────────────────────────────
    # ── USD stage에서 웨이퍼 월드 포즈를 읽는 헬퍼 ──────────────────────────
    def _get_wafer_world_pose(self):
        """
        [기능] 현재 웨이퍼(self._current_wafer_prim_path)의 월드 좌표를
               USD XformCache를 통해 읽어 numpy array로 반환
        [입력] 없음
        [출력] (position: np.ndarray shape(3,), orientation: None)
        """
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(self._current_wafer_prim_path)
        if not prim.IsValid():
            print(f"[WARN] prim not valid: {self._current_wafer_prim_path}")
            return np.zeros(3), None

        xform_cache = UsdGeom.XformCache()
        world_mat   = xform_cache.GetLocalToWorldTransform(prim)
        translation = world_mat.ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]]), None

    def _load_next_wafer(self, world: World):
        """
        [기능] wafer_index에 해당하는 웨이퍼 prim 경로를 설정하고
               상태 머신을 Phase 1로 리셋함.
               world.scene.add / get_object 대신 USD stage API 직접 사용.
        [입력]
          - world (World): 사용하지 않지만 시그니처 호환성 유지
        [출력] bool - 다음 웨이퍼가 존재하면 True, 모두 완료면 False
        """
        if self.wafer_index >= len(self.WAFER_PRIM_PATHS):
            print("[Sequence] 모든 웨이퍼 처리 완료.")
            return False

        self._current_wafer_prim_path = self.WAFER_PRIM_PATHS[self.wafer_index]
        # self.wafer는 "웨이퍼가 활성 상태"임을 나타내는 플래그 용도로 True 유지
        self.wafer = True
        print(f"[Sequence] 다음 웨이퍼 로드: wafer{self.wafer_index + 1} "
              f"({self._current_wafer_prim_path})")

        # 상태 머신 초기화
        self.task_phase     = 1
        self._wait_counter  = 0   # Phase 1/2 대기 카운터 리셋
        self._debug_printed = False
        return True

    # ── 물리 스텝 콜백 ───────────────────────────────────────────────────────
    def physics_step(self, step_size):
        """
        [기능] 매 물리 스텝마다 호출되어 상태 머신(Phase)에 따라 로봇 이동 및
               그리퍼 제어를 수행. wafer_index를 통해 wafer1→wafer2→wafer3 순서로
               순차 처리함.
        [입력]
          - step_size (float): 이전 콜백과 현재 콜백 사이의 경과 시간
        [출력] 없음 (물리 엔진에 직접 Action 인가)
        """
        # 처리할 웨이퍼가 없으면(모두 완료) 콜백 즉시 반환
        if not self.wafer:
            return

        # ── 디버그: Phase 1 최초 진입(첫 스텝)에 1회만 출력 ────────────────────
        if not self._debug_printed and self.task_phase == 1 and self._wait_counter == 0:
            robot_actual_pos, _ = self.robots.get_world_pose()
            wafer_actual_pos, _ = self._get_wafer_world_pose()
            rmpflow_base        = self.cspace_controller._default_position

            print("=" * 50)
            print(f"[DEBUG] Robot actual world pose : {robot_actual_pos}")
            print(f"[DEBUG] RMPFlow base pose       : {rmpflow_base}")
            print(f"[DEBUG] Wafer actual world pose : {wafer_actual_pos}")
            print(f"[DEBUG] self.robot_position     : {self.robot_position}")
            print(f"[DEBUG] wafer - robot_position  : {wafer_actual_pos - self.robot_position}")
            print(f"[DEBUG] wafer - rmpflow_base    : {wafer_actual_pos - rmpflow_base}")
            print("=" * 50)
            self._debug_printed = True

        # ── Phase 1: 물리 엔진 안정화를 위해 30스텝 대기 ─────────────────────
        # 맵에 고정 배치된 웨이퍼이므로 컨베이어 도착 대기 없이
        # 물리 초기화가 완료될 때까지 짧게 대기한 뒤 바로 위치를 확정함
        if self.task_phase == 1:
            if self._wait_counter < 30:
                self._wait_counter += 1
            else:
                self._brown_wafer_position, _ = self._get_wafer_world_pose()
                print(f"[Phase 1] Wafer position confirmed: {self._brown_wafer_position} → Phase 2")
                self.task_phase = 2

        # ── Phase 2: 웨이퍼 위쪽 접근 전 추가 안정화 대기(10스텝) ──────────────
        elif self.task_phase == 2:
            if self._wait_counter < 40:
                self._wait_counter += 1
            else:
                # 위치를 한 번 더 읽어 최종 확정
                self._brown_wafer_position, _ = self._get_wafer_world_pose()
                print(f"[Phase 2] Wafer final position: {self._brown_wafer_position} → Phase 3")
                self.task_phase = 3

        # ── Phase 3: 웨이퍼 위쪽 접근 위치로 엔드이펙터 이동 ─────────────────
        # 웨이퍼 XY 위치 기준, Z만 높이 오프셋(+0.25m)을 더한 접근 좌표 계산
        elif self.task_phase == 3:
            _target_position = self._brown_wafer_position.copy() - self.robot_position
            _target_position[2] += 0.25   # 웨이퍼 위 25cm 접근 고도
            print(f"[Phase 3] Approach target (robot-relative): {_target_position}")

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
                print("[Phase 3] Approach done → Phase 4")
                self.cspace_controller.reset()
                self.task_phase = 4

        # ── Phase 4: 흡착을 위해 실제 웨이퍼 위치로 하강 이동 ────────────────
        elif self.task_phase == 4:
            pick_position    = self._brown_wafer_position.copy()
            _target_position = pick_position - self.robot_position

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            print("[Debugging Log] Phase 4) end_effector_orientation done")

            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            print("[Debugging Log] Phase 4) action done")

            current_joint_positions = self.robots.get_joint_positions()
            print(current_joint_positions)

            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.1
            ):
                print("[Phase 4] Descend done → Phase 5")
                self.cspace_controller.reset()
                self.task_phase = 5

        # ── Phase 5: 그리퍼 흡착 활성화 ─────────────────────────────────────
        elif self.task_phase == 5:
            print("[Phase 5] Gripper close → Phase 6")
            self.robots.gripper.close()
            self.task_phase = 6

        # ── Phase 6: 웨이퍼를 파지한 상태로 중간 경유 위치로 상승 ─────────────
        # 충돌 회피를 위해 충분히 높이 들어올린 뒤 Phase 7 목표 위치로 이동
        elif self.task_phase == 6:
            _target_position = self._brown_wafer_position.copy() - self.robot_position
            _target_position[2] += 1.5   # 현재 웨이퍼 위치에서 1.5m 상승

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            print(end_effector_orientation)

            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            print("[Debugging Log] Phase 6 apply_action done")

            current_joint_positions = self.robots.get_joint_positions()

            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                print("[Phase 6] Lift done → Phase 7")
                self.cspace_controller.reset()
                self.task_phase = 7
                print("[Debugging Log] Now Phase 7")

        # ── Phase 7: 목표 배치 위치로 이동 (검출 결과에 따라 좌표 분기) ────────
        elif self.task_phase == 7:
            print("[Debugging Log] Phase 7 start")

            # /def_det_result 토픽 수신값에 따라 배치 위치 결정
            det_result = self._det_subscriber.latest_result
            if det_result in ("scratch", "donut"):
                _target_position = self.PLACE_TARGET_DEFECT.copy()
                print(f"[Phase 7] 검출 결과='{det_result}' → 불량 배치 좌표 사용: {_target_position}")
            else:
                # none 이거나 아직 결과를 받지 못한 경우 기본(정상) 배치 좌표 사용
                _target_position = self.PLACE_TARGET_NONE.copy()
                print(f"[Phase 7] 검출 결과='{det_result}' → 정상 배치 좌표 사용: {_target_position}")

            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,
                target_end_effector_orientation=end_effector_orientation,
            )
            self.robots.apply_action(action)
            print("[Debugging Log] Phase 7 apply_action done")

            current_joint_positions = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                print("[Phase 7] Place position reached → Phase 8")
                self.cspace_controller.reset()
                self.task_phase = 8

        # ── Phase 8: 웨이퍼를 놓기 위해 그리퍼 개방 ─────────────────────────
        elif self.task_phase == 8:
            print("[Phase 8] Gripper open → Phase 9")
            self.robots.gripper.open()
            self.task_phase = 9

        # ── Phase 9: 초기 관절 자세로 복귀 ───────────────────────────────────
        elif self.task_phase == 9:
            home_joint_positions = np.array(
                [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]
            )

            from isaacsim.core.utils.types import ArticulationAction
            action = ArticulationAction(joint_positions=home_joint_positions)
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - home_joint_positions) < 0.01):
                print(
                    f"[Phase 9] Home position reached. "
                    f"wafer{self.wafer_index + 1} sequence complete."
                )
                self.task_phase = 10

        # ── Phase 10: 다음 웨이퍼로 전환 또는 전체 시퀀스 종료 ──────────────
        elif self.task_phase == 10:
            self.wafer_index += 1
            if self.wafer_index < len(self.WAFER_PRIM_PATHS):
                print(f"[Sequence] wafer{self.wafer_index + 1} 처리 시작...")
                self._load_next_wafer(self._world_ref)  # 다음 웨이퍼 로드 후 Phase 1로 리셋
            else:
                print("[Sequence] 모든 웨이퍼 처리 완료. 시뮬레이션 종료 가능.")
                self.wafer = None  # physics_step 즉시 반환 조건


# =============================================================================
# 메인 비동기 실행 함수
# =============================================================================
async def main():
    """
    [기능] 메인 비동기 실행 함수. World 초기화 및 시뮬레이션 환경 구축 후 루프 시작.
    [입력] 없음
    [출력] 없음
    """
    # World 인스턴스 초기화
    world = World.instance()
    if world is not None:
        world.stop()
        world.clear_instance()

    await omni.usd.get_context().new_stage_async()

    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    sim = WaferPickup()

    # ── 배경 맵 로드 (smcnd_factory_v12_2.usd) ──────────────────────────────
    background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v12_2.usd"
    add_reference_to_stage(usd_path=background_usd, prim_path="/World")

    setup_camera_and_light()

    await world.reset_async()

    # 로봇 및 웨이퍼 씬 설정
    sim.setup_scene(world)
    await world.reset_async()

    # 로봇 오브젝트 할당
    # 로봇만 world.scene.get_object 사용 (웨이퍼는 USD stage API 직접 접근)
    sim.robots = world.scene.get_object("my_ur10")
    sim.robots.set_world_pose(position=sim.robot_position)

    # World 참조 저장 (Phase 10에서 다음 웨이퍼 로드 시 사용)
    sim._world_ref = world

    # RMPflow 제어기 초기화
    sim.cspace_controller = RMPFlowController(
        name="my_ur10_cspace_controller",
        robot_articulation=sim.robots,
        attach_gripper=True,
    )

    # ── 첫 번째 웨이퍼(wafer1) 로드 ─────────────────────────────────────────
    sim._load_next_wafer(world)

    # ── 초기 환경 구성 디버그 출력 ───────────────────────────────────────────
    actual_pos, _ = sim.robots.get_world_pose()
    wafer_pos, _  = sim._get_wafer_world_pose()
    print("=" * 50)
    print(f"[INIT] Robot set_world_pose target : {sim.robot_position}")
    print(f"[INIT] Robot actual world pose     : {actual_pos}")
    print(f"[INIT] RMPFlow base pose           : {sim.cspace_controller._default_position}")
    print(f"[INIT] First wafer world pose      : {wafer_pos}")
    print(f"[INIT] background_usd              : {background_usd}")
    print("=" * 50)

    # 매 물리 프레임마다 physics_step 콜백 등록
    world.add_physics_callback("sim_step", callback_fn=sim.physics_step)

    await world.play_async()
    print("Simulation started. Processing wafer1 → wafer2 → wafer3 ...")


# 이벤트 루프에 메인 코루틴 등록
asyncio.ensure_future(main())