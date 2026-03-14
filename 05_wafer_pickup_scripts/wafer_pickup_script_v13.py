"""
[코드 기능]
무한반복, 완성.
Isaac Sim 환경에서 UR10 매니퓰레이터와 흡착 그리퍼를 이용해 동적 원기둥(웨이퍼) 객체를 픽업하고 지정된 위치에 배치하는 작업을 수행하는 물리 기반 시뮬레이션 스크립트입니다. 상태 머신(Phase 1~10)을 기반으로 모션 정책 제어기(RMPflow)를 통해 궤적을 생성합니다.

[입력(Input)]
- 환경 리소스: UR10 로봇 USD, 공장 배경 USD 파일
- 초기 상태값: 로봇의 월드 좌표(robot_position), 웨이퍼의 생성 좌표(_wafer_position), 목표 배치 좌표(place_position)
- 제어 설정값: RMPflow 구성 파일(UR10, RMPflowSuction)

[출력(Output)]
- 로봇 동작: 매 물리 스텝마다 산출되는 로봇의 관절 위치 목표값(Joint Position Action) 및 그리퍼 개폐 명령
- 콘솔 출력: 로봇 및 웨이퍼의 현재 좌표 디버그 정보, 각 작업 단계(Task Phase) 전환 상태
"""

import numpy as np
import asyncio

import omni.usd
import omni.kit.app
import omni.kit.viewport.utility as vp_util

from pxr import Gf, UsdGeom, UsdLux, Sdf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicCylinder

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    # 충돌 회피 및 모션 생성을 담당하는 RMPflow 제어기 클래스
    
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
        # 그리퍼 장착 여부에 따라 다른 RMPflow 설정 파일을 로드함
        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflow"
            )

        # RMPflow 모션 정책 인스턴스화
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        # 로봇 아티큘레이션과 RMPflow를 연결하는 정책 래퍼 생성
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        # 부모 클래스 초기화 호출
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)

        # 로봇의 초기 위치 및 방향을 획득하여 RMPflow 기준 좌표계로 설정함
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


def setup_camera_and_light():
    """
    [기능] 시뮬레이션 스테이지 내 스크립트 기반 카메라 및 구형 조명 생성/배치
    [입력] 없음
    [출력] 없음
    """
    # 현재 활성화된 USD 스테이지 컨텍스트를 가져옴
    stage = omni.usd.get_context().get_stage()
    
    # 카메라 Prim(기본 요소) 경로 지정 및 생성
    cam_path = "/World/ScriptCamera"
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))

    # 카메라 위치 및 회전을 정의하는 4x4 변환 행렬
    transform_matrix = Gf.Matrix4d(
        -0.99978,  0.02106,  0.00066, 0.0,
        -0.01815,  -6.0,     0.48032, 0.0,
        -0.0107,  -0.4802,  -0.87709, 0.0,
        -1.26865,  -8.66478,  5.24915, 1.0,
    )
    # 카메라의 기존 변환 연산을 지우고 새로운 행렬 적용
    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform_matrix)

    # 카메라 렌즈 설정 (수평 조리개 및 초점 거리)
    cam_prim.GetHorizontalApertureAttr().Set(20.955)
    cam_prim.GetFocalLengthAttr().Set(8.0)

    # 활성 뷰포트를 스크립트로 생성한 카메라로 전환
    viewport = vp_util.get_active_viewport()
    if viewport:
        viewport.set_active_camera(cam_path)

    # 조명 Prim 경로 지정 및 구형 조명(SphereLight) 생성
    light_path = "/World/CameraLight"
    light = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
    # 조명 속성 설정 (반경, 강도, 색상)
    light.GetRadiusAttr().Set(1.6)
    light.GetIntensityAttr().Set(70000.0)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # 조명 위치 이동 연산 적용
    light_xform = UsdGeom.Xformable(light)
    light_xform.ClearXformOpOrder()
    light_xform.AddTranslateOp().Set(Gf.Vec3d(-4.95977, -8.53786, 8.14303))


class WaferPickup:
    # 픽앤플레이스 작업의 상태 머신 및 씬 설정을 관리하는 주 작업 클래스

    def __init__(self):
        """
        [기능] 작업에 필요한 좌표 변수 및 상태 머신 변수 초기화
        [입력] 없음
        [출력] 없음
        """
        # 객체 색상 지정
        self.BROWN = np.array([0.5, 0.2, 0.1])
        
        # 위치 좌표 초기화 설정
        self._wafer_position = np.array([-2.703093513886345, -8.012578810344934, 2.2778853351258435]) # 웨이퍼가 최초 생성되는 고정 좌표
        self.task_phase = 1 # 작업 단계 상태 변수 (1부터 시작)
        self._wait_counter = 0 # 지연 대기를 위한 카운터 변수
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])
        # self.place_position = np.array([-2.7030932903289795, -9.049080848693848, 2.149994373321533])
        self.place_position = np.array([-2.9288041591644287, -8.730463027954102, 2.124993324279785])
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285]) # 웨이퍼가 컨베이어 등을 타고 이동한 후 실제로 멈춘 좌표
        self._debug_printed = False  # 디버그 텍스트 중복 출력 방지 플래그

    def setup_scene(self, world: World):
        """
        [기능] 로봇, 그리퍼, 웨이퍼(원기둥) 에셋을 스테이지에 로드하고 초기화
        [입력] 
          - world (World): 객체가 추가될 물리 시뮬레이션 World 인스턴스
        [출력] 없음
        """
        # 기본 지면 추가
        world.scene.add_default_ground_plane()

        # 원격 서버에서 UR10 로봇 에셋 로드 및 스테이지 참조 추가
        assets_root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        
        # 로봇 모델의 그리퍼 변형(Variant)을 짧은 흡착기(Short_Suction)로 설정
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        # 표면 흡착 그리퍼 인스턴스 생성
        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            surface_gripper_path="/World/UR10/ee_link/SurfaceGripper",
        )

        # 씬에 매니퓰레이터(로봇+그리퍼) 추가 및 레퍼런스 저장
        ur10 = world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10",
                name="my_ur10",
                end_effector_prim_path="/World/UR10/ee_link",
                gripper=gripper,
            )
        )
        # 로봇 관절의 초기 각도 설정
        ur10.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )

        # 씬에 픽업 대상인 웨이퍼(동적 원기둥) 객체 추가
        world.scene.add(
            DynamicCylinder(
                prim_path="/World/wafer",
                name="wafer",
                position=self._wafer_position,
                scale=np.array([0.05, 0.05, 0.05]),
                # scale=np.array([0.3, 0.3, 0.1]),
                color=self.BROWN,
            )
        )

    # 무한반복용
    def _reset_state(self):
        self._wafer_count = getattr(self, '_wafer_count', 1) + 1

        # 위치 + 방향 모두 초기화 (기울어진 상태 리셋)
        self.wafer.set_world_pose(
            position=self._wafer_position,
            orientation=np.array([1, 0, 0, 0])  # ← 쿼터니언 단위행렬 (회전 없음)
        )
        self.wafer.set_linear_velocity(np.array([0, 0, 0]))
        self.wafer.set_angular_velocity(np.array([0, 0, 0]))

        self.task_phase = 0
        self._reset_wait_counter = 0
        self._wait_counter = 0
        self._phase1_lock_counter = 0
        self._debug_printed = False
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self.cspace_controller.reset()
        print(f"[RESET] Wafer #{self._wafer_count} repositioned. Waiting for physics init...")


    def physics_step(self, step_size):
        # 물리 백엔드 재초기화 중 발생하는 에러 방어
        try:
            self._physics_step_impl(step_size)
        except Exception as e:
            if "Failed to get DOF" in str(e):
                pass  # 백엔드 초기화 중 무시
            else:
                raise e  # 다른 에러는 그대로 올림
        
    def _physics_step_impl(self, step_size):
        """
        [기능] 매 물리 스텝마다 호출되어 상태 머신(Phase)에 따라 로봇 이동 및 그리퍼 제어 수행
        [입력] 
          - step_size (float): 이전 콜백과 현재 콜백 사이의 경과 시간 (물리 스텝 크기)
        [출력] 없음 (물리 엔진에 직접 Action 인가)
        """

        # ── 디버그: Phase 1 진입 시 1회만 출력 ──
        # 로봇, 웨이퍼 좌표 초기값 검증을 위해 콘솔에 출력
        if not self._debug_printed and self.task_phase == 1:
            robot_actual_pos, robot_actual_ori = self.robots.get_world_pose()
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

        # Phase 0: 웨이퍼 재생성 후 물리 엔진 안정화 대기
        if self.task_phase == 0:
            self.wafer.set_linear_velocity(np.array([0, 0, 0]))
            self.wafer.set_angular_velocity(np.array([0, 0, 0]))
            self.wafer.set_world_pose(
                position=self._wafer_position,
                orientation=np.array([1, 0, 0, 0])
            )

            self._reset_wait_counter = getattr(self, '_reset_wait_counter', 0) + 1

            # ── 디버그: 매 스텝마다 웨이퍼 Y좌표 출력 ──
            wafer_pos, _ = self.wafer.get_world_pose()
            print(f"[Phase 0] step={self._reset_wait_counter} wafer_pos={wafer_pos}")

            if self._reset_wait_counter >= 30:
                print("[Phase 0] Physics stabilized → Phase 1")
                self.task_phase = 1

        # Phase 1: 웨이퍼가 지정된 X축 위치 임계값 이상 도달할 때까지 대기
        elif self.task_phase == 1:

            # Phase 1 진입 초반 30스텝 동안 위치 강제 고정 (컨베이어 안착 대기)
            self._phase1_lock_counter = getattr(self, '_phase1_lock_counter', 0) + 1
            if self._phase1_lock_counter <= 30:
                self.wafer.set_world_pose(
                    position=self._wafer_position,
                    orientation=np.array([1, 0, 0, 0])
                )
                self.wafer.set_linear_velocity(np.array([0, 0, 0]))
                self.wafer.set_angular_velocity(np.array([0, 0, 0]))
                return  # 고정 중에는 트리거 체크 안 함

            wafer_position, _ = self.wafer.get_world_pose()
            current_x_position = wafer_position[0]
            if current_x_position >= -1.575:
                print(f"[Phase 1] Wafer X ({current_x_position:.4f}) reached the barrier → Phase 2")
                self._phase1_lock_counter = 0  # 다음 사이클을 위해 초기화
                self.task_phase = 2

        # Phase 2: 물리 엔진 안정화를 위해 10스텝 대기 후 현재 웨이퍼 위치 저장
        elif self.task_phase == 2:
            if self._wait_counter < 10:
                self._wait_counter += 1
            else:
                self._brown_wafer_position, _ = self.wafer.get_world_pose()
                print(f"[Phase 2] Wafer stabilized at {self._brown_wafer_position} → Phase 3")
                self.task_phase = 3

        # Phase 3: 웨이퍼 위쪽(접근 위치)으로 엔드이펙터 이동
        elif self.task_phase == 3:
            # 하드코딩 대신 실제 웨이퍼 위치 기반으로 동적 계산
            _target_position = self._brown_wafer_position.copy()
            
            # 접근 위치 도달 로그
            print('[Debugging Log] Phase 3 )  configuring target position done')
            print(_target_position)
            
            # Z축 좌표 재설정 (상대적 하강)
            _target_position[2] = 1.91

            # 목표 방위각(오일러각을 쿼터니언으로 변환)
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0])) 
            
            # RMPflow를 통해 관절 제어 명령(Action) 생성
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,      # 목표 위치
                target_end_effector_orientation=end_effector_orientation,  # 목표 방향
            )
            # 산출된 명령을 로봇 관절에 인가
            self.robots.apply_action(action)

            # 현재 관절 위치가 목표 관절 위치에 도달했는지 오차 범위(0.001) 내에서 검사
            current_joint_positions = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001 
                # action.joint_positions : target_end_effector_position, target_end_effector_orientation 이 두 목표값을 향해 "이번 1스텝만큼" 이동할 관절 각도
            ):
                print("[Debugging Log] Phase 3 )  Approach done → Phase 4")
                self.cspace_controller.reset() # 궤적 완료 후 제어기 초기화
                self.task_phase = 4

        # Phase 4: 흡착을 위해 실제 웨이퍼 위치로 하강 이동
        elif self.task_phase == 4:

            # Phase 2에서 저장한 웨이퍼의 월드 좌표를 복사 (원본 보존용)
            pick_position = self._brown_wafer_position.copy()
            
            # RMPflow 입력용으로 월드 좌표 → 로봇 베이스 기준 상대 좌표로 변환
            # (RMPflow는 로봇 베이스를 원점으로 하는 상대 좌표를 입력으로 받음)
            _target_position = pick_position - self.robot_position

            # 엔드이펙터가 아래를 향하도록 방향 설정
            # Y축 기준 90도 회전 → 그리퍼가 수직 하향
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
            print("[Debugging Log] Phase 4 ) end_effector_orientation done")
            
            # RMPflow 제어기에 목표 위치/방향을 전달하여 관절 제어 명령(Action) 생성
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position,       # 목표 위치 (상대 좌표)
                target_end_effector_orientation=end_effector_orientation,  # 목표 방향
            )
            
            # 계산된 관절 명령을 로봇에 인가 (실제 로봇 이동 시작)
            self.robots.apply_action(action)
            print("[Debugging Log] Phase 4 ) action done")

            # 현재 실제 관절 각도 읽기
            current_joint_positions = self.robots.get_joint_positions()
            print(current_joint_positions)
            
            '''
            ===================관계도===================
            [입력]
            end_effector_orientation (엔드이펙터의 목표 방향, 쿼터니언 4개)
            _target_position         (목표 위치, XYZ 3개)
                    ↓
                RMPflow 내부 계산
                (역기구학 + 충돌회피)
                    ↓
            [출력]
            action.joint_positions   (관절 1~6번 각도, 6개)
            '''
            
            # 목표 관절값이 유효하고, 현재 관절값과 목표값의 차이가 모두 0.001 미만이면
            # → 로봇이 목표 위치에 도달했다고 판단
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.1
            ):
                print("[Phase 4] Descend done → Phase 5")
                self.cspace_controller.reset()  # 다음 동작을 위해 제어기 내부 상태 초기화
                self.task_phase = 5             # 그리퍼 흡착 단계로 전환

        # Phase 5: 그리퍼 흡착 활성화
        elif self.task_phase == 5:
            print("[Phase 5] Gripper close → Phase 6") 
            self.robots.gripper.close() # 그리퍼 닫음 명령
            self.task_phase = 6 

        # Phase 6: 웨이퍼를 파지한 상태로 수직 상승
        elif self.task_phase == 6:
            _target_position = self._brown_wafer_position.copy() - self.robot_position
            # Z축 상승을 위한 좌표 설정
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
            ("[Debugging Log] Phase 6 apply_action done")
            

            current_joint_positions = self.robots.get_joint_positions()
            
            if action.joint_positions is not None and np.all(
                np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001
            ):
                print("[Phase 6] Lift done → Phase 7")
                self.cspace_controller.reset()
                self.task_phase = 7
                print('[Debugging Log] Now Phase 7') 

        # Phase 7: 목표 배치 위치(Place Position)로 이동
        elif self.task_phase == 7:
            print('[Debugging Log] Phase 7 start')  
            # BEFORE) _target_position = self.place_position.copy() - self.robot_position
            # AFTER)
            _target_position = np.array([-2.7030928134918213, -9.1062, 2.124995470046997]) # 주운 웨이퍼를 PLACE하는 좌표

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

        # Phase 8: 웨이퍼를 놓기 위해 그리퍼 개방
        elif self.task_phase == 8:
            print("[Phase 8] Gripper open → Phase 9")
            self.robots.gripper.open() # 그리퍼 열림 명령 (흡착 해제)
            self.task_phase = 9

        # Phase 9: 초기 관절 자세로 복귀 후 시퀀스 종료
        elif self.task_phase == 9:
            home_joint_positions = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
            
            from isaacsim.core.utils.types import ArticulationAction
            action = ArticulationAction(joint_positions=home_joint_positions)
            self.robots.apply_action(action)

            current_joint_positions = self.robots.get_joint_positions()
            if np.all(np.abs(current_joint_positions[:6] - home_joint_positions) < 0.01):
                print("[Phase 9] Home position reached. Waiting for next trigger...")
                self._reset_state()  # 상태 초기화 후 Phase 1 대기

        


async def main():
    """
    [기능] 메인 비동기 실행 함수, World 초기화 및 시뮬레이션 환경 구축 후 루프 시작
    [입력] 없음
    [출력] 없음
    """
    # World 인스턴스 초기화. 기존 인스턴스가 존재하면 정지 후 삭제
    world = World.instance()
    if world is not None:
        world.stop()
        world.clear_instance()

    # 새로운 비동기 USD 스테이지 생성
    await omni.usd.get_context().new_stage_async()

    # 1단위를 미터(1.0)로 설정하여 World 인스턴스 재생성
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    # 작업 클래스 인스턴스화
    sim = WaferPickup()

    # 배경 공장 USD 씬 로드 및 스테이지 참조
    # background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v12.usd"
    background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v12.usd"
    add_reference_to_stage(usd_path=background_usd, prim_path="/World/Factory")

    # 카메라 및 조명 셋업 함수 호출
    setup_camera_and_light()

    # 비동기 환경 리셋
    await world.reset_async()

    # 로봇 및 객체 셋업 수행
    sim.setup_scene(world)
    await world.reset_async() # 객체 등록 후 물리 엔진 상태 업데이트를 위해 리셋

    # 씬 등록 객체 불러와서 변수에 할당
    sim.wafer  = world.scene.get_object("wafer")
    sim.robots = world.scene.get_object("my_ur10")

    # 로봇의 기준 월드 좌표 설정
    sim.robots.set_world_pose(position=sim.robot_position)

    # 관절 공간 제어를 담당할 RMPflow 제어기 초기화 할당
    sim.cspace_controller = RMPFlowController(
        name="my_ur10_cspace_controller",
        robot_articulation=sim.robots,
        attach_gripper=True,
    )

    # main() 안에서 아래 한 줄 추가
    sim._world = world

    # ── 핵심 디버그 출력 ──────────────────────────────────────
    # 초기 환경 구성이 완료된 직후 로봇과 객체의 실제 좌표 상태 점검
    actual_pos, actual_ori = sim.robots.get_world_pose()
    wafer_pos, _ = sim.wafer.get_world_pose()
    print("=" * 50)
    print(f"[INIT] Robot set_world_pose target : {sim.robot_position}")
    print(f"[INIT] Robot actual world pose     : {actual_pos}")
    print(f"[INIT] RMPFlow base pose           : {sim.cspace_controller._default_position}")
    print(f"[INIT] Wafer world pose            : {wafer_pos}")
    print("=" * 50)
    # ─────────────────────────────────────────────────────────

    # 매 물리 프레임마다 sim.physics_step 함수가 실행되도록 콜백 등록
    world.add_physics_callback("sim_step", callback_fn=sim.physics_step)

    # 비동기 시뮬레이션 시작
    await world.play_async()
    print("Simulation started. Waiting for wafer...")


# 이벤트 루프에 메인 코루틴 등록하여 실행
asyncio.ensure_future(main())


# ===================================================================================