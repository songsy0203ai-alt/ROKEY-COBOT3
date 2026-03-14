"""
[코드 기능]
NVIDIA Isaac Sim 환경에서 smcnd_factory_v10.usd 맵을 로드하고, 컨베이어 벨트 위의 웨이퍼(Wafer6)를 제어하는 Pick and Place 스크립트입니다. 
지정된 조건(웨이퍼의 X좌표가 -1.775 이하로 감소)이 충족되면 UR10 로봇 팔이 웨이퍼를 집어 초기 위치로 이동시킵니다.

[입력(Input)]
- 환경 USD 경로: /home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v10.usd
- 컨베이어 위치: (-3.062, -5.970, 1.470)
- 로봇 베이스 위치: (-1.69498, -9.02501299679875, 2.1)
- 웨이퍼 초기 위치: (-2.703093513886345, -8.012578810344934, 2.2778853351258435) / 스케일: (0.5, 0.5, 0.1)
- 웨이퍼 설정: Physics(Rigid Body, Collider), Material(OmniPBR_6, Weaker than Descendants)
- 물리적 시간 간격(step_size)

[출력(Output)]
- 조건 충족 시 계산된 RMPflow 기반의 로봇 관절 위치 (Joint Positions) 및 제어 액션 적용
- USD 속성(Physics API, Material Binding)이 동적으로 수정된 웨이퍼 객체의 시뮬레이션 상태 변화
"""

import numpy as np # 수학적 벡터 및 행렬 연산을 위한 라이브러리
import sys # 파이썬 시스템 변수 제어
import carb # Omniverse 기본 로깅 유틸리티
import asyncio # 비동기 프로그래밍 표준 라이브러리
import omni.usd # Omniverse USD 제어 API

# USD 프림의 물리 및 머티리얼 속성을 직접 수정하기 위한 pxr 모듈
from pxr import UsdPhysics, UsdShade 

from isaacsim.core.api import World # 시뮬레이션 환경 총괄 클래스
from isaacsim.core.utils.stage import add_reference_to_stage, clear_stage # 씬 구성 유틸리티
from isaacsim.storage.native import get_assets_root_path # Isaac Sim 에셋 기본 경로 검색
from isaacsim.robot.manipulators.grippers import SurfaceGripper # 흡착식 그리퍼 제어 클래스
from isaacsim.robot.manipulators import SingleManipulator # 매니퓰레이터 제어 클래스

import isaacsim.robot_motion.motion_generation as mg # RMPflow 모션 생성 관련 모듈
from isaacsim.core.utils.rotations import euler_angles_to_quat # 오일러 각도를 쿼터니언으로 변환
from isaacsim.core.prims import SingleArticulation # 로봇 관절 트리 제어
from isaacsim.core.prims import SingleGeometryPrim, RigidPrim # 단일 기하학 및 강체 제어 클래스

class RMPFlowController(mg.MotionPolicyController):
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:
        """
        [기능] RMPFlow 컨트롤러 초기화 및 로봇 모션 정책 설정
        [입력] name, robot_articulation, physics_dt, attach_gripper
        [출력] 초기화된 RMPFlowController 인스턴스
        """
        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflowSuction")
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
        
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        
        (self._default_position, self._default_orientation) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(robot_position=self._default_position, robot_orientation=self._default_orientation)

    def reset(self):
        """
        [기능] 컨트롤러 내부 버퍼 및 상태 초기화
        [입력] 없음
        [출력] 없음
        """
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(robot_position=self._default_position, robot_orientation=self._default_orientation)

class UR10PickAndPlaceTask:
    def __init__(self, world):
        """
        [기능] 웨이퍼 Pick and Place 작업을 위한 맵 기반 환경 변수 및 상태 초기화
        [입력] world (Isaac Sim World 인스턴스)
        [출력] 초기화된 인스턴스
        """
        self.world = world
        self.task_phase = 1 
        self._wait_counter = 0 
        
        # 제공된 좌표값에 기반한 위치 변수 선언
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1]) 
        self._wafer_initial_position = np.array([-2.703093513886345, -8.012578810344934, 2.2778853351258435])
        self.conveyor_position = np.array([-3.0622000674686682, -5.970562922643824, 1.4701799527804797])
        
        # 웨이퍼 복귀 목표 좌표 (초기 좌표와 동일하게 설정)
        self.place_position = self._wafer_initial_position.copy()
        self.callback_name = "ur10_wafer_task_step"

    async def setup_and_run(self):
        """
        [기능] 환경 에셋 로드, USD 속성(물리, 머티리얼) 적용 및 물리 엔진 시뮬레이션 개시
        [입력] 없음
        [출력] 없음
        """
        await self.world.initialize_simulation_context_async()

        # 1. 사용자의 팩토리 맵(v10)을 World 씬의 루트에 추가
        self.background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v10.usd"
        add_reference_to_stage(usd_path=self.background_usd, prim_path="/World/Background")
        self.world.scene.add_default_ground_plane()    

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        # 2. 로봇 셋업 (지정된 좌표에 베이스 배치)
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot_prim.GetVariantSets().GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
        
        gripper = SurfaceGripper(end_effector_prim_path="/World/UR10/ee_link", surface_gripper_path="/World/UR10/ee_link/SurfaceGripper")
        self.robots = self.world.scene.add(SingleManipulator(prim_path="/World/UR10", name="my_ur10", end_effector_prim_path="/World/UR10/ee_link", gripper=gripper))
        self.robots.set_joints_default_state(positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]))

        # 3. 웨이퍼(Wafer6) USD 속성 조작 (Physics 및 Material 적용)
        wafer_prim_path = "/World/Background/smcnd_factory_v4/wafers/wafer6" # Background 하위 참조 기준
        wafer_prim = self.world.stage.GetPrimAtPath(wafer_prim_path)
        
        # 맵에 존재하는 웨이퍼를 RigidPrim으로 읽어오기만 합니다. (위치/스케일 강제 주입 제거)
        self.wafer = self.world.scene.add(RigidPrim(
            prim_path=wafer_prim_path, 
            name="wafer6"
            # position, scale 인자 삭제 -> USD 맵에 있는 원본 위치/크기 사용
        ))

        # 주의: 웨이퍼에 Physics 속성이나 Material 속성이 부족하다면, 
        # 스크립트가 아닌 Isaac Sim GUI에서 smcnd_factory_v10.usd 파일을 열고
        # 직접 속성을 추가한 후 저장해야 엔진 충돌을 피할 수 있습니다.

        # 5. 시뮬레이션 물리 적용을 위한 리셋 및 플레이
        await self.world.reset_async()
        await self.world.play_async()

        # 6. Post Load 세팅
        self.cspace_controller = RMPFlowController(name="my_ur10_cspace_controller", robot_articulation=self.robots, attach_gripper=True)
        self.robots.set_world_pose(position=self.robot_position)
        
        # # 7. 컨베이어 벨트 위치 설정 (맵 내부의 컨베이어 경로. 필요시 실제 맵 구조에 맞춰 수정)
        # conveyor_prim_path = "/World/Background/smcnd_factory_v4/conveyor" 
        # conveyor_geom = SingleGeometryPrim(prim_path=conveyor_prim_path, name="conveyor")
        # # 물리 엔진 플레이 후 위치 설정 적용
        # conveyor_geom.set_world_pose(position=self.conveyor_position)

        # 8. 물리 콜백 등록
        self.task_phase = 1
        self.world.add_physics_callback(self.callback_name, callback_fn=self.physics_step)

    def physics_step(self, step_size):
        """
        [기능] 매 스텝마다 x <= -1.775 트리거를 감지하여 로봇 모션을 제어
        [입력] step_size (물리 시간 간격)
        [출력] 없음 (내부 상태 변경 및 action 전송)
        """
        # 단계 1: 웨이퍼가 -1.775 이하로 진입할 때까지 감지
        if self.task_phase == 1:
            wafer_position, _ = self.wafer.get_world_pose()
            current_x_position = wafer_position[0]
            # 좌표값이 작아지다가 -1.775 이하가 되는 순간을 트리거
            if current_x_position <= -1.775:
                print(f"Wafer X ({current_x_position}) reached trigger point (<= -1.775).")
                self.task_phase = 2

        # 단계 2: 위치 안정화를 위한 10 스텝 대기
        elif self.task_phase == 2:
            if self._wait_counter < 10:
                self._wait_counter += 1
            else:
                self._wafer_target_position, _ = self.wafer.get_world_pose()
                self.task_phase = 3
                
        # 단계 3: 웨이퍼 상단(안전 고도 Z + 0.4)으로 어프로치
        elif self.task_phase == 3:
            _target_position = self._wafer_target_position.copy() - self.robot_position
            _target_position[2] = 0.4 
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            
            action = self.cspace_controller.forward(target_end_effector_position=_target_position, target_end_effector_orientation=end_effector_orientation)
            self.robots.apply_action(action)
            
            if np.all(np.abs(self.robots.get_joint_positions()[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 4

        # 단계 4: 파지를 위한 하강 (고도 Z + 0.2)
        elif self.task_phase == 4:
            _target_position = self._wafer_target_position.copy() - self.robot_position
            _target_position[2] = 0.2
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            
            action = self.cspace_controller.forward(target_end_effector_position=_target_position, target_end_effector_orientation=end_effector_orientation)
            self.robots.apply_action(action)
            
            if np.all(np.abs(self.robots.get_joint_positions()[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 5

        # 단계 5: 그리퍼 흡착 작동
        elif self.task_phase == 5:
            self.robots.gripper.close()
            self.task_phase = 6

        # 단계 6: 웨이퍼를 들고 상단(안전 고도 Z + 0.4)으로 복귀
        elif self.task_phase == 6:
            _target_position = self._wafer_target_position.copy() - self.robot_position
            _target_position[2] = 0.4
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            
            action = self.cspace_controller.forward(target_end_effector_position=_target_position, target_end_effector_orientation=end_effector_orientation)
            self.robots.apply_action(action)
            
            if np.all(np.abs(self.robots.get_joint_positions()[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 7
    
        # 단계 7: 배치 목표 위치(초기 위치)로 이동
        elif self.task_phase == 7:
            _target_position = self.place_position - self.robot_position
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            
            action = self.cspace_controller.forward(target_end_effector_position=_target_position, target_end_effector_orientation=end_effector_orientation)
            self.robots.apply_action(action)
            
            if np.all(np.abs(self.robots.get_joint_positions()[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 8

        # 단계 8: 그리퍼 해제 (웨이퍼 놓기)
        elif self.task_phase == 8:
            self.robots.gripper.open() 
            self.task_phase = 9

        # 단계 9: 픽업 지점 회피를 위한 상단 이동 (Z + 0.5)
        elif self.task_phase == 9:
            _target_position = self.place_position - self.robot_position
            _target_position[2] = 0.5
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            
            action = self.cspace_controller.forward(target_end_effector_position=_target_position, target_end_effector_orientation=end_effector_orientation)
            self.robots.apply_action(action)
            
            if np.all(np.abs(self.robots.get_joint_positions()[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 10 

import omni.kit.app

async def run_task():
    if World.instance() is not None:
        World.instance().clear_all_callbacks()
        World.instance().clear_instance()
        
    clear_stage()
    await omni.kit.app.get_app().next_update_async()

    world = World()
    task = UR10PickAndPlaceTask(world)
    await task.setup_and_run()

asyncio.ensure_future(run_task())