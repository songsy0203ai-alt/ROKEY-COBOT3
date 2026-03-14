"""
[코드 기능]
NVIDIA Isaac Sim 환경에서 UR10 로봇 팔을 이용해 컨베이어 벨트 위의 큐브를 집어 지정된 위치로 옮기는(Pick and Place) 작업을 시뮬레이션하는 스크립트입니다. RMPflow 알고리즘을 활용하여 로봇의 모션 계획(Motion Planning) 및 제어를 수행합니다.

[입력(Input)]
- 배경, 컨베이어 벨트, UR10 로봇의 USD 파일 경로 및 로봇 초기 관절 상태
- 대상 큐브(DynamicCuboid)의 초기 위치, 크기, 색상 등 환경 구성 변수
- 시뮬레이션 상의 물리적 시간 간격(step_size)

[출력(Output)]
- RMPflow를 통해 계산된 로봇의 목표 관절 위치 (Joint Positions) 및 제어 액션 (로봇에 적용됨)
- 물리 엔진이 적용된 시뮬레이션 환경 내 로봇 및 큐브의 상태 변화 (화면 렌더링)
"""

import numpy as np
import sys
import carb
import asyncio
import omni.usd

# World 및 Stage 관리를 위한 추가 import
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, clear_stage

from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicCuboid

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.prims import GeometryPrim 

# RMPFlowController는 기존 그대로 유지
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
        [입력] 
          - name: 컨트롤러 식별자(문자열)
          - robot_articulation: 제어할 로봇의 Articulation 객체
          - physics_dt: 물리 엔진 시뮬레이션의 시간 간격 (기본값 1/60초)
          - attach_gripper: 그리퍼 부착 여부 (불리언)
        [출력] 초기화된 RMPFlowController 인스턴스 (명시적 반환값 없음)
        """
        # 그리퍼 부착 여부에 따라 RMPflow 설정 파일 로드
        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
        
        # 로드된 설정을 바탕으로 RMPflow 인스턴스 생성
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        # 로봇 Articulation과 RMPflow를 연결하는 정책 객체 생성
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        # 부모 클래스 초기화
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        
        # 로봇의 초기 월드 좌표 및 방향 획득
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        
        # 획득한 초기 자세를 모션 정책에 반영
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        """
        [기능] 컨트롤러의 상태를 초기화
        [입력] 없음
        [출력] 없음 (내부 상태 초기화)
        """
        # 부모 클래스의 리셋 메서드 호출
        mg.MotionPolicyController.reset(self)
        # 로봇의 베이스 포즈를 기본값으로 재설정
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

# BaseSample을 제거하고 일반 클래스로 상태(State)와 콜백을 관리
class UR10PickAndPlaceTask:
    def __init__(self, world):
        """
        [기능] UR10 로봇의 Pick and Place 작업을 위한 환경 및 상태 변수 초기화
        [입력] 
          - world: Isaac Sim의 World 인스턴스
        [출력] 초기화된 UR10PickAndPlaceTask 인스턴스
        """
        self.world = world
        self.BROWN = np.array([0.5, 0.2, 0.1]) # 큐브 색상 정의 (RGB)
        self._brown_cube_position = np.array([-1.5, 0.0, 0.5]) # 큐브 초기 위치 정의
        self.task_phase = 1 # 작업의 현재 단계를 나타내는 변수 (1부터 시작)
        self._wait_counter = 0 # 특정 단계에서의 대기 시간을 계산하는 카운터
        self.robot_position = np.array([1.0, 0.0, 0.0]) # 로봇의 초기 배치 위치
        self.place_position = np.array([1.5, 0.5, 0.05]) # 큐브를 내려놓을 목표 위치
        
        # 콜백 이름 (Script Editor 여러번 실행 시 중복 방지용)
        self.callback_name = "ur10_pick_and_place_step"

    async def setup_and_run(self):
        """
        [기능] 환경 에셋(USD) 로드, 물리 객체 생성 및 시뮬레이션 시작 (비동기 처리)
        [입력] 없음
        [출력] 없음 (비동기 루프 내에서 환경 설정 완료 후 시뮬레이션 실행)
        """
        # 1. 씬 초기화 (Script editor에서 여러 번 누를 때를 대비하여 컨텍스트 초기화)
        await self.world.initialize_simulation_context_async()

        # 2. 씬 세팅 (배경 및 바닥 평면 추가)
        self.background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v9.usd"
        add_reference_to_stage(usd_path=self.background_usd, prim_path="/World/Background")
        self.world.scene.add_default_ground_plane()    

        # Isaac Sim의 기본 에셋 경로 가져오기
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        # 컨베이어 벨트 USD 불러오기 및 Stage에 추가 (수정됨: 문자열 닫는 따옴표 추가)
        conveyor_usd = "/컨베이어/벨트/경로/conveyor.usd"
        add_reference_to_stage(usd_path=conveyor_usd, prim_path="/World/Conveyor")

        # 컨베이어 벨트 위치 수정 (X, Y, Z 좌표를 설정하여 원하는 위치로 이동)
        conveyor_geom = GeometryPrim(prim_path="/World/Conveyor", name="conveyor")
        conveyor_geom.set_local_pose(translation=np.array([0.0, 1.0, 0.0]))
        
        # 3. 로봇 세팅 (UR10 USD 로드 및 Stage 추가)
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        
        # 로봇 그리퍼 Variant를 "Short_Suction" 모델로 설정
        robot_prim.GetVariantSets().GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
        
        # SurfaceGripper(흡착식 그리퍼) 객체 생성 및 연결
        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link", surface_gripper_path="/World/UR10/ee_link/SurfaceGripper"
        )
        
        # 씬에 로봇 매니퓰레이터 객체 추가
        self.robots = self.world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10", name="my_ur10", end_effector_prim_path="/World/UR10/ee_link", gripper=gripper
            )
        )
        # 로봇의 초기 관절 각도(Joint Positions) 설정
        self.robots.set_joints_default_state(positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]))
        
        # 4. 큐브 세팅 (동적 큐브 객체를 씬에 추가)
        self.cube = self.world.scene.add(DynamicCuboid(
            prim_path="/World/BrownCube", 
            name="brown_cube",
            position=self._brown_cube_position, 
            scale=np.array([0.05, 0.05, 0.05]), 
            color=self.BROWN
        ))

        # 5. 시뮬레이션 리셋 및 재생 (물리 엔진 활성화)
        await self.world.reset_async()
        await self.world.play_async()

        # 6. Post Load 세팅 (RMPFlow 컨트롤러 생성 및 로봇 초기 위치 적용)
        self.cspace_controller = RMPFlowController(name="my_ur10_cspace_controller", robot_articulation=self.robots, attach_gripper=True)
        self.robots.set_world_pose(position=self.robot_position)
        
        # 7. Physics Step 콜백 등록 (매 물리 시뮬레이션 스텝마다 physics_step 함수 실행)
        self.task_phase = 1
        self.world.add_physics_callback(self.callback_name, callback_fn=self.physics_step)

    # 기존의 physics_step 로직 (변경 없음)
    def physics_step(self, step_size):
        """
        [기능] 매 시뮬레이션 스텝마다 호출되어 task_phase 상태에 따라 로봇의 행동을 결정하고 제어 명령을 내림
        [입력] 
          - step_size: 현재 물리 스텝의 시간 간격 (float)
        [출력] 없음 (내부적으로 로봇 관절에 action을 적용하여 움직임 발생)
        """
        # Phase 1: 큐브가 특정 X 좌표(>= -0.09)에 도달할 때까지 대기
        if self.task_phase == 1:
            cube_position, cube_orientation = self.cube.get_world_pose()
            current_x_position = cube_position[0]
            if current_x_position >= -0.09:
                print(f"Cube X ({current_x_position}) reached target range (>= -0.0824).")
                self.task_phase = 2

        # Phase 2: 큐브가 목표에 도달한 후, 안정화를 위해 10 스텝 동안 대기
        elif self.task_phase == 2:
            if self._wait_counter < 10:
                self._wait_counter += 1
            else:
                self._brown_cube_position, _ = self.cube.get_world_pose() # 최종 큐브 위치 업데이트
                self.task_phase = 3
                
        # Phase 3: 로봇 엔드 이펙터가 큐브 위(Z=0.4) 위치로 이동
        elif self.task_phase == 3:
            _target_position = self._brown_cube_position.copy() - self.robot_position
            _target_position[2] = 0.4
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0])) # 그리퍼가 아래를 향하도록 자세 설정
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action) # 계산된 관절 명령을 로봇에 적용
            current_joint_positions = self.robots.get_joint_positions()
            
            # 로봇이 목표 관절 위치에 도달했는지 확인 후 다음 Phase로 전환
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 4

        # Phase 4: 로봇 엔드 이펙터가 큐브 바로 위(Z=0.2)로 하강
        elif self.task_phase == 4:
            _target_position = self._brown_cube_position.copy() - self.robot_position
            _target_position[2] = 0.2
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 5

        # Phase 5: 그리퍼 작동 (큐브 잡기)
        elif self.task_phase == 5:
            self.robots.gripper.close() # 그리퍼 닫기 명령
            self.task_phase = 6

        # Phase 6: 큐브를 잡은 상태로 다시 위(Z=0.4)로 상승
        elif self.task_phase == 6:
            _target_position = self._brown_cube_position.copy() - self.robot_position
            _target_position[2] = 0.4
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 7
    
        # Phase 7: 목표 위치(Place Position)로 로봇 이동
        elif self.task_phase == 7:
            _target_position = self.place_position - self.robot_position
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 8

        # Phase 8: 그리퍼 개방 (큐브 놓기)
        elif self.task_phase == 8:
            self.robots.gripper.open() # 그리퍼 열기 명령
            self.task_phase = 9

        # Phase 9: 큐브를 놓은 후 안전 위치(Z=0.5)로 로봇 상승 복귀
        elif self.task_phase == 9:
            _target_position = self.place_position - self.robot_position
            _target_position[2] = 0.5
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_target_position, 
                target_end_effector_orientation=end_effector_orientation
            )
            self.robots.apply_action(action)
            current_joint_positions = self.robots.get_joint_positions()
            
            if np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < 0.001):
                self.cspace_controller.reset()
                self.task_phase = 10 # 전체 작업 종료

# -------------------------------------------------------------------
# Script Editor 전용 실행 파트 (수정됨)
# -------------------------------------------------------------------
import omni.kit.app

async def run_task():
    """
    [기능] 기존 World 인스턴스를 초기화하고 새로운 Task를 안전하게 실행하는 비동기 진입점
    [입력] 없음
    [출력] 없음 (이전 상태 클리어 후 UR10PickAndPlaceTask 구동)
    """
    # 1. 기존 World 인스턴스가 메모리에 남아있다면 완전히 파괴해서 충돌 방지
    if World.instance() is not None:
        World.instance().clear_all_callbacks()
        World.instance().clear_instance()
        
    # 2. 스테이지 지우기 및 한 프레임 대기 (USD 찌꺼기 완벽 제거)
    clear_stage()
    await omni.kit.app.get_app().next_update_async()

    # 3. 깨끗한 상태에서 World 새로 생성
    world = World()

    # 4. Task 인스턴스화 및 실행 (setup_and_run 호출)
    task = UR10PickAndPlaceTask(world)
    await task.setup_and_run()

# 비동기 실행 루프에 등록하여 스크립트 에디터에서 실행 시 즉시 동작하도록 함
asyncio.ensure_future(run_task())