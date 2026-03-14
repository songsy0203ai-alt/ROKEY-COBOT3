# ur10 (6축) 로봇에 흡입기(Gripper)를 부착해서 블록을 흡입(Grip) 해 Pick and Place하는 모션

"""
[코드 기능]
- Isaac Sim 5.0 환경에서 UR10 로봇 팔과 흡착형 그리퍼(Suction Gripper)를 생성하고 제어함.
- RMPflow(리만 모션 정책) 알고리즘을 사용하여 말단 장치(End-effector)의 목표 위치로 역기구학(IK) 기반의 경로를 생성함.
- 갈색 큐브를 인식하여 접근(Pick), 흡착(Grip), 상승(Lift), 해제(Place)하는 시퀀스 제어를 수행함.

[입력(Input)]
- 로봇 설정: UR10 로봇 USD 에셋 경로 및 그리퍼 종류(Short_Suction).
- 환경 설정: 지면(Ground Plane) 및 목표 물체(DynamicCuboid)의 초기 위치 및 크기.
- 제어 파라미터: 목표 말단 장치 좌표(Position) 및 자세(Orientation).

[출력(Output)]
- 시뮬레이션 상의 로봇 관절 각도 변화 및 물리적 상호작용(그리퍼 흡착 및 이동).
- 로봇의 현재 상태와 목표 달성 여부(Boolean).
"""

import numpy as np
import sys
import carb

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import SurfaceGripper
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.core.api.objects import DynamicCuboid

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    """
    RMPFlow 알고리즘을 활용하여 로봇의 모션을 계산하고 제어하는 컨트롤러 클래스
    """
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:
        """
        [Input]
        - name: 컨트롤러의 고유 이름
        - robot_articulation: 제어 대상 로봇 객체
        - physics_dt: 물리 업데이트 시간 간격
        - attach_gripper: 그리퍼 장착 여부에 따른 RMPflow 설정 선택 변수
        """
        # 그리퍼 유무에 따라 적합한 RMPflow 설정 파일 로드
        if attach_gripper:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
        
        # Lula 라이브러리를 사용한 RmpFlow 인스턴스 생성
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        # 로봇 관절과 RMPFlow 정책을 연결
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        # 부모 클래스(MotionPolicyController) 초기화
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        
        # 로봇의 초기 위치 및 방향 저장 (리셋 시 사용)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        
        # RMP 모션 정책에 로봇의 초기 베이스 위치 설정
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        """
        컨트롤러와 로봇 베이스 포즈를 초기 상태로 리셋
        """
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )


class Gripper_UR10(BaseSample):
    """
    Isaac Sim 메인 샘플 클래스: 씬 구성 및 시뮬레이션 루프 관리
    """
    def __init__(self) -> None:
        super().__init__()
        # 색상 및 타겟 큐브 위치 초기화
        self.BROWN = np.array([0.5, 0.2, 0.1])
        self._brown_cube_position = np.array([0.40, 0.0, 0.025])
        self.task_phase = 1 # 시퀀스 제어를 위한 현재 단계 표시
        return

    def setup_scene(self):
        """
        [Output] Isaac Sim 스테이지에 지면, 로봇, 물체 등을 생성함
        """
        world = self.get_world()
        world.scene.add_default_ground_plane()    

        # 에셋 루트 경로 확인
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            simulation_app.close()
            sys.exit()
        
        # UR10 로봇 USD 로드 및 스테이지 추가
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        
        # UR10의 그리퍼를 'Short_Suction' 타입으로 선택
        robot.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
        
        # SurfaceGripper(흡착식) 객체 생성 및 경로 지정
        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link", surface_gripper_path="/World/UR10/ee_link/SurfaceGripper"
        )
        
        # Manipulator 객체로 래핑하여 씬에 추가
        ur10 = world.scene.add(
            SingleManipulator(
                prim_path="/World/UR10", name="my_ur10", end_effector_prim_path="/World/UR10/ee_link", gripper=gripper
            )
        )
        
        # 로봇의 초기 관절 위치 설정
        ur10.set_joints_default_state(positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]))
        
        # 상호작용할 갈색 큐브 생성
        world.scene.add(DynamicCuboid(
            prim_path="/World/BrownCube", 
            name="brown_cube",
            position=self._brown_cube_position, 
            scale=np.array([0.05, 0.05, 0.05]), 
            color=self.BROWN
        ))
        return

    def move_point(self, goal_position: np.ndarray, end_effector_orientation: np.ndarray=np.array([0, np.pi/2, 0])) -> bool:
        """
        [Input]
        - goal_position: 말단 장치가 도달해야 할 3D 좌표 (x, y, z)
        - end_effector_orientation: 오일러 각도 기준 자세 (기본값: 수직 하향)
        
        [Output]
        - is_reached (bool): 로봇 관절이 목표치에 충분히 근접했는지 여부
        """
        # 오일러 각도를 쿼터니언으로 변환
        end_effector_orientation = euler_angles_to_quat(end_effector_orientation)
        
        # RMPFlow를 통해 목표 지점에 도달하기 위한 다음 관절 값 계산
        target_joint_positions = self.cspace_controller.forward(
            target_end_effector_position=goal_position, 
            target_end_effector_orientation=end_effector_orientation
        )
        
        # 계산된 관절 값을 로봇에 적용
        self.robots.apply_action(target_joint_positions)
        
        # 현재 관절 위치와 목표 관절 위치 사이의 오차 계산하여 도달 여부 판단
        current_joint_positions = self.robots.get_joint_positions()
        is_reached = np.all(np.abs(current_joint_positions[:7] - target_joint_positions.joint_positions) < 0.001)
        return is_reached

    async def setup_post_load(self):
        """
        [Output] 시뮬레이션 시작 전 컨트롤러 초기화 및 물리 콜백 등록
        """
        self._world = self.get_world()
        self.robots = self._world.scene.get_object("my_ur10")
        self._cube = self._world.scene.get_object("brown_cube")
        
        # RMPFlow 컨트롤러 인스턴스화
        self.cspace_controller=RMPFlowController(name="my_ur10_cspace_controller", robot_articulation=self.robots, attach_gripper=True)
        
        self.task_phase = 1
        self._goal_reached = False
        
        # 매 물리 스텝마다 physics_step 함수가 실행되도록 등록
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        """
        [Input] step_size: 물리 업데이트 간격
        [Function] 상태 머신을 활용한 로봇 작업 시퀀스 제어
        """
        # Phase 1: 큐브 위치로 로봇 팔 이동
        if self.task_phase == 1:
            cube_position, _ = self._cube.get_world_pose()
            cube_position[2] = 0.045 # 큐브 상단에 흡착하기 위한 높이 조정

            self._goal_reached = self.move_point(cube_position)
            if self._goal_reached:
                self.cspace_controller.reset()
                self.task_phase = 2

        # Phase 2: 그리퍼 작동 (흡착)
        elif self.task_phase == 2:
            self.robots.gripper.close() # Suction 활성화
            self.task_phase = 3

        # Phase 3: 큐브를 위로 들어 올림
        elif self.task_phase == 3:
            cube_position, _ = self._cube.get_world_pose()
            cube_position[2] = 0.4 # 공중으로 들어올릴 높이 설정

            self._goal_reached = self.move_point(cube_position)
            if self._goal_reached:
                self.cspace_controller.reset()
                self.task_phase = 4

        # Phase 4: 그리퍼 해제 (분리)
        elif self.task_phase == 4:
            self.robots.gripper.open() # Suction 비활성화
            self.task_phase = 5
        return