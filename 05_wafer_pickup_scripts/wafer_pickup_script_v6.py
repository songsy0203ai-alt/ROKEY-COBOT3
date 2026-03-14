"""
[코드 기능]: 
1. UR10 로봇이 RMPFlow 컨트롤러를 사용하여 특정 위치의 물체(wafer)를 감지하고 집어 올리는 시뮬레이션 수행.
2. 물체의 X 좌표가 특정 임계값(-1.775)에 도달할 때까지 대기 후, 단계별(Task Phase)로 동작 수행.
3. 그리퍼(SurfaceGripper)를 제어하여 wafer를 흡착하고 지정된 목적지(place_position)로 이동 및 하차.

[Script Editor 실행용 - Isaac Sim 5.0]
: Window > Script Editor 에 붙여넣고 Run 버튼 클릭
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


# ──────────────────────────────────────────────
# RMPFlow 컨트롤러
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 카메라 & 조명 설정
# ──────────────────────────────────────────────
def setup_camera_and_light():
    stage = omni.usd.get_context().get_stage()
    cam_path = "/World/ScriptCamera"
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))

    # look-at 계산 대신 Transform 직접 지정
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

    # 조명은 카메라 위치(마지막 행)에 배치
    light_path = "/World/CameraLight"
    light = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
    light.GetRadiusAttr().Set(1.6)
    light.GetIntensityAttr().Set(70000.0)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    light_xform = UsdGeom.Xformable(light)
    light_xform.ClearXformOpOrder()
    light_xform.AddTranslateOp().Set(Gf.Vec3d(-4.95977, -8.53786, 8.14303))


# ──────────────────────────────────────────────
# 메인 시뮬레이션 클래스
# ──────────────────────────────────────────────
class WaferPickup:
    def __init__(self):
        self.BROWN = np.array([0.5, 0.2, 0.1])
        self._wafer_position = np.array([-2.703093513886345, -8.012578810344934, 2.2778853351258435])
        self.task_phase = 1
        self._wait_counter = 0
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])
        self.place_position = np.array([-2.7030932903289795, -9.049080848693848, 2.149994373321533])
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])

    def setup_scene(self, world: World):
        world.scene.add_default_ground_plane()

        assets_root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"

        # UR10 로봇 로드
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

        # Wafer 생성
        world.scene.add(
            DynamicCylinder(
                prim_path="/World/wafer",
                name="wafer",
                position=self._wafer_position,
                scale=np.array([0.3, 0.3, 0.1]),
                color=self.BROWN,
            )
        )

    def physics_step(self, step_size):
        """매 프레임 호출되는 FSM 로직"""

        # Phase 1: wafer X 좌표가 임계값에 도달하기를 대기
        if self.task_phase == 1:
            wafer_position, _ = self.wafer.get_world_pose()
            current_x_position = wafer_position[0]
            if current_x_position >= -1.82825:
                print(f"[Phase 1] Wafer X ({current_x_position:.4f}) reached the barrier → Phase 2")
                self.task_phase = 2

        # Phase 2: 짧은 안정화 대기
        elif self.task_phase == 2:
            if self._wait_counter < 10:
                self._wait_counter += 1
            else:
                self._brown_wafer_position, _ = self.wafer.get_world_pose()
                print(f"[Phase 2] Wafer stabilized at {self._brown_wafer_position} → Phase 3")
                self.task_phase = 3

        # Phase 3: 물체 위쪽으로 접근
        elif self.task_phase == 3:
            _target_position = self._brown_wafer_position.copy() - self.robot_position
            _target_position[2] = 1.0

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

        # Phase 4: 물체를 잡기 위해 하강
        elif self.task_phase == 4:
            pick_position = self._brown_wafer_position.copy()
            pick_position[0] += 0.0   # 월드 X 보정
            pick_position[1] += 0.0   # 월드 Y 보정

            _target_position = pick_position - self.robot_position
            _target_position[2] = 0.1

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
                print("[Phase 4] Descend done → Phase 5")
                self.cspace_controller.reset()
                self.task_phase = 5

        # Phase 5: 그리퍼 닫기 (흡착)
        elif self.task_phase == 5:
            print("[Phase 5] Gripper close → Phase 6")
            self.robots.gripper.close()
            self.task_phase = 6

        # Phase 6: 물체를 들고 상승
        elif self.task_phase == 6:
            _target_position = self._brown_wafer_position.copy() - self.robot_position
            _target_position[2] = 1.0

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
                print("[Phase 6] Lift done → Phase 7")
                self.cspace_controller.reset()
                self.task_phase = 7

        # Phase 7: 목적지로 이동
        elif self.task_phase == 7:
            _target_position = self.place_position - self.robot_position

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
                print("[Phase 7] Place position reached → Phase 8")
                self.cspace_controller.reset()
                self.task_phase = 8

        # Phase 8: 그리퍼 열기 (내려놓기)
        elif self.task_phase == 8:
            print("[Phase 8] Gripper open → Phase 9")
            self.robots.gripper.open()
            self.task_phase = 9

        # Phase 9: 회피 상승 후 종료
        elif self.task_phase == 9:
            _target_position = self.place_position.copy() - self.robot_position
            _target_position[2] = 0.5

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
                print("[Phase 9] Sequence complete.")
                self.cspace_controller.reset()
                self.task_phase = 10  # 종료


# ──────────────────────────────────────────────
# 진입점: async main
# ──────────────────────────────────────────────
async def main():
    # 기존 월드 정리 (재실행 시 중복 방지)
    world = World.instance()
    if world is not None:
        world.stop()
        world.clear_instance()

    await omni.usd.get_context().new_stage_async()

    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    sim = WaferPickup()

    # ── 1단계: 배경 USD 먼저 스테이지에 추가 ────────────────────
    background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v12.usd"
    add_reference_to_stage(usd_path=background_usd, prim_path="/World/Factory")

    # ── 2단계: 카메라 & 조명 설정 ────────────────────────────────
    setup_camera_and_light()

    # ── 3단계: 1차 reset (배경·카메라·조명 확정) ─────────────────
    await world.reset_async()

    # ── 4단계: 로봇·wafer 씬 구성 ────────────────────────────────
    sim.setup_scene(world)
    await world.reset_async()  # 2차 reset (로봇·wafer 반영)

    sim.wafer  = world.scene.get_object("wafer")
    sim.robots = world.scene.get_object("my_ur10")

    sim.robots.set_world_pose(position=sim.robot_position)

    sim.cspace_controller = RMPFlowController(
        name="my_ur10_cspace_controller",
        robot_articulation=sim.robots,
        attach_gripper=True,
    )

    world.add_physics_callback("sim_step", callback_fn=sim.physics_step)

    await world.play_async()
    print("Simulation started. Waiting for wafer...")


asyncio.ensure_future(main())