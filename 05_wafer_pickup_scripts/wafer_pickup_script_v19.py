"""
[코드 기능]
wafer1 → wafer2 → wafer3 순차 처리 버전.
RigidPrim 버전 호환성 문제를 _WaferProxy(USD 직접 접근)로 우회.
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

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.prims import SingleArticulation


# ──────────────────────────────────────────────────────────────────────────────
# RigidPrim 생성자 인자가 버전마다 달라 TypeError가 발생하는 문제를 우회하기 위해
# USD stage에 직접 접근하는 경량 프록시 클래스를 사용한다.
# get_world_pose()만 지원하면 되므로 이 방식으로 충분하다.
# ──────────────────────────────────────────────────────────────────────────────
class _WaferProxy:
    """
    [기능] 맵에 이미 존재하는 웨이퍼 Prim의 월드 좌표를 읽기 위한 USD 직접 접근 래퍼.
           RigidPrim의 버전별 생성자 차이를 우회한다.
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

        # ── 타임코드 0 고정 → 현재 시뮬레이션 시간으로 교체 ──────────
        # 0으로 고정하면 물리 시뮬레이션 중 웨이퍼가 움직여도
        # 좌표가 갱신되지 않는다. Usd.TimeCode.Default()를 쓰면
        # 매 스텝 최신 변환 행렬을 읽어온다.
        from pxr import Usd
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # ────────────────────────────────────────────────────────────

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

    # 맵에 이미 존재하는 웨이퍼 Prim 경로 목록 (처리 순서)
    WAFER_PRIM_PATHS = [
        "/World/Factory/smcnd_factory_v4/wafers/wafer1",
        "/World/Factory/smcnd_factory_v4/wafers/wafer2",
        "/World/Factory/smcnd_factory_v4/wafers/wafer3",
    ]

    def __init__(self):
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])
        self.place_position = np.array([-2.9288041591644287, -8.730463027954102, 2.124993324279785])

        self._wafer_index = 0
        # _WaferProxy 인스턴스. main()에서 _advance_wafer()로 설정됨.
        # RigidPrim을 쓰지 않으므로 버전 호환성 문제 없음.
        self.wafer = None

        self.task_phase = 1
        self._wait_counter = 0
        self._phase1_lock_counter = 0
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self._debug_printed = False

    def _advance_wafer(self, world: World):
        if self._wafer_index >= len(self.WAFER_PRIM_PATHS):
            return False

        prim_path = self.WAFER_PRIM_PATHS[self._wafer_index]
        self.wafer = _WaferProxy(prim_path)

        # ── 웨이퍼 질량 강제 설정 (그리퍼가 들 수 있도록) ───────────
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.GetMassAttr().Set(0.01)  # 10g으로 설정
            print(f"[WAFER] Mass set to 0.01 kg for {prim_path}")
        # ────────────────────────────────────────────────────────────

        print(f"[WAFER] Active wafer → {prim_path} (index={self._wafer_index})")
        return True
    

    def _boost_gripper_force(self):
        """
        [기능] SurfaceGripper Prim의 흡착력 관련 속성을 USD API로 직접 상향 설정.
            SurfaceGripper 생성자 인자 방식은 버전마다 달라 TypeError가 발생하므로
            생성 후 Prim 속성을 직접 수정하는 방식을 사용한다.
        """
        stage = omni.usd.get_context().get_stage()
        gripper_prim_path = "/World/UR10/ee_link/SurfaceGripper"
        gripper_prim = stage.GetPrimAtPath(gripper_prim_path)

        if not gripper_prim.IsValid():
            print(f"[WARN] _boost_gripper_force: prim not found at {gripper_prim_path}")
            return

        # PhysX SurfaceGripper 속성 직접 설정
        # 속성명은 physxSurfaceGripper:* 네임스페이스 아래에 위치한다
        attr_map = {
            "physxSurfaceGripper:forceLimit":   1.0e2,   # 흡착력 상향(기본값 ~1.0e4 수준)
            "physxSurfaceGripper:torqueLimit":  1.0e4,   # 토크 제한 상향
            "physxSurfaceGripper:gripThreshold": 0.02,   # 흡착 감지 거리 (m)
            "physxSurfaceGripper:retryClose":   True,    # 흡착 실패 시 재시도
        }

        for attr_name, value in attr_map.items():
            attr = gripper_prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(value)
                print(f"[GRIPPER] {attr_name} = {value}")
            else:
                print(f"[WARN] attribute not found: {attr_name}")

        # ── 현재 속성값 전체 출력 (디버그용) ──────────────────────
        print("[GRIPPER] All attributes on SurfaceGripper prim:")
        for attr in gripper_prim.GetAttributes():
            print(f"  {attr.GetName()} = {attr.Get()}")
        # ────────────────────────────────────────────────────────

    def setup_scene(self, world: World):
        world.scene.add_default_ground_plane()

        assets_root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path="/World/UR10")
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")

        # ── 버전 호환을 위해 최소 인자로만 생성 ──────────────────────
        # translate, direction 등 버전별로 다른 인자는 사용하지 않음
        # 흡착력은 아래에서 USD API로 직접 설정한다
        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            surface_gripper_path="/World/UR10/ee_link/SurfaceGripper",
        )
        # ────────────────────────────────────────────────────────────

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

        # ── SurfaceGripper 흡착력을 USD API로 직접 상향 설정 ────────
        # SurfaceGripper Prim의 PhysxSurfaceGripperAPI 속성을 직접 수정한다.
        # 이 방법은 생성자 인자 버전 차이에 무관하게 동작한다.
        self._boost_gripper_force()
        # ────────────────────────────────────────────────────────────

    def _reset_for_next_wafer(self):
        self._wait_counter = 0
        self._phase1_lock_counter = 0
        self._phase8_counter = 0   # ← 추가
        self._debug_printed = False
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self.cspace_controller.reset()
        self.task_phase = 1
        print(f"[NEXT] State reset done. Starting wafer index={self._wafer_index}")

    def physics_step(self, step_size):
        if self.wafer is None:
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

        # Phase 1: 웨이퍼 X축 임계값 도달 대기
        if self.task_phase == 1:
            self._phase1_lock_counter = getattr(self, '_phase1_lock_counter', 0) + 1
            if self._phase1_lock_counter <= 30:
                return

            wafer_position, _ = self.wafer.get_world_pose()
            current_x_position = wafer_position[0]
            if current_x_position >= -1.67499:
                print(f"[Phase 1] Wafer X ({current_x_position:.4f}) reached the barrier → Phase 2")
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

        # Phase 7: Place 위치로 이동
        elif self.task_phase == 7:
            print('[Debugging Log] Phase 7 start')
            _target_position = np.array([-2.7030928134918213, -9.1062, 2.124995470046997])

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

            # 60스텝(약 1초) 동안 open()을 반복 호출한 뒤 무조건 전환
            # SurfaceGripper joint 해제에 필요한 시간을 충분히 확보한다
            if self._phase8_counter >= 60:
                wafer_pos, _ = self.wafer.get_world_pose()
                ee_pos, _ = self.robots.end_effector.get_world_pose()
                dist = np.linalg.norm(wafer_pos - ee_pos)
                print(f"[Phase 8] Released after {self._phase8_counter} steps | dist={dist:.4f} → Phase 9")
                self._phase8_counter = 0
                self.task_phase = 9
            # ────────────────────────────────────────────────────────────

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
                    self.wafer = None  # physics_step 가드 활성화


async def main():
    world = World.instance()
    if world is not None:
        world.stop()
        world.clear_instance()

    await omni.usd.get_context().new_stage_async()

    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    sim = WaferPickup()

    background_usd = "/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v17.usd"
    add_reference_to_stage(usd_path=background_usd, prim_path="/World/Factory")

    setup_camera_and_light()
    await world.reset_async()

    sim.setup_scene(world)
    await world.reset_async()

    sim.robots = world.scene.get_object("my_ur10")
    sim.robots.set_world_pose(position=sim.robot_position)

    # ── 첫 번째 웨이퍼(wafer1) 활성화 ─────────────────────────
    # RigidPrim 대신 _WaferProxy를 사용하므로 TypeError 없음
    sim._advance_wafer(world)
    # ──────────────────────────────────────────────────────────

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


asyncio.ensure_future(main())