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
# 중첩 클래스 없이 콜백을 DetectionSubscriber 메서드로 직접 정의하여
# self 참조 혼동 문제를 원천 차단한다.
# ──────────────────────────────────────────────────────────────────────────────
class DetectionSubscriber:
    VALID_CLASSES  = {"empty", "none", "scratch", "donut"}
    RESULT_CLASSES = {"none", "scratch", "donut"}

    def __init__(self):
        self.latest_class = "empty"
        self._latched     = False
        self._lock        = threading.Lock()
        self._init_ros()

    # ── ROS2 초기화 ────────────────────────────────────────────────────────
    def _init_ros(self):
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String

            if not rclpy.ok():
                rclpy.init()

            # DetectionSubscriber 인스턴스를 node에 전달해 콜백에서 직접 접근
            outer = self

            class _Node(Node):
                def __init__(self):
                    super().__init__("wafer_detection_subscriber")
                    self.create_subscription(String, "/def_det_result", self._cb, 10)

                def _cb(self, msg):
                    outer._on_message(msg.data)

            self._node = _Node()
            t = threading.Thread(target=self._spin, daemon=True)
            t.start()
            print("[DetectionSubscriber] ROS2 subscriber started on /def_det_result")

        except Exception as e:
            print(f"[DetectionSubscriber] ROS2 init failed: {e}")

    def _spin(self):
        import rclpy
        try:
            rclpy.spin(self._node)
        except Exception as e:
            print(f"[DetectionSubscriber] spin error: {e}")

    # ── 메시지 수신 처리 (DetectionSubscriber 메서드이므로 self가 명확) ────
    def _on_message(self, data: str):
        value = data.strip().lower()
        if value not in self.VALID_CLASSES:
            print(f"[DetectionSubscriber] Unknown class: '{data}' → ignored")
            return

        with self._lock:
            # 래치된 상태에서 empty는 무시
            if self._latched and value == "empty":
                return

            prev = self.latest_class
            self.latest_class = value

            if value in self.RESULT_CLASSES:
                self._latched = True

        if prev != value:
            latch_tag = " [LATCHED]" if self._latched else ""
            print(f"[DetectionSubscriber] Class updated: '{prev}' → '{value}'{latch_tag}")

    # ── 외부 인터페이스 ────────────────────────────────────────────────────
    def get_class(self) -> str:
        with self._lock:
            return self.latest_class

    def reset(self):
        with self._lock:
            self.latest_class = "empty"
            self._latched     = False
        print("[DetectionSubscriber] Reset → 'empty' (latch cleared)")


# ──────────────────────────────────────────────────────────────────────────────
# RigidPrim 버전 호환성 우회용 USD 직접 접근 래퍼
# ──────────────────────────────────────────────────────────────────────────────
class _WaferProxy:
    def __init__(self, prim_path: str):
        self._prim_path = prim_path

    def get_world_pose(self):
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(self._prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"[_WaferProxy] Invalid prim path: {self._prim_path}")
        from pxr import Usd
        xform = UsdGeom.Xformable(prim)
        tf    = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t     = tf.ExtractTranslation()
        q     = tf.ExtractRotationQuat()
        im    = q.GetImaginary()
        return (
            np.array([t[0], t[1], t[2]]),
            np.array([q.GetReal(), im[0], im[1], im[2]]),
        )


class RMPFlowController(mg.MotionPolicyController):

    def __init__(self, name, robot_articulation,
                 physics_dt=1.0/60.0, attach_gripper=False):
        cfg_name = "RMPflowSuction" if attach_gripper else "RMPflow"
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
            "UR10", cfg_name)
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt)
        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp)
        (self._default_position,
         self._default_orientation) = \
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)


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
    _PLACE_POS_NORMAL = np.array([-2.75777, -8.8769,  2.125  ])
    _PLACE_POS_DEFECT = np.array([-0.73097, -8.94011, 2.09002])

    def __init__(self):
        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1])
        self.place_position = np.array([-2.9288041591644287,
                                        -8.730463027954102, 2.124993324279785])
        self._wafer_index   = 0
        self.wafer          = None
        self.task_phase     = 1
        self._wait_counter  = 0
        self._phase1_lock_counter = 0
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self._debug_printed = False
        self._detection_class = "empty"
        self._place_target    = self._PLACE_POS_NORMAL.copy()

    def _advance_wafer(self, world):
        if self._wafer_index >= len(self.WAFER_PRIM_PATHS):
            return False
        prim_path = self.WAFER_PRIM_PATHS[self._wafer_index]
        self.wafer = _WaferProxy(prim_path)
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.GetMassAttr().Set(0.01)
            print(f"[WAFER] Mass set to 0.01 kg for {prim_path}")
        print(f"[WAFER] Active wafer → {prim_path} (index={self._wafer_index})")
        return True

    def _boost_gripper_force(self):
        stage = omni.usd.get_context().get_stage()
        gp    = stage.GetPrimAtPath("/World/UR10/ee_link/SurfaceGripper")
        if not gp.IsValid():
            print("[WARN] SurfaceGripper prim not found")
            return
        for k, v in {
            "physxSurfaceGripper:forceLimit":    1.0e2,
            "physxSurfaceGripper:torqueLimit":   1.0e4,
            "physxSurfaceGripper:gripThreshold": 0.02,
            "physxSurfaceGripper:retryClose":    True,
        }.items():
            a = gp.GetAttribute(k)
            if a.IsValid():
                a.Set(v)
                print(f"[GRIPPER] {k} = {v}")
            else:
                print(f"[WARN] attribute not found: {k}")

    def setup_scene(self, world):
        world.scene.add_default_ground_plane()
        assets_root = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
        robot_prim  = add_reference_to_stage(
            usd_path=assets_root + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
            prim_path="/World/UR10")
        robot_prim.GetVariantSet("Gripper").SetVariantSelection("Short_Suction")
        gripper = SurfaceGripper(
            end_effector_prim_path="/World/UR10/ee_link",
            surface_gripper_path="/World/UR10/ee_link/SurfaceGripper")
        ur10 = world.scene.add(SingleManipulator(
            prim_path="/World/UR10", name="my_ur10",
            end_effector_prim_path="/World/UR10/ee_link", gripper=gripper))
        ur10.set_joints_default_state(
            positions=np.array([-np.pi/2, -np.pi/2, -np.pi/2,
                                 -np.pi/2,  np.pi/2, 0]))
        self._boost_gripper_force()

    def _reset_for_next_wafer(self):
        self._wait_counter        = 0
        self._phase1_lock_counter = 0
        self._phase8_counter      = 0
        self._debug_printed       = False
        self._brown_wafer_position = np.array([-1.83094, -7.92573, 1.94285])
        self._detection_class     = "empty"
        self._place_target        = self._PLACE_POS_NORMAL.copy()
        self._detection_subscriber.reset()
        self.cspace_controller.reset()
        self.task_phase = 1
        print(f"[NEXT] State reset done. Starting wafer index={self._wafer_index}")

    def _update_detection_class(self):
        cls = self._detection_subscriber.get_class()
        if cls == self._detection_class:
            return
        self._detection_class = cls
        if cls == "none":
            self._place_target = self._PLACE_POS_NORMAL.copy()
            print(f"[DETECTION] class='{cls}' → NORMAL place {self._place_target}")
        elif cls in ("scratch", "donut"):
            self._place_target = self._PLACE_POS_DEFECT.copy()
            print(f"[DETECTION] class='{cls}' → DEFECT place {self._place_target}")

    def physics_step(self, step_size):
        if self.wafer is None:
            return
        self._update_detection_class()
        try:
            self._physics_step_impl(step_size)
        except Exception as e:
            if "Failed to get DOF" not in str(e):
                raise e

    def _physics_step_impl(self, step_size):

        if not self._debug_printed and self.task_phase == 1:
            rp, _  = self.robots.get_world_pose()
            wp, _  = self.wafer.get_world_pose()
            rb     = self.cspace_controller._default_position
            print("=" * 50)
            print(f"[DEBUG] Robot actual world pose : {rp}")
            print(f"[DEBUG] RMPFlow base pose       : {rb}")
            print(f"[DEBUG] Wafer actual world pose : {wp}")
            print(f"[DEBUG] self.robot_position     : {self.robot_position}")
            print(f"[DEBUG] wafer - robot_position  : {wp - self.robot_position}")
            print(f"[DEBUG] wafer - rmpflow_base    : {wp - rb}")
            print("=" * 50)
            self._debug_printed = True

        # ── Phase 1: detection 'empty' 해제 대기만 수행 ──────────────────────
        # detection이 none/scratch/donut으로 래치된 시점이 웨이퍼 구역 진입이므로
        # X축 조건 없이 detection class만으로 Phase 2 진입을 결정한다.
        if self.task_phase == 1:
            self._phase1_lock_counter += 1
            if self._phase1_lock_counter <= 30:
                return
            if self._detection_class == "empty":
                return
            wafer_pos, _ = self.wafer.get_world_pose()
            print(f"[Phase 1] detection='{self._detection_class}' "
                  f"| Wafer X={wafer_pos[0]:.4f} → Phase 2")
            self._phase1_lock_counter = 0
            self.task_phase = 2

        # ── Phase 2: 10스텝 대기 후 웨이퍼 위치 저장 ───────────────────────
        elif self.task_phase == 2:
            if self._wait_counter < 10:
                self._wait_counter += 1
            else:
                self._brown_wafer_position, _ = self.wafer.get_world_pose()
                print(f"[Phase 2] Wafer stabilized at {self._brown_wafer_position} → Phase 3")
                self.task_phase = 3

        # ── Phase 3: 웨이퍼 위쪽 접근 ─────────────────────────────────────
        elif self.task_phase == 3:
            _tp = self._brown_wafer_position.copy()
            _tp[2] = 1.82
            ori    = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_tp,
                target_end_effector_orientation=ori)
            self.robots.apply_action(action)
            cjp = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                    np.abs(cjp[:6] - action.joint_positions) < 0.001):
                print("[Phase 3] Approach done → Phase 4")
                self.cspace_controller.reset()
                self.task_phase = 4

        # ── Phase 4: 웨이퍼 위치로 하강 ───────────────────────────────────
        elif self.task_phase == 4:
            _tp  = self._brown_wafer_position.copy() - self.robot_position
            ori  = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_tp,
                target_end_effector_orientation=ori)
            self.robots.apply_action(action)
            cjp = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                    np.abs(cjp[:6] - action.joint_positions) < 0.1):
                print("[Phase 4] Descend done → Phase 5")
                self.cspace_controller.reset()
                self.task_phase = 5

        # ── Phase 5: 그리퍼 흡착 ──────────────────────────────────────────
        elif self.task_phase == 5:
            print("[Phase 5] Gripper close → Phase 6")
            self.robots.gripper.close()
            self.task_phase = 6

        # ── Phase 6: 수직 상승 ────────────────────────────────────────────
        elif self.task_phase == 6:
            _tp    = self._brown_wafer_position.copy() - self.robot_position
            _tp[0] -= 4.0
            _tp[1] -= 2.0
            _tp[2] += 3.0
            ori    = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_tp,
                target_end_effector_orientation=ori)
            self.robots.apply_action(action)
            cjp = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                    np.abs(cjp[:6] - action.joint_positions) < 0.001):
                print("[Phase 6] Lift done → Phase 7")
                self.cspace_controller.reset()
                self.task_phase = 7

        # ── Phase 7: Place 위치로 이동 (detection 분기) ───────────────────
        elif self.task_phase == 7:
            _tp  = self._place_target.copy()
            print(f"[Phase 7] Target={_tp} (detection='{self._detection_class}')")
            ori  = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            action = self.cspace_controller.forward(
                target_end_effector_position=_tp,
                target_end_effector_orientation=ori)
            self.robots.apply_action(action)
            cjp = self.robots.get_joint_positions()
            if action.joint_positions is not None and np.all(
                    np.abs(cjp[:6] - action.joint_positions) < 0.001):
                print("[Phase 7] Place position reached → Phase 8")
                self.cspace_controller.reset()
                self.task_phase = 8

        # ── Phase 8: 그리퍼 개방 ──────────────────────────────────────────
        elif self.task_phase == 8:
            self.robots.gripper.open()
            self._phase8_counter = getattr(self, '_phase8_counter', 0) + 1
            if self._phase8_counter >= 60:
                wp, _ = self.wafer.get_world_pose()
                ep, _ = self.robots.end_effector.get_world_pose()
                print(f"[Phase 8] Released | dist={np.linalg.norm(wp-ep):.4f} → Phase 9")
                self._phase8_counter = 0
                self.task_phase = 9

        # ── Phase 9: 홈 복귀 후 다음 웨이퍼 ──────────────────────────────
        elif self.task_phase == 9:
            home = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0])
            from isaacsim.core.utils.types import ArticulationAction
            self.robots.apply_action(ArticulationAction(joint_positions=home))
            cjp = self.robots.get_joint_positions()
            if np.all(np.abs(cjp[:6] - home) < 0.01):
                print(f"[Phase 9] Home reached. wafer index={self._wafer_index} done.")
                self._wafer_index += 1
                if self._wafer_index < len(self.WAFER_PRIM_PATHS):
                    self._advance_wafer(self._world)
                    self._reset_for_next_wafer()
                    print(f"[NEXT] → Now processing: "
                          f"{self.WAFER_PRIM_PATHS[self._wafer_index - 1]}")
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
    sim._detection_subscriber = DetectionSubscriber()

    add_reference_to_stage(
        usd_path="/home/rokey/cobot3_ws/01_digital_twin_map/smcnd_factory_v17.usd",
        prim_path="/World/Factory")

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
        attach_gripper=True)
    sim._world = world

    ap, _ = sim.robots.get_world_pose()
    wp, _ = sim.wafer.get_world_pose()
    print("=" * 50)
    print(f"[INIT] Robot set_world_pose target : {sim.robot_position}")
    print(f"[INIT] Robot actual world pose     : {ap}")
    print(f"[INIT] RMPFlow base pose           : {sim.cspace_controller._default_position}")
    print(f"[INIT] Wafer world pose (wafer1)   : {wp}")
    print("=" * 50)

    world.add_physics_callback("sim_step", callback_fn=sim.physics_step)
    await world.play_async()
    print("Simulation started. Processing wafer1 → wafer2 → wafer3 ...")
    print("Detection routing: none→NORMAL | scratch/donut→DEFECT")


asyncio.ensure_future(main())