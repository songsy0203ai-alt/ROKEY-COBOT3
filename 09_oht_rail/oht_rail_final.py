"""
[통합 기능]
- 1번 코드의 전체 OHT/POD/컨베이어/재공급/레일 그래프/웹 미리보기 기능 유지
- 1번 코드 안에 있던 기존 UR10 로직은 제거하고, 3번 코드의 결함 분기 개념을 반영한 UR10 로직으로 교체
- 2번 코드와의 연동 규약(/def_det_result = empty | none | scratch | donut)을 그대로 사용
- code_7 OHT가 UNLOAD 구간에서 드롭 이벤트를 만들면, 브리지가 물리 DynamicCylinder 웨이퍼를 생성
- UR10은 /def_det_result 가 empty 가 아닐 때까지 대기한 뒤 픽업
- none 은 NORMAL 적재 위치, scratch/donut 은 DEFECT 적재 위치로 배치

[실행 구조]
- 이 파일은 Isaac Sim 씬 실행용이다.
- YOLO detector 는 별도 ROS2 노드(def_det companion script)로 실행하고 /def_det_result 를 발행하면 된다.
"""


import builtins
import json
import math
import heapq
import os
import random
import time
import asyncio
import threading
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import omni.usd
import omni.kit.commands
import omni.kit.viewport.utility as vp_util
import omni.kit.app
import omni.physx
import omni.timeline

from pxr import Usd, Sdf, UsdGeom, UsdPhysics, Gf, UsdShade, UsdLux

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualCuboid, DynamicCylinder
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot.manipulators import SingleManipulator
    from isaacsim.robot.manipulators.grippers import SurfaceGripper
    import isaacsim.robot_motion.motion_generation as mg
    ISAAC_VERSION = "isaacsim"
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, VisualCuboid, DynamicCylinder
    from omni.isaac.core.prims import SingleArticulation
    from omni.isaac.core.utils.rotations import euler_angles_to_quat
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.manipulators import SingleManipulator
    from omni.isaac.manipulators.grippers import SurfaceGripper
    import omni.isaac.motion_generation as mg
    ISAAC_VERSION = "omni.isaac"

# =============================================================================
# 유틸
# =============================================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def yaw_to_quat(yaw: float) -> np.ndarray:
    # scalar-first quaternion [w, x, y, z]
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis /= n
    s = math.sin(0.5 * angle)
    return np.array([math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    a = np.array(v_from, dtype=float)
    b = np.array(v_to, dtype=float)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    a /= na
    b /= nb
    dot = float(np.dot(a, b))

    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    if dot < -0.999999:
        axis = np.cross(a, np.array([0.0, 0.0, 1.0], dtype=float))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=float))
        return quat_normalize(quat_from_axis_angle(axis, math.pi))

    axis = np.cross(a, b)
    return quat_normalize(np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=float))


def rotate_local(local: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]],
        dtype=float,
    )


def _stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


def _stage_prim_valid(path: str) -> bool:
    prim = _stage().GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def _unsubscribe_previous_physics_callback():
    sub = globals().get("_graph_oht_subscription", None)
    if sub is not None:
        try:
            sub.unsubscribe()
        except Exception:
            pass
        globals()["_graph_oht_subscription"] = None

    sub = getattr(builtins, "_graph_oht_subscription", None)
    if sub is not None:
        try:
            sub.unsubscribe()
        except Exception:
            pass
        builtins._graph_oht_subscription = None


def _delete_prims_if_exist(paths: List[str]):
    stage = _stage()
    existing = [p for p in paths if stage.GetPrimAtPath(p).IsValid()]
    if existing:
        omni.kit.commands.execute("DeletePrims", paths=existing)


def spawn_fixed_box(world: World, prim_path: str, name: str, pos, scale, color, q=None):
    obj = world.scene.add(
        FixedCuboid(
            prim_path=prim_path,
            name=name,
            position=np.array(pos, dtype=float),
            scale=np.array(scale, dtype=float),
            color=np.array(color, dtype=float),
        )
    )
    if q is not None:
        obj.set_world_pose(position=np.array(pos, dtype=float), orientation=np.array(q, dtype=float))
    return obj


def spawn_visual_box(world: World, prim_path: str, name: str, pos, scale, color):
    return world.scene.add(
        VisualCuboid(
            prim_path=prim_path,
            name=name,
            position=np.array(pos, dtype=float),
            scale=np.array(scale, dtype=float),
            color=np.array(color, dtype=float),
        )
    )


def local_point_to_world(root_path: str, local_xyz: Tuple[float, float, float]) -> np.ndarray:
    stage = _stage()
    prim = stage.GetPrimAtPath(root_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"로컬->월드 변환용 prim 경로가 잘못되었습니다: {root_path}")

    local = Gf.Vec3d(float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2]))
    world_mtx = omni.usd.get_world_transform_matrix(prim)
    world = world_mtx.Transform(local)
    return np.array([float(world[0]), float(world[1]), float(world[2])], dtype=float)


def resolve_map_root_path(cfg) -> str:
    return cfg.map_root_path if cfg.map_root_path else "/World/UserMap"


def resolve_drop_world_pos(cfg) -> np.ndarray:
    if cfg.use_local_conveyor_drop:
        root_path = resolve_map_root_path(cfg)
        drop_world = local_point_to_world(root_path, cfg.conveyor_local_pos)
        drop_world[2] += cfg.conveyor_drop_z_offset
        return drop_world
    return np.array(cfg.drop_world_pos, dtype=float)


# =============================================================================
# 이미지 텍스처 헬퍼
# =============================================================================
_TEXTURE_MATERIAL_CACHE = {}


def _sanitize_name(text: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in text)


def resolve_image_files(path_or_dir: str) -> List[str]:
    if not path_or_dir:
        return []

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    if os.path.isfile(path_or_dir):
        ext = os.path.splitext(path_or_dir)[1].lower()
        return [path_or_dir] if ext in exts else []

    if os.path.isdir(path_or_dir):
        files = []
        for name in sorted(os.listdir(path_or_dir)):
            full = os.path.join(path_or_dir, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
                files.append(full)
        return files

    return []


def _ensure_scope(path: str):
    stage = _stage()
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return prim
    return UsdGeom.Scope.Define(stage, path).GetPrim()


def get_or_create_texture_material(image_file: str) -> UsdShade.Material:
    stage = _stage()
    if image_file in _TEXTURE_MATERIAL_CACHE:
        return _TEXTURE_MATERIAL_CACHE[image_file]

    _ensure_scope("/World/Looks")

    safe_base = _sanitize_name(os.path.splitext(os.path.basename(image_file))[0])
    safe_hash = str(abs(hash(image_file)))[:10]
    mat_path = f"/World/Looks/WaferTex_{safe_base}_{safe_hash}"

    material = UsdShade.Material.Get(stage, mat_path)
    if not material or not material.GetPrim().IsValid():
        material = UsdShade.Material.Define(stage, mat_path)

        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        tex = UsdShade.Shader.Define(stage, f"{mat_path}/DiffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image_file))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")

        st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/PrimvarReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    _TEXTURE_MATERIAL_CACHE[image_file] = material
    return material


def bind_material_to_prim(prim_path: str, material: UsdShade.Material):
    stage = _stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(material)


def create_textured_plane(prim_path: str, image_file: str):
    stage = _stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Plane",
            prim_path=prim_path,
        )
        prim = stage.GetPrimAtPath(prim_path)

    material = get_or_create_texture_material(image_file)
    bind_material_to_prim(prim_path, material)
    return prim


def set_prim_pose_scale(prim_path: str, pos: np.ndarray, yaw: float, scale_xyz: Tuple[float, float, float]):
    stage = _stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return

    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate((float(pos[0]), float(pos[1]), float(pos[2])))
    api.SetRotate((0.0, 0.0, math.degrees(yaw)), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale((float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))


def set_prim_pose_scale_rpy(
    prim_path: str,
    pos: np.ndarray,
    rpy_deg: Tuple[float, float, float],
    scale_xyz: Tuple[float, float, float],
):
    stage = _stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return

    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate((float(pos[0]), float(pos[1]), float(pos[2])))
    api.SetRotate(
        (float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    api.SetScale((float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))


# =============================================================================
# 설정
# =============================================================================
@dataclass
class IntegrationConfig:
    map_usd_path: Optional[str] = None
    map_root_path: Optional[str] = "/World/UserMap"

    # 레일 높이
    use_manual_rail_world_z: bool = True
    manual_rail_world_z: float = 9.38
    use_manual_rail_height_from_floor: bool = False
    manual_rail_height_from_floor: float = 2.55

    # 루프 외곽
    manual_rect_mode: bool = True
    manual_left_x: float = -20.425
    manual_right_x: float = 20.575
    manual_top_y: float = -8.00
    manual_bottom_y: float = -30.20

    # station 수동 기준
    manual_load_x: float = 1.0
    manual_unload_x: float = -1.0
    load_station_edge: Optional[str] = None
    unload_station_edge: Optional[str] = None

    # OHT
    num_ohts: int = 3
    oht_speed: float = 2.2
    oht_safe_distance: float = 3.6

    # pick/drop
    enable_pick_drop_cycle: bool = True
    pickup_use_world_xy_for_load_node: bool = True
    pickup_world_pos: Tuple[float, float, float] = (-10.0, -8.0, 0.7)

    drop_use_world_xy_for_unload_node: bool = True
    drop_world_pos: Tuple[float, float, float] = (10.0, -8.0, 1.10)

    use_local_conveyor_drop: bool = True
    conveyor_local_pos: Tuple[float, float, float] = (-2.0, -8.0, 1.4)
    conveyor_drop_z_offset: float = 0.7

    pickup_dwell_s: float = 0.5
    drop_dwell_s: float = 0.5
    drop_display_s: float = 1.5
    recycle_wafer_after_drop: bool = True
    head_target_clearance: float = 0.02
    min_hoist_offset: float = -12.0

    # hoist
    hoist_top_offset: float = -0.08
    hoist_hold_offset: float = -0.08
    hoist_speed: float = 2.0

    # pickup zone
    build_pickup_transfer_zone: bool = True

    # POD / 내부 이미지
    wafer_image_path: str = "/home/rokey/work/wafer/image"
    pod_internal_wafer_count: int = 5
    pod_internal_wafer_size: Tuple[float, float, float] = (0.26, 0.26, 1.0)

    pod_body_size: Tuple[float, float, float] = (0.56, 0.56, 0.36)
    pod_lid_size: Tuple[float, float, float] = (0.70, 0.70, 0.07)
    pod_grip_clearance: float = 0.05
    pod_carried_offset_z: float = -0.24

    # 컨베이어 재공급
    pickup_refill_delay_s: float = 10.0
    pickup_conveyor_speed: float = 0.38

    visualize_rect_debug: bool = False
    auto_print_stage_paths: bool = False
    debug_graph: bool = True


# =============================================================================
# 레일 그래프
# =============================================================================
@dataclass
class RailNode:
    name: str
    pos: np.ndarray
    kind: str = "normal"


@dataclass
class RailEdge:
    name: str
    start: str
    end: str
    enabled: bool = True
    powered: bool = True
    length: float = 1.0
    meta: dict = field(default_factory=dict)


def bbox_cache() -> UsdGeom.BBoxCache:
    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )


def world_bbox(prim: Usd.Prim, cache: UsdGeom.BBoxCache) -> Tuple[np.ndarray, np.ndarray]:
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    return (
        np.array([float(mn[0]), float(mn[1]), float(mn[2])], dtype=float),
        np.array([float(mx[0]), float(mx[1]), float(mx[2])], dtype=float),
    )


class RailGraphSystem:
    def __init__(self, rail_z: float = 4.8):
        self.rail_z = rail_z
        self.nodes: Dict[str, RailNode] = {}
        self.edges: Dict[str, RailEdge] = {}
        self.outgoing_edges: Dict[str, List[str]] = {}
        self.loop_edge_order: List[str] = []
        self.edge_progress_offset: Dict[str, float] = {}
        self.node_progress_offset: Dict[str, float] = {}
        self.total_loop_length: float = 1.0

    def add_node(self, name: str, pos_xyz: np.ndarray, kind: str = "normal"):
        self.nodes[name] = RailNode(name=name, pos=np.array(pos_xyz, dtype=float), kind=kind)
        self.outgoing_edges.setdefault(name, [])

    def add_edge(self, name: str, start: str, end: str, meta: Optional[dict] = None):
        p0 = self.nodes[start].pos
        p1 = self.nodes[end].pos
        length = max(float(np.linalg.norm(p1 - p0)), 1e-6)
        edge = RailEdge(name=name, start=start, end=end, length=length, meta=meta or {})
        self.edges[name] = edge
        self.outgoing_edges[start].append(name)

    def sample_edge(self, edge_name: str, t: float) -> np.ndarray:
        edge = self.edges[edge_name]
        return lerp(self.nodes[edge.start].pos, self.nodes[edge.end].pos, t)

    def edge_yaw(self, edge_name: str) -> float:
        e = self.edges[edge_name]
        d = self.nodes[e.end].pos - self.nodes[e.start].pos
        return math.atan2(d[1], d[0])

    def finalize_loop(self, edge_order: List[str]):
        self.loop_edge_order = edge_order[:]
        self.edge_progress_offset.clear()
        self.node_progress_offset.clear()

        s = 0.0
        for edge_name in self.loop_edge_order:
            edge = self.edges[edge_name]
            self.edge_progress_offset[edge_name] = s
            self.node_progress_offset[edge.start] = s
            s += edge.length

        self.total_loop_length = max(s, 1e-6)

    def progress_of(self, current_node: Optional[str], current_edge: Optional[str], edge_t: float) -> float:
        if current_edge is not None:
            return (self.edge_progress_offset.get(current_edge, 0.0) + edge_t * self.edges[current_edge].length) % self.total_loop_length
        if current_node is not None:
            return self.node_progress_offset.get(current_node, 0.0) % self.total_loop_length
        return 0.0

    def build_rail_visuals(self, world: World, dense_spacing: float = 0.26):
        for edge_name in self.loop_edge_order:
            edge = self.edges[edge_name]
            p0 = self.nodes[edge.start].pos
            p1 = self.nodes[edge.end].pos

            tangent = p1 - p0
            norm = float(np.linalg.norm(tangent[:2]))
            if norm < 1e-6:
                tangent = np.array([1.0, 0.0, 0.0], dtype=float)
                norm = 1.0
            tangent = tangent / norm
            side = np.array([-tangent[1], tangent[0], 0.0], dtype=float)

            yaw = self.edge_yaw(edge_name)
            rail_q = yaw_to_quat(yaw)
            sleeper_q = yaw_to_quat(yaw + math.pi * 0.5)

            n_seg = max(4, int(math.ceil(edge.length / dense_spacing)))
            seg_len = edge.length / n_seg

            for i in range(n_seg):
                t = (i + 0.5) / n_seg
                p = self.sample_edge(edge_name, t)

                for lane_sign in (-1.0, 1.0):
                    spawn_fixed_box(
                        world,
                        f"/World/OHTInfra/Rails/{edge_name}/Rail_{i}_{int(lane_sign > 0)}",
                        f"{edge_name}_rail_{i}_{int(lane_sign > 0)}",
                        p + side * 0.13 * lane_sign,
                        [seg_len * 1.02, 0.040, 0.040],
                        [0.28, 0.28, 0.30],
                        rail_q,
                    )

                spawn_fixed_box(
                    world,
                    f"/World/OHTInfra/Rails/{edge_name}/Bus_{i}",
                    f"{edge_name}_bus_{i}",
                    p + np.array([0.0, 0.0, 0.045]),
                    [seg_len * 1.01, 0.020, 0.018],
                    [0.52, 0.40, 0.18],
                    rail_q,
                )

                if i % 2 == 0:
                    spawn_fixed_box(
                        world,
                        f"/World/OHTInfra/Rails/{edge_name}/Sleeper_{i}",
                        f"{edge_name}_sleeper_{i}",
                        p + np.array([0.0, 0.0, -0.055]),
                        [0.08, 0.34, 0.035],
                        [0.34, 0.22, 0.10],
                        sleeper_q,
                    )


# =============================================================================
# pickup zone helper
# =============================================================================
def resolve_pickup_world_pos_from_layout(cfg: IntegrationConfig, layout: dict) -> np.ndarray:
    if cfg.pickup_use_world_xy_for_load_node:
        load_pos = layout["load_pos"]
        return np.array(
            [float(load_pos[0]), float(load_pos[1]), float(cfg.pickup_world_pos[2])],
            dtype=float,
        )
    return np.array(cfg.pickup_world_pos, dtype=float)


def station_edge_yaw(edge: str) -> float:
    if edge == "top":
        return 0.0
    if edge == "right":
        return -math.pi * 0.5
    if edge == "bottom":
        return math.pi
    if edge == "left":
        return math.pi * 0.5
    return 0.0


def place_local(base_world: np.ndarray, yaw: float, local_xyz: Tuple[float, float, float]) -> np.ndarray:
    return base_world + rotate_local(np.array(local_xyz, dtype=float), yaw)


def spawn_oriented_fixed_box(
    world: World,
    prim_path: str,
    name: str,
    pos,
    scale,
    color,
    yaw: float = 0.0,
):
    return spawn_fixed_box(
        world,
        prim_path,
        name,
        pos,
        scale,
        color,
        q=yaw_to_quat(yaw),
    )


# =============================================================================
# POD / 컨베이어 공급
# =============================================================================
class PodAssembly:
    def __init__(
        self,
        world: World,
        base_path: str,
        name_prefix: str,
        body_size: Tuple[float, float, float],
        lid_size: Tuple[float, float, float],
        cfg: IntegrationConfig,
        enable_inner_images: bool = False,
        max_inner_planes: int = 0,
    ):
        self.world = world
        self.base_path = base_path
        self.name_prefix = name_prefix
        self.body_size = np.array(body_size, dtype=float)
        self.lid_size = np.array(lid_size, dtype=float)
        self.cfg = cfg
        self.enable_inner_images = enable_inner_images
        self.max_inner_planes = max(0, int(max_inner_planes))

        bx, by, bz = self.body_size

        self.parts = []
        self.slot_parts = []
        self.internal_planes = []
        self.current_inner_images: List[str] = []

        wall_t = 0.025
        side_h = bz * 0.86
        side_z = -0.02

        spec = [
            ("Bottom",      [0.0, 0.0, -bz * 0.5 + 0.025], [bx * 0.94, by * 0.94, 0.05], [0.06, 0.06, 0.07]),
            ("BackWall",    [0.0, by * 0.5 - wall_t * 0.5, side_z], [bx, wall_t, side_h], [0.07, 0.07, 0.08]),
            ("LeftWall",    [-bx * 0.5 + wall_t * 0.5, 0.0, side_z], [wall_t, by, side_h], [0.07, 0.07, 0.08]),
            ("RightWall",   [bx * 0.5 - wall_t * 0.5, 0.0, side_z], [wall_t, by, side_h], [0.07, 0.07, 0.08]),
            ("FrontLow",    [0.0, -by * 0.5 + wall_t * 0.5, -bz * 0.26], [bx, wall_t, bz * 0.18], [0.07, 0.07, 0.08]),
            ("TopRailL",    [0.0, by * 0.5 - 0.05, bz * 0.34], [bx * 0.82, 0.04, 0.04], [0.12, 0.12, 0.14]),
            ("TopRailR",    [0.0, -by * 0.5 + 0.05, bz * 0.34], [bx * 0.82, 0.04, 0.04], [0.12, 0.12, 0.14]),
            ("LatchFL",     [bx * 0.33, by * 0.5 - 0.05, bz * 0.28], [0.06, 0.04, 0.06], [0.86, 0.87, 0.90]),
            ("LatchRL",     [-bx * 0.33, by * 0.5 - 0.05, bz * 0.28], [0.06, 0.04, 0.06], [0.86, 0.87, 0.90]),
            ("LatchFR",     [bx * 0.33, -by * 0.5 + 0.05, bz * 0.28], [0.06, 0.04, 0.06], [0.86, 0.87, 0.90]),
            ("LatchRR",     [-bx * 0.33, -by * 0.5 + 0.05, bz * 0.28], [0.06, 0.04, 0.06], [0.86, 0.87, 0.90]),
            ("FrontRibL",   [-bx * 0.26, -by * 0.5 + 0.025, -0.03], [0.03, 0.03, bz * 0.68], [0.11, 0.11, 0.12]),
            ("FrontRibC",   [0.0, -by * 0.5 + 0.025, -0.03], [0.03, 0.03, bz * 0.68], [0.11, 0.11, 0.12]),
            ("FrontRibR",   [bx * 0.26, -by * 0.5 + 0.025, -0.03], [0.03, 0.03, bz * 0.68], [0.11, 0.11, 0.12]),
        ]

        for part_name, offset, scale, color in spec:
            part = spawn_visual_box(
                world,
                f"{base_path}/{part_name}",
                f"{name_prefix}_{part_name.lower()}",
                np.array([0.0, 0.0, -100.0]),
                scale,
                color,
            )
            self.parts.append((part, np.array(offset, dtype=float)))

        slot_depths = np.linspace(by * 0.18, -by * 0.08, max(1, self.max_inner_planes))
        for i, local_y in enumerate(slot_depths):
            for side_name, sx in [("L", -bx * 0.40), ("R", bx * 0.40)]:
                part = spawn_visual_box(
                    world,
                    f"{base_path}/Slots/Slot_{side_name}_{i:02d}",
                    f"{name_prefix}_slot_{side_name.lower()}_{i:02d}",
                    np.array([0.0, 0.0, -100.0]),
                    [0.012, 0.018, bz * 0.58],
                    [0.20, 0.20, 0.22],
                )
                self.slot_parts.append(
                    {
                        "part": part,
                        "offset": np.array([sx, float(local_y), -0.02], dtype=float),
                    }
                )

        if self.enable_inner_images and self.max_inner_planes > 0:
            image_pool = resolve_image_files(cfg.wafer_image_path)
            placeholder = image_pool[0] if image_pool else None

            depth_list = np.linspace(by * 0.18, -by * 0.08, self.max_inner_planes)

            for i, local_y in enumerate(depth_list):
                prim_path = f"{base_path}/InnerWafers/WaferTex_{i:02d}"

                if placeholder is not None:
                    create_textured_plane(prim_path, placeholder)
                else:
                    omni.kit.commands.execute(
                        "CreateMeshPrimWithDefaultXform",
                        prim_type="Plane",
                        prim_path=prim_path,
                    )

                self.internal_planes.append(
                    {
                        "prim_path": prim_path,
                        "offset": np.array([0.0, float(local_y), -0.02], dtype=float),
                        "scale": cfg.pod_internal_wafer_size,
                        "visible": False,
                    }
                )

    def set_inner_images(self, image_files: List[str]):
        self.current_inner_images = list(image_files[: self.max_inner_planes])

        for i, item in enumerate(self.internal_planes):
            if i < len(self.current_inner_images):
                create_textured_plane(item["prim_path"], self.current_inner_images[i])
                item["visible"] = True
            else:
                item["visible"] = False

    def set_pose(self, center: np.ndarray, yaw: float = 0.0, visible: bool = True):
        hidden = np.array([0.0, 0.0, -100.0], dtype=float)
        q_hidden = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        if not visible:
            for part, _ in self.parts:
                part.set_world_pose(position=hidden, orientation=q_hidden)
            for item in self.slot_parts:
                item["part"].set_world_pose(position=hidden, orientation=q_hidden)
            for item in self.internal_planes:
                set_prim_pose_scale_rpy(item["prim_path"], hidden, (0.0, 0.0, 0.0), item["scale"])
            return

        q = yaw_to_quat(yaw)

        for part, offset in self.parts:
            p = center + rotate_local(offset, yaw)
            part.set_world_pose(position=p, orientation=q)

        for item in self.slot_parts:
            p = center + rotate_local(item["offset"], yaw)
            item["part"].set_world_pose(position=p, orientation=q)

        for item in self.internal_planes:
            if item["visible"]:
                p = center + rotate_local(item["offset"], yaw)
                set_prim_pose_scale_rpy(
                    item["prim_path"],
                    p,
                    (90.0, 0.0, math.degrees(yaw)),
                    item["scale"],
                )
            else:
                set_prim_pose_scale_rpy(item["prim_path"], hidden, (0.0, 0.0, 0.0), item["scale"])


class PickupPodConveyorManager:
    def __init__(
        self,
        world: World,
        base_path: str,
        pickup_pos: np.ndarray,
        yaw: float,
        pod_body_size: Tuple[float, float, float],
        pod_lid_size: Tuple[float, float, float],
        grip_clearance: float,
        cfg: IntegrationConfig,
    ):
        self.world = world
        self.base_path = base_path
        self.pickup_pos = np.array(pickup_pos, dtype=float)
        self.yaw = float(yaw)
        self.body_size = np.array(pod_body_size, dtype=float)
        self.lid_size = np.array(pod_lid_size, dtype=float)
        self.grip_clearance = float(grip_clearance)
        self.cfg = cfg

        self.image_pool = resolve_image_files(cfg.wafer_image_path)
        self.inner_count = max(1, int(cfg.pod_internal_wafer_count))
        self.refill_delay_s = float(cfg.pickup_refill_delay_s)
        self.conveyor_speed = float(cfg.pickup_conveyor_speed)

        self.ready_center = self.pickup_pos.copy()
        self.start_center = place_local(self.pickup_pos, self.yaw, (-2.35, 0.08, 0.0))

        self.current_center = self.ready_center.copy()
        self.current_images: List[str] = []
        self.state = "READY"
        self.refill_timer = 0.0

        self.pod = PodAssembly(
            world=world,
            base_path=f"{base_path}/LivePod",
            name_prefix="live_pickup_pod",
            body_size=pod_body_size,
            lid_size=pod_lid_size,
            cfg=cfg,
            enable_inner_images=True,
            max_inner_planes=self.inner_count,
        )

        self.current_images = self._sample_random_images()
        self.pod.set_inner_images(self.current_images)
        self.pod.set_pose(self.current_center, self.yaw, visible=True)

    def _sample_random_images(self) -> List[str]:
        if not self.image_pool:
            return []
        k = min(self.inner_count, len(self.image_pool))
        return random.sample(self.image_pool, k)

    def _grip_pos_from_center(self, center: np.ndarray) -> np.ndarray:
        z = (
            float(center[2])
            + float(self.body_size[2]) * 0.5
            + 0.03
            + self.grip_clearance
        )
        return np.array([float(center[0]), float(center[1]), z], dtype=float)

    def has_stock(self) -> bool:
        return self.state == "READY"

    def get_pickup_world_pos(self) -> Optional[np.ndarray]:
        if self.state != "READY":
            return None
        return self._grip_pos_from_center(self.ready_center)

    def consume_one(self) -> Optional[List[str]]:
        if self.state != "READY":
            return None

        picked_images = list(self.current_images)

        self.pod.set_pose(np.array([0.0, 0.0, -100.0]), self.yaw, visible=False)
        self.current_images = []
        self.state = "WAIT_REFILL"
        self.refill_timer = self.refill_delay_s

        return picked_images

    def get_count(self) -> int:
        return 1 if self.state == "READY" else 0

    def update(self, dt: float):
        if self.state == "WAIT_REFILL":
            self.refill_timer = max(0.0, self.refill_timer - dt)
            if self.refill_timer <= 0.0:
                self.current_images = self._sample_random_images()
                self.pod.set_inner_images(self.current_images)
                self.current_center = self.start_center.copy()
                self.pod.set_pose(self.current_center, self.yaw, visible=True)
                self.state = "MOVING"

        elif self.state == "MOVING":
            vec = self.ready_center - self.current_center
            dist = float(np.linalg.norm(vec))
            step = self.conveyor_speed * dt

            if dist < 1e-6 or dist <= step:
                self.current_center = self.ready_center.copy()
                self.state = "READY"
            else:
                self.current_center += (vec / dist) * step

            self.pod.set_pose(self.current_center, self.yaw, visible=True)

        elif self.state == "READY":
            self.pod.set_pose(self.ready_center, self.yaw, visible=True)


# =============================================================================
# Controller
# =============================================================================
class CentralController:
    def __init__(self, graph):
        self.graph = graph
        self.station_owner = {
            "LOAD": None,
            "UNLOAD": None,
        }
        self.pickup_stack: Optional[PickupPodConveyorManager] = None

    def bind_pickup_stack(self, pickup_stack: PickupPodConveyorManager):
        self.pickup_stack = pickup_stack

    def update(self, dt: float):
        if self.pickup_stack is not None:
            self.pickup_stack.update(dt)

    def has_pickup_stock(self) -> bool:
        if self.pickup_stack is None:
            return False
        return self.pickup_stack.has_stock()

    def get_pickup_world_pos(self, fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if self.pickup_stack is not None:
            pos = self.pickup_stack.get_pickup_world_pos()
            if pos is not None:
                return pos
        if fallback is None:
            return None
        return np.array(fallback, dtype=float)

    def consume_pickup_pod(self) -> Optional[List[str]]:
        if self.pickup_stack is None:
            return None
        return self.pickup_stack.consume_one()

    def pickup_remaining_count(self) -> int:
        if self.pickup_stack is None:
            return 0
        return self.pickup_stack.get_count()

    def plan_route(self, start_node: str, goal_node: str) -> List[str]:
        pq = [(0.0, start_node, [])]
        best = {start_node: 0.0}

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal_node:
                return path

            if cost > best.get(node, float("inf")):
                continue

            for edge_name in self.graph.outgoing_edges.get(node, []):
                edge = self.graph.edges[edge_name]
                if not edge.enabled or not edge.powered:
                    continue

                nxt = edge.end
                ncost = cost + edge.length
                if ncost < best.get(nxt, float("inf")):
                    best[nxt] = ncost
                    heapq.heappush(pq, (ncost, nxt, path + [edge_name]))

        return []

    def request_edge_entry(self, oht_id: str, edge_name: str) -> Tuple[bool, str]:
        return True, "OK"

    def leave_edge(self, oht_id: str, edge_name: str):
        return None

    def request_station(self, oht_id: str, station_name: str) -> Tuple[bool, str]:
        owner = self.station_owner.get(station_name, None)
        if owner is None or owner == oht_id:
            self.station_owner[station_name] = oht_id
            return True, "OK"
        return False, f"BUSY:{owner}"

    def release_station(self, oht_id: str, station_name: str):
        owner = self.station_owner.get(station_name, None)
        if owner == oht_id:
            self.station_owner[station_name] = None


# =============================================================================
# OHT 외형
# =============================================================================
BODY_PARTS = [
    ("TopPlate", [0.00, 0.00, 0.12], [0.82, 0.28, 0.04], [0.94, 0.95, 0.96]),
    ("TopBeam", [0.00, 0.00, 0.05], [0.94, 0.24, 0.05], [0.82, 0.83, 0.85]),
    ("TopBeam2", [0.00, 0.00, -0.01], [0.90, 0.18, 0.04], [0.68, 0.70, 0.73]),
    ("RailGuideL", [0.00, 0.18, 0.02], [0.94, 0.03, 0.03], [0.82, 0.83, 0.85]),
    ("RailGuideR", [0.00, -0.18, 0.02], [0.94, 0.03, 0.03], [0.82, 0.83, 0.85]),
    ("RedBar", [0.00, 0.00, 0.135], [0.80, 0.02, 0.015], [0.78, 0.18, 0.16]),
    ("BlueBar", [0.00, 0.00, -0.045], [0.76, 0.02, 0.015], [0.13, 0.20, 0.82]),
    ("CabinetL", [0.00, 0.27, -0.52], [0.18, 0.14, 0.98], [0.94, 0.95, 0.96]),
    ("CabinetR", [0.00, -0.27, -0.52], [0.18, 0.14, 0.98], [0.94, 0.95, 0.96]),
    ("FanL1", [0.02, -0.355, -0.34], [0.13, 0.02, 0.16], [0.15, 0.15, 0.17]),
    ("FanL2", [0.02, -0.355, -0.56], [0.13, 0.02, 0.16], [0.15, 0.15, 0.17]),
    ("SensorPod", [0.12, 0.355, -0.38], [0.16, 0.025, 0.18], [0.82, 0.83, 0.85]),
    ("MechBase", [0.00, 0.00, -0.22], [0.30, 0.20, 0.16], [0.82, 0.83, 0.85]),
    ("MechMid", [0.08, 0.00, -0.36], [0.20, 0.16, 0.16], [0.68, 0.70, 0.73]),
    ("ArmL", [0.03, 0.10, -0.46], [0.16, 0.02, 0.04], [0.18, 0.78, 0.80]),
    ("ArmR", [0.03, -0.10, -0.46], [0.16, 0.02, 0.04], [0.18, 0.78, 0.80]),
]

WHEEL_PARTS = [
    ("Wheel1", [-0.39, 0.18, 0.03], [0.08, 0.08, 0.08], [0.82, 0.70, 0.24]),
    ("Wheel2", [-0.39, -0.18, 0.03], [0.08, 0.08, 0.08], [0.82, 0.70, 0.24]),
    ("Wheel3", [0.39, 0.18, 0.03], [0.08, 0.08, 0.08], [0.82, 0.70, 0.24]),
    ("Wheel4", [0.39, -0.18, 0.03], [0.08, 0.08, 0.08], [0.82, 0.70, 0.24]),
]

LIFT_PARTS = [
    ("GripTopPlate", [0.00, 0.00, 0.08], [0.54, 0.34, 0.03], [0.95, 0.96, 0.98]),
    ("GripBeamL",    [0.00, 0.18, 0.03], [0.46, 0.02, 0.03], [0.14, 0.14, 0.16]),
    ("GripBeamR",    [0.00, -0.18, 0.03], [0.46, 0.02, 0.03], [0.14, 0.14, 0.16]),

    ("GripPostFL", [0.19, 0.14, -0.10], [0.02, 0.02, 0.30], [0.20, 0.20, 0.22]),
    ("GripPostFR", [0.19, -0.14, -0.10], [0.02, 0.02, 0.30], [0.20, 0.20, 0.22]),
    ("GripPostRL", [-0.19, 0.14, -0.10], [0.02, 0.02, 0.30], [0.20, 0.20, 0.22]),
    ("GripPostRR", [-0.19, -0.14, -0.10], [0.02, 0.02, 0.30], [0.20, 0.20, 0.22]),

    ("GripLatchL", [0.00, 0.18, -0.22], [0.46, 0.02, 0.05], [0.92, 0.92, 0.94]),
    ("GripLatchR", [0.00, -0.18, -0.22], [0.46, 0.02, 0.05], [0.92, 0.92, 0.94]),
]


class GraphOHT:
    def __init__(
        self,
        world,
        graph,
        controller,
        oht_id,
        color,
        start_node,
        move_speed,
        safe_distance,
        cfg: IntegrationConfig,
        enable_transport: bool = True,
    ):
        self.world = world
        self.graph = graph
        self.controller = controller
        self.id = oht_id
        self.color = color
        self.cfg = cfg
        self.enable_transport = enable_transport

        self.root_path = f"/World/OHTFleet/{self.id}"
        self.body_path = f"{self.root_path}/Body"

        self.fleet = []
        self.current_node = start_node
        self.current_edge = None
        self.edge_t = 0.0
        self.route = []
        self.route_index = 0

        self.move_speed = move_speed
        self.safe_distance = safe_distance
        self.edge_enter_cooldown = 0.0

        self.body_pos = self.graph.nodes[start_node].pos.copy()
        self.body_yaw = 0.0
        self.body_to_anchor = 0.18

        self.hoist_top_offset = cfg.hoist_top_offset
        self.current_hoist_offset = cfg.hoist_hold_offset
        self.target_hoist_offset = cfg.hoist_hold_offset
        self.hoist_speed = cfg.hoist_speed

        if cfg.pickup_use_world_xy_for_load_node:
            load_node_xy = self.graph.nodes["LOAD"].pos
            self.pickup_world_pos = np.array(
                [float(load_node_xy[0]), float(load_node_xy[1]), float(cfg.pickup_world_pos[2])],
                dtype=float,
            )
        else:
            self.pickup_world_pos = np.array(cfg.pickup_world_pos, dtype=float)

        drop_world = resolve_drop_world_pos(cfg)
        if cfg.drop_use_world_xy_for_unload_node:
            unload_node_xy = self.graph.nodes["UNLOAD"].pos
            drop_world[0] = float(unload_node_xy[0])
            drop_world[1] = float(unload_node_xy[1])
        self.drop_world_pos = np.array(drop_world, dtype=float)

        self.state = "WAIT_AT_NODE"
        self.target_station = "LOAD"
        self.carrying = False
        self.station_lock = None

        self.wafer_state = "AT_PICKUP"
        self.drop_display_timer = 0.0
        self.timer = 0.0

        self.body = spawn_visual_box(
            world, f"{self.root_path}/Body", f"{self.id}_body",
            self.body_pos, [0.08, 0.08, 0.08], [0.10, 0.10, 0.12]
        )
        self.hoist = spawn_visual_box(
            world, f"{self.root_path}/Hoist", f"{self.id}_hoist",
            self.body_pos + np.array([0.0, 0.0, -0.18]), [0.18, 0.18, 0.12], [0.82, 0.84, 0.86]
        )
        self.pipe = spawn_visual_box(
            world, f"{self.root_path}/Pipe", f"{self.id}_pipe",
            self.body_pos + np.array([0.0, 0.0, -0.42]), [0.035, 0.035, 0.45], [0.72, 0.74, 0.78]
        )
        self.head = spawn_visual_box(
            world, f"{self.root_path}/Head", f"{self.id}_head",
            self.body_pos + np.array([0.0, 0.0, -0.62]), [0.16, 0.16, 0.05], [0.16, 0.16, 0.18]
        )

        self.cargo = PodAssembly(
            world=world,
            base_path=f"{self.root_path}/Cargo",
            name_prefix=f"{self.id}_cargo",
            body_size=cfg.pod_body_size,
            lid_size=cfg.pod_lid_size,
            cfg=cfg,
            enable_inner_images=True,
            max_inner_planes=cfg.pod_internal_wafer_count,
        )

        self.body_parts = []
        self.lift_parts = []
        self.telescopic_parts = {}
        self.fold_links = []

        self.camera_path = f"{self.root_path}/Body/OHTCamera"
        self._build_shell()
        self._build_oht_camera()
        self._ensure_route()
        self._update_visuals(0.0)

    def _alive(self):
        return _stage_prim_valid(self.body_path)

    def _add_body_part(self, name, offset, scale, color):
        if name in ("RedBar", "BlueBar"):
            color = self.color.tolist()
        part = spawn_visual_box(
            self.world,
            f"{self.root_path}/BodyShell/{name}",
            f"{self.id}_{name.lower()}",
            self.body_pos + np.array(offset),
            scale,
            color,
        )
        self.body_parts.append((part, np.array(offset, dtype=float)))

    def _add_lift_part(self, name, offset, scale, color):
        part = spawn_visual_box(
            self.world,
            f"{self.root_path}/LiftShell/{name}",
            f"{self.id}_{name.lower()}",
            self.body_pos + np.array(offset),
            scale,
            color,
        )
        self.lift_parts.append((part, np.array(offset, dtype=float)))

    def _add_telescopic(self, name, scale, color):
        self.telescopic_parts[name] = spawn_visual_box(
            self.world,
            f"{self.root_path}/Telescopic/{name}",
            f"{self.id}_{name.lower()}",
            self.body_pos,
            scale,
            color,
        )

    def _add_fold_link(self, name, top_local, bottom_local, ty, tz, color):
        part = spawn_visual_box(
            self.world,
            f"{self.root_path}/FoldLinks/{name}",
            f"{self.id}_{name.lower()}",
            self.body_pos,
            [0.2, ty, tz],
            color,
        )
        self.fold_links.append(
            {
                "part": part,
                "top": np.array(top_local, dtype=float),
                "bottom": np.array(bottom_local, dtype=float),
                "ty": ty,
                "tz": tz,
            }
        )

    def _build_shell(self):
        for row in BODY_PARTS + WHEEL_PARTS:
            self._add_body_part(*row)
        for row in LIFT_PARTS:
            self._add_lift_part(*row)

        self._add_telescopic("ShellOuter", [0.18, 0.18, 0.24], [0.68, 0.70, 0.73])
        self._add_telescopic("ShellMid", [0.14, 0.14, 0.22], [0.82, 0.83, 0.85])
        self._add_telescopic("ShellInner", [0.10, 0.10, 0.20], [0.94, 0.95, 0.96])
        self._add_telescopic("GuideL", [0.04, 0.12, 0.20], [0.68, 0.70, 0.73])
        self._add_telescopic("GuideR", [0.04, 0.12, 0.20], [0.68, 0.70, 0.73])

        for args in [
            ("FoldLinkLF", [0.08, 0.11, -0.20], [0.10, 0.13, -0.02], 0.025, 0.025, [0.18, 0.78, 0.80]),
            ("FoldLinkLR", [-0.08, 0.11, -0.20], [-0.10, 0.13, -0.02], 0.025, 0.025, [0.18, 0.78, 0.80]),
            ("FoldLinkRF", [0.08, -0.11, -0.20], [0.10, -0.13, -0.02], 0.025, 0.025, [0.18, 0.78, 0.80]),
            ("FoldLinkRR", [-0.08, -0.11, -0.20], [-0.10, -0.13, -0.02], 0.025, 0.025, [0.18, 0.78, 0.80]),
        ]:
            self._add_fold_link(*args)

    def _build_oht_camera(self):
        stage = _stage()
        cam = UsdGeom.Camera.Define(stage, Sdf.Path(self.camera_path))
        xform = UsdGeom.XformCommonAPI(cam.GetPrim())
        # OHT 전면 하단에 가깝게 두고, 전방(+X)을 보면서 화면의 위쪽이 월드 +Z가 되도록 정렬
        # 기본 USD 카메라는 -Z를 바라보고 +Y가 화면 위쪽이므로,
        # XYZ 회전 (84, 0, -90)로 진행방향을 향하게 하고 약간 아래(-Z)로 숙인다.
        xform.SetTranslate((0.40, 0.0, -0.04))
        xform.SetRotate((84.0, 0.0, -90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetFocalLengthAttr().Set(8.0)

    def get_camera_path(self) -> str:
        return self.camera_path

    def _position_on_graph(self) -> np.ndarray:
        if self.current_edge is not None:
            return self.graph.sample_edge(self.current_edge, self.edge_t)
        return self.graph.nodes[self.current_node].pos

    def _yaw_on_graph(self) -> float:
        if self.current_edge is not None:
            return self.graph.edge_yaw(self.current_edge)
        if self.current_node is not None and self.route_index < len(self.route):
            return self.graph.edge_yaw(self.route[self.route_index])
        return self.body_yaw

    def _loop_progress(self) -> float:
        return self.graph.progress_of(self.current_node, self.current_edge, self.edge_t)

    def _is_blocked_by_front_oht(self) -> bool:
        my_s = self._loop_progress()
        for other in self.fleet:
            if other.id == self.id:
                continue
            other_s = other._loop_progress()
            gap = (other_s - my_s) % self.graph.total_loop_length
            if 1e-4 < gap < self.safe_distance:
                return True
        return False

    def _current_pickup_target(self) -> Optional[np.ndarray]:
        pos = self.controller.get_pickup_world_pos(self.pickup_world_pos)
        if pos is None:
            return None
        return np.array(pos, dtype=float)

    def _target_head_z(self, target_world_pos: np.ndarray) -> float:
        return float(target_world_pos[2] + self.cfg.head_target_clearance)

    def _hoist_offset_for_target(self, target_world_pos: np.ndarray) -> float:
        desired = self._target_head_z(target_world_pos) - float(self.body_pos[2]) + 0.12
        return clamp(desired, self.cfg.min_hoist_offset, self.hoist_top_offset)

    def _head_matches_target(self, target_world_pos: np.ndarray) -> bool:
        head_pos, _ = self.head.get_world_pose()
        tx, ty, _ = target_world_pos
        target_head_z = self._target_head_z(target_world_pos)
        return (
            abs(float(head_pos[0]) - float(tx)) < 0.10
            and abs(float(head_pos[1]) - float(ty)) < 0.10
            and abs(float(head_pos[2]) - float(target_head_z)) < 0.08
        )

    def _ensure_route(self):
        if self.current_node is None:
            return
        if self.target_station == self.current_node:
            self.route = []
            self.route_index = 0
            return
        self.route = self.controller.plan_route(self.current_node, self.target_station)
        self.route_index = 0

    def _try_enter_next_edge(self):
        if self.current_node is None:
            return False
        if self.route_index >= len(self.route):
            self._ensure_route()
        if self.route_index >= len(self.route):
            self.state = "WAIT_NO_ROUTE"
            return False

        next_edge = self.route[self.route_index]
        ok, _ = self.controller.request_edge_entry(self.id, next_edge)
        if ok:
            self.current_edge = next_edge
            self.current_node = None
            self.edge_t = 0.0
            self.state = "MOVING"
            return True
        return False

    def _update_telescopic(self, body_q, anchor_top, head_center):
        z_top = anchor_top[2] - 0.03
        z_bottom = head_center[2] + 0.06
        span = max(0.16, z_top - z_bottom)

        outer_len = clamp(0.24 + 0.08 * span, 0.20, 0.70)
        mid_len = clamp(0.22 + 0.10 * span, 0.18, 0.92)
        inner_len = clamp(span - (outer_len - 0.08) - (mid_len - 0.10), 0.18, 1.20)
        guide_len = clamp(0.24 + 0.10 * span, 0.22, 1.00)

        outer_center = np.array([self.body_pos[0], self.body_pos[1], z_top - outer_len * 0.5], dtype=float)
        mid_top = z_top - (outer_len - 0.08)
        mid_center = np.array([self.body_pos[0], self.body_pos[1], mid_top - mid_len * 0.5], dtype=float)
        inner_center = np.array([self.body_pos[0], self.body_pos[1], z_bottom + inner_len * 0.5], dtype=float)
        guide_center = np.array([self.body_pos[0], self.body_pos[1], 0.5 * (z_top + z_bottom)], dtype=float)

        self.telescopic_parts["ShellOuter"].set_world_pose(position=outer_center, orientation=body_q)
        self.telescopic_parts["ShellOuter"].set_local_scale(np.array([0.18, 0.18, outer_len]))

        self.telescopic_parts["ShellMid"].set_world_pose(position=mid_center, orientation=body_q)
        self.telescopic_parts["ShellMid"].set_local_scale(np.array([0.14, 0.14, mid_len]))

        self.telescopic_parts["ShellInner"].set_world_pose(position=inner_center, orientation=body_q)
        self.telescopic_parts["ShellInner"].set_local_scale(np.array([0.10, 0.10, inner_len]))

        for name, sign in [("GuideL", 1.0), ("GuideR", -1.0)]:
            gp = guide_center + rotate_local(np.array([0.0, 0.11 * sign, 0.0]), self.body_yaw)
            self.telescopic_parts[name].set_world_pose(position=gp, orientation=body_q)
            self.telescopic_parts[name].set_local_scale(np.array([0.04, 0.12, guide_len]))

    def _update_fold_links(self, head_center):
        for item in self.fold_links:
            top_world = self.body_pos + rotate_local(item["top"], self.body_yaw)
            bottom_world = head_center + rotate_local(item["bottom"], self.body_yaw)
            vec = bottom_world - top_world
            length = max(0.05, float(np.linalg.norm(vec)))
            center = 0.5 * (top_world + bottom_world)
            q = quat_from_two_vectors(np.array([1.0, 0.0, 0.0]), vec)
            item["part"].set_world_pose(position=center, orientation=q)
            item["part"].set_local_scale(np.array([length, item["ty"], item["tz"]]))

    def _update_visuals(self, dt: float):
        if not self._alive():
            return

        self.body_pos = self._position_on_graph()
        self.body_yaw = self._yaw_on_graph()
        body_q = yaw_to_quat(self.body_yaw)

        diff = self.target_hoist_offset - self.current_hoist_offset
        step = self.hoist_speed * dt
        if abs(diff) <= step:
            self.current_hoist_offset = self.target_hoist_offset
        else:
            self.current_hoist_offset += step if diff > 0.0 else -step

        hoist_center = self.body_pos + np.array([0.0, 0.0, self.current_hoist_offset])
        anchor_top = self.body_pos + np.array([0.0, 0.0, -self.body_to_anchor])
        head_center = hoist_center + np.array([0.0, 0.0, -0.12])
        pipe_center = 0.5 * (anchor_top + head_center)
        pipe_length = max(0.12, abs(anchor_top[2] - head_center[2]))

        self.body.set_world_pose(position=self.body_pos, orientation=body_q)
        for part, offset in self.body_parts:
            part.set_world_pose(position=self.body_pos + rotate_local(offset, self.body_yaw), orientation=body_q)

        self.hoist.set_world_pose(position=hoist_center, orientation=body_q)
        self.pipe.set_world_pose(position=pipe_center, orientation=body_q)
        self.pipe.set_local_scale(np.array([0.035, 0.035, pipe_length]))
        self.head.set_world_pose(position=head_center, orientation=body_q)

        self._update_telescopic(body_q, anchor_top, head_center)
        self._update_fold_links(head_center)

        for part, offset in self.lift_parts:
            part.set_world_pose(position=head_center + rotate_local(offset, self.body_yaw), orientation=body_q)

        if self.wafer_state == "AT_PICKUP":
            self.cargo.set_pose(np.array([0.0, 0.0, -100.0]), self.body_yaw, visible=False)

        elif self.wafer_state == "CARRIED":
            cargo_center = head_center + np.array([0.0, 0.0, self.cfg.pod_carried_offset_z], dtype=float)
            self.cargo.set_pose(cargo_center, self.body_yaw, visible=True)

        elif self.wafer_state == "AT_DROP":
            self.cargo.set_pose(self.drop_world_pos, self.body_yaw, visible=True)

        else:
            self.cargo.set_pose(np.array([0.0, 0.0, -100.0]), self.body_yaw, visible=False)

    def update(self, dt: float):
        if not self._alive():
            return

        if self.wafer_state == "AT_DROP" and self.drop_display_timer > 0.0:
            self.drop_display_timer = max(0.0, self.drop_display_timer - dt)
            if self.drop_display_timer <= 0.0 and self.cfg.recycle_wafer_after_drop:
                self.wafer_state = "AT_PICKUP"

        if self.edge_enter_cooldown > 0.0:
            self.edge_enter_cooldown = max(0.0, self.edge_enter_cooldown - dt)

        is_blocked = self._is_blocked_by_front_oht()
        current_speed = 0.0 if is_blocked else self.move_speed

        if self.cfg.enable_pick_drop_cycle and self.enable_transport:
            if self.state == "LOWER_PICK":
                pickup_target = self._current_pickup_target()
                if pickup_target is None:
                    if self.station_lock == "LOAD":
                        self.controller.release_station(self.id, "LOAD")
                        self.station_lock = None
                    self.state = "WAIT_AT_NODE"
                else:
                    self.target_hoist_offset = self._hoist_offset_for_target(pickup_target)
                    if self._head_matches_target(pickup_target):
                        self.timer = 0.0
                        self.state = "PICK_WAIT"

            elif self.state == "PICK_WAIT":
                self.timer += dt
                if self.timer >= self.cfg.pickup_dwell_s:
                    picked_images = self.controller.consume_pickup_pod()

                    if picked_images is None:
                        if self.station_lock == "LOAD":
                            self.controller.release_station(self.id, "LOAD")
                            self.station_lock = None
                        self.state = "WAIT_AT_NODE"
                    else:
                        self.cargo.set_inner_images(picked_images)
                        self.carrying = True
                        self.wafer_state = "CARRIED"
                        self.target_hoist_offset = self.hoist_top_offset
                        self.state = "RAISE_AFTER_PICK"

            elif self.state == "RAISE_AFTER_PICK":
                self.target_hoist_offset = self.hoist_top_offset
                if abs(self.current_hoist_offset - self.hoist_top_offset) < 0.03:
                    if self.station_lock == "LOAD":
                        self.controller.release_station(self.id, "LOAD")
                        self.station_lock = None

                    self.target_station = "UNLOAD"
                    self._ensure_route()
                    self.edge_enter_cooldown = 0.05
                    self.state = "WAIT_AT_NODE"

            elif self.state == "LOWER_DROP":
                self.target_hoist_offset = self._hoist_offset_for_target(self.drop_world_pos)
                if self._head_matches_target(self.drop_world_pos):
                    self.timer = 0.0
                    self.state = "DROP_WAIT"

            elif self.state == "DROP_WAIT":
                self.timer += dt
                if self.timer >= self.cfg.drop_dwell_s:
                    self.carrying = False
                    self.wafer_state = "AT_DROP"
                    self.drop_display_timer = self.cfg.drop_display_s
                    self.target_hoist_offset = self.hoist_top_offset
                    self.state = "RAISE_AFTER_DROP"

            elif self.state == "RAISE_AFTER_DROP":
                self.target_hoist_offset = self.hoist_top_offset
                if abs(self.current_hoist_offset - self.hoist_top_offset) < 0.03:
                    if self.station_lock == "UNLOAD":
                        self.controller.release_station(self.id, "UNLOAD")
                        self.station_lock = None

                    self.target_station = "LOAD"
                    self._ensure_route()
                    self.edge_enter_cooldown = 0.05
                    self.state = "WAIT_AT_NODE"

            else:
                self.target_hoist_offset = self.hoist_top_offset

                if self.current_edge is not None:
                    edge = self.graph.edges[self.current_edge]
                    delta_t = (current_speed * dt) / max(edge.length, 1e-6)
                    self.edge_t += delta_t
                    if self.edge_t >= 1.0:
                        self.edge_t = 1.0
                        finished_edge = self.current_edge
                        self.controller.leave_edge(self.id, finished_edge)
                        self.current_node = self.graph.edges[finished_edge].end
                        self.current_edge = None
                        self.route_index += 1
                        self.edge_enter_cooldown = 0.05
                        self.state = "WAIT_AT_NODE"
                else:
                    if not is_blocked:
                        if (
                            self.current_node == "LOAD"
                            and (not self.carrying)
                            and self.wafer_state == "AT_PICKUP"
                            and self.controller.has_pickup_stock()
                        ):
                            ok, _ = self.controller.request_station(self.id, "LOAD")
                            if ok:
                                self.station_lock = "LOAD"
                                self.state = "LOWER_PICK"

                        elif self.current_node == "UNLOAD" and self.carrying:
                            ok, _ = self.controller.request_station(self.id, "UNLOAD")
                            if ok:
                                self.station_lock = "UNLOAD"
                                self.state = "LOWER_DROP"

                        elif self.edge_enter_cooldown <= 0.0:
                            self._try_enter_next_edge()

            self._update_visuals(dt)
            return

        self.target_hoist_offset = self.hoist_top_offset

        if self.current_edge is not None:
            edge = self.graph.edges[self.current_edge]
            delta_t = (current_speed * dt) / max(edge.length, 1e-6)
            self.edge_t += delta_t
            if self.edge_t >= 1.0:
                self.edge_t = 1.0
                finished_edge = self.current_edge
                self.controller.leave_edge(self.id, finished_edge)
                self.current_node = self.graph.edges[finished_edge].end
                self.current_edge = None
                self.route_index += 1
                self.edge_enter_cooldown = 0.05
        else:
            if not is_blocked and self.edge_enter_cooldown <= 0.0:
                self._try_enter_next_edge()

        self._update_visuals(dt)


# =============================================================================
# Stage helpers
# =============================================================================
def iter_stage_paths(stage: Usd.Stage):
    for prim in stage.Traverse():
        yield str(prim.GetPath()), prim


def maybe_print_stage_paths(stage: Usd.Stage, enabled: bool):
    if not enabled:
        return
    print("\n========== Stage Prim Paths ==========")
    for path_str, _ in iter_stage_paths(stage):
        print(path_str)
    print("======================================\n")


def _top_level_candidate_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    world = stage.GetPrimAtPath("/World")
    if not world or not world.IsValid():
        return []
    ignore = {"Environment", "Render", "Looks", "PhysicsScene", "OHTInfra", "OHTFleet"}
    out = []
    for child in world.GetChildren():
        if child.GetName() in ignore:
            continue
        out.append(child)
    return out


def find_map_root(stage: Usd.Stage, cfg: IntegrationConfig) -> Usd.Prim:
    if cfg.map_root_path:
        p = stage.GetPrimAtPath(cfg.map_root_path)
        if p and p.IsValid():
            return p

    p = stage.GetPrimAtPath("/World/UserMap")
    if p and p.IsValid():
        return p

    p = stage.GetPrimAtPath("/World/Factory")
    if p and p.IsValid():
        return p

    candidates = []
    cache = bbox_cache()
    for prim in _top_level_candidate_prims(stage):
        try:
            mn, mx = world_bbox(prim, cache)
            size = mx - mn
            vol = float(max(size[0], 0.0) * max(size[1], 0.0) * max(size[2], 0.0))
            if vol > 1e-3:
                candidates.append((vol, prim))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    return stage.GetPrimAtPath("/World")


def freeze_imported_map_rigid_bodies(root_path: str = "/World/UserMap"):
    stage = _stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[freeze_imported_map_rigid_bodies] root not found: {root_path}")
        return

    frozen_count = 0
    articulation_removed = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue

        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                rb_api.CreateRigidBodyEnabledAttr(False)
                frozen_count += 1
        except Exception:
            attr = prim.GetAttribute("physics:rigidBodyEnabled")
            if attr and attr.IsValid():
                attr.Set(False)
                frozen_count += 1

        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                articulation_removed += 1
        except Exception:
            pass

    print(
        f"[freeze_imported_map_rigid_bodies] frozen rigid bodies: {frozen_count}, "
        f"removed articulation roots: {articulation_removed}"
    )


# =============================================================================
# Layout
# =============================================================================
def infer_station_edge(
    ref_xy: np.ndarray,
    left_x: float,
    right_x: float,
    top_y: float,
    bottom_y: float,
    preferred_edge: Optional[str] = None,
) -> str:
    if preferred_edge in {"top", "bottom", "left", "right"}:
        return preferred_edge

    x = float(ref_xy[0])
    y = float(ref_xy[1])
    distances = {
        "top": abs(y - top_y),
        "bottom": abs(y - bottom_y),
        "left": abs(x - left_x),
        "right": abs(x - right_x),
    }
    return min(distances.items(), key=lambda kv: kv[1])[0]


def snap_station_to_edge(
    ref_xy: np.ndarray,
    edge: str,
    left_x: float,
    right_x: float,
    top_y: float,
    bottom_y: float,
    rail_z: float,
) -> np.ndarray:
    x = float(ref_xy[0])
    y = float(ref_xy[1])

    if edge == "top":
        return np.array([clamp(x, left_x, right_x), top_y, rail_z], dtype=float)
    if edge == "bottom":
        return np.array([clamp(x, left_x, right_x), bottom_y, rail_z], dtype=float)
    if edge == "left":
        return np.array([left_x, clamp(y, bottom_y, top_y), rail_z], dtype=float)
    if edge == "right":
        return np.array([right_x, clamp(y, bottom_y, top_y), rail_z], dtype=float)

    raise ValueError(f"지원하지 않는 station edge: {edge}")


def infer_rect_layout(stage: Usd.Stage, cfg: IntegrationConfig):
    root_prim = find_map_root(stage, cfg)
    root_min, root_max = world_bbox(root_prim, bbox_cache())

    floor_z = float(root_min[2])
    roof_z = float(root_max[2])

    if cfg.use_manual_rail_world_z:
        rail_z = cfg.manual_rail_world_z
    elif cfg.use_manual_rail_height_from_floor:
        rail_z = floor_z + cfg.manual_rail_height_from_floor
    else:
        rail_z = roof_z - 0.30

    if not cfg.manual_rect_mode:
        raise ValueError("이번 버전은 manual_rect_mode=True 기준입니다.")

    left_x = float(cfg.manual_left_x)
    right_x = float(cfg.manual_right_x)
    top_y = float(cfg.manual_top_y)
    bottom_y = float(cfg.manual_bottom_y)

    if cfg.pickup_use_world_xy_for_load_node:
        load_ref_xy = np.array([cfg.pickup_world_pos[0], cfg.pickup_world_pos[1]], dtype=float)
    else:
        load_ref_xy = np.array([cfg.manual_load_x, top_y], dtype=float)

    load_edge = infer_station_edge(
        load_ref_xy, left_x, right_x, top_y, bottom_y, cfg.load_station_edge
    )
    load_pos = snap_station_to_edge(load_ref_xy, load_edge, left_x, right_x, top_y, bottom_y, rail_z)

    if cfg.drop_use_world_xy_for_unload_node:
        raw_drop_world = resolve_drop_world_pos(cfg)
        unload_ref_xy = raw_drop_world[:2].copy()
    else:
        unload_ref_xy = np.array([cfg.manual_unload_x, top_y], dtype=float)

    unload_edge = infer_station_edge(
        unload_ref_xy, left_x, right_x, top_y, bottom_y, cfg.unload_station_edge
    )
    unload_pos = snap_station_to_edge(unload_ref_xy, unload_edge, left_x, right_x, top_y, bottom_y, rail_z)

    return {
        "root_prim": root_prim,
        "root_min": root_min,
        "root_max": root_max,
        "rail_z": rail_z,
        "left_x": left_x,
        "right_x": right_x,
        "top_y": top_y,
        "bottom_y": bottom_y,
        "load_pos": load_pos,
        "unload_pos": unload_pos,
        "load_edge": load_edge,
        "unload_edge": unload_edge,
    }


def add_debug_anchor(world: World, prim_path: str, name: str, pos: np.ndarray, color: np.ndarray):
    world.scene.add(
        FixedCuboid(
            prim_path=prim_path,
            name=name,
            position=pos,
            scale=np.array([0.22, 0.22, 0.22]),
            color=color,
        )
    )


def add_rect_debug_markers(world: World, layout: dict):
    z = layout["rail_z"]
    pts = {
        "TL": np.array([layout["left_x"], layout["top_y"], z]),
        "TR": np.array([layout["right_x"], layout["top_y"], z]),
        "BL": np.array([layout["left_x"], layout["bottom_y"], z]),
        "BR": np.array([layout["right_x"], layout["bottom_y"], z]),
        "LD": layout["load_pos"],
        "UD": layout["unload_pos"],
    }
    colors = {
        "TL": np.array([1.0, 0.8, 0.1]),
        "TR": np.array([1.0, 0.8, 0.1]),
        "BL": np.array([0.1, 0.8, 1.0]),
        "BR": np.array([0.1, 0.8, 1.0]),
        "LD": np.array([0.1, 1.0, 0.2]),
        "UD": np.array([1.0, 0.2, 0.2]),
    }
    for k, p in pts.items():
        add_debug_anchor(world, f"/World/OHTInfra/RectDebug/{k}", f"rect_debug_{k.lower()}", p, colors[k])


def _station_sort_key(edge: str, pos: np.ndarray) -> float:
    x = float(pos[0])
    y = float(pos[1])

    if edge == "top":
        return x
    if edge == "right":
        return -y
    if edge == "bottom":
        return -x
    if edge == "left":
        return y
    return 0.0


def build_two_room_rect_graph(layout: dict) -> RailGraphSystem:
    rail_z = layout["rail_z"]
    graph = RailGraphSystem(rail_z=rail_z)

    left_x = layout["left_x"]
    right_x = layout["right_x"]
    top_y = layout["top_y"]
    bottom_y = layout["bottom_y"]

    corners = {
        "TOP_LEFT":  np.array([left_x,  top_y,    rail_z], dtype=float),
        "TOP_RIGHT": np.array([right_x, top_y,    rail_z], dtype=float),
        "BOT_RIGHT": np.array([right_x, bottom_y, rail_z], dtype=float),
        "BOT_LEFT":  np.array([left_x,  bottom_y, rail_z], dtype=float),
    }

    for name, pos in corners.items():
        graph.add_node(name, pos, kind="normal")

    graph.add_node("LOAD", layout["load_pos"], kind="station")
    graph.add_node("UNLOAD", layout["unload_pos"], kind="station")

    station_infos = {
        "LOAD": {"edge": layout["load_edge"], "pos": layout["load_pos"]},
        "UNLOAD": {"edge": layout["unload_edge"], "pos": layout["unload_pos"]},
    }

    top_stations = [n for n in station_infos if station_infos[n]["edge"] == "top"]
    right_stations = [n for n in station_infos if station_infos[n]["edge"] == "right"]
    bottom_stations = [n for n in station_infos if station_infos[n]["edge"] == "bottom"]
    left_stations = [n for n in station_infos if station_infos[n]["edge"] == "left"]

    top_stations.sort(key=lambda n: _station_sort_key("top", station_infos[n]["pos"]))
    right_stations.sort(key=lambda n: _station_sort_key("right", station_infos[n]["pos"]))
    bottom_stations.sort(key=lambda n: _station_sort_key("bottom", station_infos[n]["pos"]))
    left_stations.sort(key=lambda n: _station_sort_key("left", station_infos[n]["pos"]))

    full_loop_nodes = (
        ["TOP_LEFT"]
        + top_stations
        + ["TOP_RIGHT"]
        + right_stations
        + ["BOT_RIGHT"]
        + bottom_stations
        + ["BOT_LEFT"]
        + left_stations
        + ["TOP_LEFT"]
    )

    edge_order = []
    for a, b in zip(full_loop_nodes[:-1], full_loop_nodes[1:]):
        en = f"E_{a}_TO_{b}"
        graph.add_edge(en, a, b)
        edge_order.append(en)

    graph.finalize_loop(edge_order)
    return graph


# =============================================================================
# pickup zone 생성
# =============================================================================
def build_pickup_transfer_zone(world: World, cfg: IntegrationConfig, layout: dict):
    pickup_pos = resolve_pickup_world_pos_from_layout(cfg, layout)
    yaw = station_edge_yaw(layout["load_edge"])

    base_path = "/World/OHTInfra/PickupZone"

    white = [0.95, 0.95, 0.96]
    light = [0.88, 0.89, 0.92]
    wall = [0.84, 0.85, 0.88]
    dark = [0.18, 0.19, 0.21]
    belt = [0.22, 0.23, 0.25]
    rail = [0.70, 0.72, 0.75]

    port_top_thickness = 0.08
    port_top_center_z_offset = -(cfg.pod_body_size[2] * 0.5 + port_top_thickness * 0.5)

    # ------------------------------------------------------------
    # 뒤쪽 메인 장비를 더 뒤로 배치
    # ------------------------------------------------------------
    main_body_center = place_local(pickup_pos, yaw, (-2.30, 0.86, 0.52))
    main_body_size = [4.30, 2.70, 2.30]

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/MainBody",
        "pickup_machine_main_body",
        main_body_center,
        main_body_size,
        white,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/Roof",
        "pickup_machine_roof",
        main_body_center + rotate_local(np.array([0.0, 0.0, main_body_size[2] * 0.5 + 0.06]), yaw),
        [4.00, 2.40, 0.10],
        [0.98, 0.98, 0.99],
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/RearUpper",
        "pickup_machine_rear_upper",
        place_local(pickup_pos, yaw, (-2.85, 0.86, 1.25)),
        [1.40, 2.15, 0.65],
        light,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/FrontNeck",
        "pickup_machine_front_neck",
        place_local(pickup_pos, yaw, (-1.45, 0.25, 0.18)),
        [1.30, 1.20, 0.98],
        wall,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/FrontUpper",
        "pickup_machine_front_upper",
        place_local(pickup_pos, yaw, (-1.48, 0.25, 0.98)),
        [1.35, 1.30, 0.58],
        light,
        yaw,
    )

    # ------------------------------------------------------------
    # 바깥 로드포트
    # ------------------------------------------------------------
    spawn_oriented_fixed_box(
        world,
        f"{base_path}/LoadPort/Pedestal",
        "pickup_loadport_pedestal",
        place_local(pickup_pos, yaw, (-0.38, 0.10, -0.35)),
        [1.05, 1.02, 0.62],
        white,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/LoadPort/TopPlate",
        "pickup_loadport_topplate",
        pickup_pos + rotate_local(np.array([-0.08, 0.0, port_top_center_z_offset]), yaw),
        [0.74, 0.74, port_top_thickness],
        [0.97, 0.97, 0.98],
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/LoadPort/RearHousing",
        "pickup_loadport_rear_housing",
        place_local(pickup_pos, yaw, (-0.98, 0.16, 0.02)),
        [0.74, 0.98, 0.94],
        wall,
        yaw,
    )

    # ------------------------------------------------------------
    # 컨베이어
    # ------------------------------------------------------------
    conv_start = place_local(pickup_pos, yaw, (-2.35, 0.10, port_top_center_z_offset - 0.04))
    conv_end = place_local(pickup_pos, yaw, (-0.18, 0.00, port_top_center_z_offset - 0.04))
    conv_mid = 0.5 * (conv_start + conv_end)
    conv_len = float(np.linalg.norm(conv_end - conv_start)) + 0.12

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Conveyor/Base",
        "pickup_conveyor_base",
        conv_mid,
        [conv_len, 0.60, 0.12],
        dark,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Conveyor/Belt",
        "pickup_conveyor_belt",
        conv_mid + np.array([0.0, 0.0, 0.05]),
        [conv_len * 0.97, 0.44, 0.04],
        belt,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Conveyor/GuideLeft",
        "pickup_conveyor_guide_left",
        conv_mid + rotate_local(np.array([0.0, 0.24, 0.09]), yaw),
        [conv_len * 0.96, 0.02, 0.05],
        rail,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Conveyor/GuideRight",
        "pickup_conveyor_guide_right",
        conv_mid + rotate_local(np.array([0.0, -0.24, 0.09]), yaw),
        [conv_len * 0.96, 0.02, 0.05],
        rail,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/LowerBoxL",
        "pickup_machine_lower_box_l",
        place_local(pickup_pos, yaw, (-1.10, 0.62, -0.48)),
        [0.82, 0.60, 0.52],
        white,
        yaw,
    )

    spawn_oriented_fixed_box(
        world,
        f"{base_path}/Machine/LowerBoxR",
        "pickup_machine_lower_box_r",
        place_local(pickup_pos, yaw, (-1.82, 0.62, -0.48)),
        [1.02, 0.64, 0.52],
        white,
        yaw,
    )

    return PickupPodConveyorManager(
        world=world,
        base_path=f"{base_path}/Supply",
        pickup_pos=pickup_pos,
        yaw=yaw,
        pod_body_size=cfg.pod_body_size,
        pod_lid_size=cfg.pod_lid_size,
        grip_clearance=cfg.pod_grip_clearance,
        cfg=cfg,
    )


# =============================================================================
# Scene build
# =============================================================================
def prepare_stage_and_world(cfg: IntegrationConfig) -> Tuple[Usd.Stage, World]:
    _unsubscribe_previous_physics_callback()

    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()

    if cfg.map_usd_path:
        if stage and stage.GetPrimAtPath("/World").IsValid():
            omni.kit.commands.execute("DeletePrims", paths=["/World"])

        omni.kit.commands.execute(
            "CreatePrimWithDefaultXform",
            prim_type="Xform",
            prim_path="/World",
        )

        omni.kit.commands.execute(
            "CreateReferenceCommand",
            usd_context=usd_context,
            path_to=Sdf.Path("/World/UserMap"),
            asset_path=cfg.map_usd_path,
            instanceable=False,
        )
        stage = usd_context.get_stage()
        freeze_imported_map_rigid_bodies("/World/UserMap")

    _delete_prims_if_exist(["/World/OHTFleet", "/World/OHTInfra"])

    if World.instance() is not None:
        World.instance().clear_instance()

    world = World()
    if not stage.GetPrimAtPath("/World").IsValid():
        omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Xform", prim_path="/World")

    return stage, world


def build_integrated_oht_scene(cfg: IntegrationConfig):
    stage, world = prepare_stage_and_world(cfg)
    maybe_print_stage_paths(stage, cfg.auto_print_stage_paths)

    layout = infer_rect_layout(stage, cfg)
    graph = build_two_room_rect_graph(layout)
    graph.build_rail_visuals(world, dense_spacing=0.26)

    controller = CentralController(graph)

    if cfg.build_pickup_transfer_zone:
        pickup_stack = build_pickup_transfer_zone(world, cfg, layout)
        controller.bind_pickup_stack(pickup_stack)

    if cfg.visualize_rect_debug:
        add_rect_debug_markers(world, layout)

    oht_specs = [
        ("OHT_A", np.array([1.00, 0.78, 0.10]), "TOP_LEFT"),
        ("OHT_B", np.array([0.10, 0.75, 1.00]), "TOP_RIGHT"),
        ("OHT_C", np.array([1.00, 0.35, 0.35]), "BOT_RIGHT"),
        ("OHT_D", np.array([0.40, 1.00, 0.40]), "BOT_LEFT"),
    ]

    num_ohts = max(1, min(int(cfg.num_ohts), len(oht_specs)))

    fleet = []
    for oht_id, color, start_node in oht_specs[:num_ohts]:
        oht = GraphOHT(
            world=world,
            graph=graph,
            controller=controller,
            oht_id=oht_id,
            color=color,
            start_node=start_node,
            move_speed=cfg.oht_speed,
            safe_distance=cfg.oht_safe_distance,
            cfg=cfg,
            enable_transport=True,
        )
        fleet.append(oht)

    for oht in fleet:
        oht.fleet = fleet

    world.reset()
    return world, graph, controller, fleet, layout



# =============================================================================
# another.py 통합용 카메라 / 로봇 제어기
# =============================================================================
def setup_camera_and_light():
    stage = omni.usd.get_context().get_stage()

    cam_path = "/World/ScriptCamera"
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
    transform_matrix = Gf.Matrix4d(
        -0.99978, 0.02106, 0.00066, 0.0,
        -0.01815, -6.0, 0.48032, 0.0,
        -0.01070, -0.4802, -0.87709, 0.0,
        -1.26865, -8.66478, 5.24915, 1.0,
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



class DetectionSubscriber:
    """
    /def_det_result(std_msgs/String) 구독용 경량 래퍼.
    file3 의 DetectionSubscriber 로직을 현재 통합 씬용으로 이식했다.
    """

    VALID_CLASSES = {"empty", "none", "scratch", "donut"}

    def __init__(self, topic_name: str = "/def_det_result"):
        self.topic_name = topic_name
        self.latest_class: str = "empty"
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

            outer_self = self

            class _InnerNode(Node):
                def __init__(inner_self):
                    super().__init__("wafer_detection_subscriber")
                    inner_self.create_subscription(
                        String,
                        outer_self.topic_name,
                        inner_self._callback,
                        10,
                    )

                def _callback(inner_self, msg):
                    value = msg.data.strip().lower()
                    if value not in DetectionSubscriber.VALID_CLASSES:
                        print(f"[DetectionSubscriber] Unknown class received: '{msg.data}' -> ignored")
                        return
                    with outer_self._lock:
                        prev = outer_self.latest_class
                        outer_self.latest_class = value
                    if prev != value:
                        print(f"[DetectionSubscriber] Class updated: '{prev}' -> '{value}'")

            self._node = _InnerNode()
            self._spin_thread = threading.Thread(
                target=self._spin_worker,
                daemon=True,
                name="ros2_detection_spin",
            )
            self._spin_thread.start()
            print(f"[DetectionSubscriber] ROS2 subscriber started on {self.topic_name}")

        except Exception as e:
            print(f"[DetectionSubscriber] ROS2 init failed: {e}")
            print("[DetectionSubscriber] Falling back to empty (no-op mode)")

    def _spin_worker(self):
        try:
            import rclpy
            rclpy.spin(self._node)
        except Exception as e:
            print(f"[DetectionSubscriber] spin error: {e}")

    def get_class(self) -> str:
        with self._lock:
            return self.latest_class

    def reset(self):
        with self._lock:
            self.latest_class = "empty"
        print("[DetectionSubscriber] Reset -> 'empty'")

class RMPFlowController(mg.MotionPolicyController):
    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
        attach_gripper: bool = False,
    ) -> None:
        if attach_gripper:
            cfg = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflowSuction"
            )
        else:
            cfg = mg.interface_config_loader.load_supported_motion_policy_config(
                "UR10", "RMPflow"
            )

        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**cfg)
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt
        )
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)

        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )


# =============================================================================
# Flask 웹 UI 연동 브리지 (self-contained)
# =============================================================================
class IsaacFlaskStateBridge:
    def __init__(
        self,
        status_path: str = "/tmp/acs_monitor/status.json",
        command_path: str = "/tmp/acs_monitor/command.json",
        publish_interval: float = 0.20,
        preview_dir: str = "/tmp/acs_monitor/previews",
        preview_interval: float = 0.80,
    ):
        self.status_path = Path(status_path)
        self.command_path = Path(command_path)
        self.publish_interval = float(publish_interval)
        self.preview_dir = Path(preview_dir)
        self.preview_interval = float(preview_interval)
        self.current_preview_path = self.preview_dir / "current.png"
        self._last_publish_time = 0.0
        self._last_preview_capture_time = 0.0
        self._preview_capture_task = None
        self._preview_force = True
        self._handled_command_ids = set()
        self._started_at = time.time()
        self.selected_camera_path = "/World/ScriptCamera"
        self.selected_camera_label = "Overview"

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.command_path.parent.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        # 예전 command 재실행 방지
        if self.command_path.exists():
            try:
                self.command_path.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 변환 유틸
    # ------------------------------------------------------------------
    def _to_jsonable(self, value: Any):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return [float(x) for x in value.tolist()]
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                if isinstance(item, (list, tuple, np.ndarray)):
                    out.append(self._to_jsonable(item))
                elif isinstance(item, (np.floating, float, np.integer, int)):
                    out.append(float(item) if isinstance(item, (np.floating, float)) else int(item))
                else:
                    out.append(item)
            return out
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _preview_meta(self) -> Dict[str, Any]:
        exists = self.current_preview_path.exists()
        mtime = None
        size = None
        if exists:
            try:
                stat = self.current_preview_path.stat()
                mtime = float(stat.st_mtime)
                size = int(stat.st_size)
            except Exception:
                mtime = None
                size = None
        return {
            "exists": exists,
            "path": str(self.current_preview_path),
            "mtime": mtime,
            "size": size,
        }

    async def _capture_current_viewport_preview_async(self):
        try:
            viewport = vp_util.get_active_viewport()
            if viewport is None:
                return
            # 카메라 전환 직후 한두 프레임 렌더를 기다린 뒤 캡처
            try:
                await omni.kit.app.get_app().next_update_async()
                await omni.kit.app.get_app().next_update_async()
            except Exception:
                pass

            helper = vp_util.capture_viewport_to_file(viewport, file_path=str(self.current_preview_path))
            if hasattr(helper, "wait_for_result"):
                await helper.wait_for_result()
            else:
                try:
                    await omni.kit.app.get_app().next_update_async()
                except Exception:
                    pass
            self._last_preview_capture_time = time.time()
        except Exception as exc:
            print(f"[WEB PREVIEW] capture failed: {exc}")
        finally:
            self._preview_capture_task = None
            if self._preview_force:
                # force 플래그는 다음 tick 에서 다시 재촬영이 필요할 수 있으므로 유지
                pass

    def tick_preview_capture(self):
        now = time.time()
        task_running = self._preview_capture_task is not None and not self._preview_capture_task.done()
        if task_running:
            return

        due = self._preview_force or (now - self._last_preview_capture_time >= self.preview_interval)
        if not due:
            return

        self._preview_force = False
        self._preview_capture_task = asyncio.ensure_future(self._capture_current_viewport_preview_async())

    def request_preview_capture(self, force: bool = False):
        if force:
            self._preview_force = True
        self.tick_preview_capture()

    def _phase_text(self, phase: Optional[int]) -> str:
        phase_map = {
            0: "대기",
            1: "픽업 상공",
            2: "픽업 하강",
            3: "흡착",
            4: "부착 대기",
            5: "상승",
            6: "이송",
            7: "배치 상공",
            8: "배치 하강",
            9: "릴리즈",
            10: "안정화",
            11: "상승 복귀",
            12: "복귀 이송",
            13: "홈 복귀",
        }
        return phase_map.get(phase, f"Phase {phase}") if phase is not None else "-"

    def _timeline_playing(self) -> bool:
        try:
            return bool(omni.timeline.get_timeline_interface().is_playing())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 상태 수집
    # ------------------------------------------------------------------
    def _build_graph_payload(self, graph) -> Dict[str, Any]:
        if graph is None:
            return {"nodes": {}, "edges": []}

        nodes = {}
        for name, node in graph.nodes.items():
            nodes[name] = {
                "name": name,
                "kind": getattr(node, "kind", "normal"),
                "pos": self._to_jsonable(getattr(node, "pos", None)),
            }

        edges = []
        order = list(getattr(graph, "loop_edge_order", []) or graph.edges.keys())
        for edge_name in order:
            edge = graph.edges[edge_name]
            edges.append(
                {
                    "name": edge.name,
                    "start": edge.start,
                    "end": edge.end,
                    "enabled": bool(edge.enabled),
                    "powered": bool(edge.powered),
                    "length": float(edge.length),
                }
            )
        return {"nodes": nodes, "edges": edges}

    def _build_layout_payload(self, layout) -> Optional[Dict[str, Any]]:
        if layout is None:
            return None
        return {
            "left_x": float(layout["left_x"]),
            "right_x": float(layout["right_x"]),
            "top_y": float(layout["top_y"]),
            "bottom_y": float(layout["bottom_y"]),
            "rail_z": float(layout["rail_z"]),
            "load_pos": self._to_jsonable(layout["load_pos"]),
            "unload_pos": self._to_jsonable(layout["unload_pos"]),
            "load_edge": layout["load_edge"],
            "unload_edge": layout["unload_edge"],
        }

    def _build_oht_payload(self, fleet) -> List[Dict[str, Any]]:
        out = []
        for oht in list(fleet or []):
            blocked = False
            try:
                blocked = bool(oht._is_blocked_by_front_oht())
            except Exception:
                blocked = False

            out.append(
                {
                    "id": getattr(oht, "id", "-"),
                    "state": getattr(oht, "state", "-"),
                    "current_node": getattr(oht, "current_node", None),
                    "current_edge": getattr(oht, "current_edge", None),
                    "target_station": getattr(oht, "target_station", None),
                    "route": list(getattr(oht, "route", []) or []),
                    "route_index": int(getattr(oht, "route_index", 0)),
                    "carrying": bool(getattr(oht, "carrying", False)),
                    "wafer_state": getattr(oht, "wafer_state", "-"),
                    "blocked": blocked,
                    "body_yaw": float(getattr(oht, "body_yaw", 0.0)),
                    "pos": self._to_jsonable(getattr(oht, "body_pos", None)),
                    "drop_world_pos": self._to_jsonable(getattr(oht, "drop_world_pos", None)),
                    "pickup_world_pos": self._to_jsonable(getattr(oht, "pickup_world_pos", None)),
                    "camera_path": getattr(oht, "camera_path", None),
                }
            )
        out.sort(key=lambda x: x["id"])
        return out

    def _build_bridge_payload(self, bridge) -> Dict[str, Any]:
        if bridge is None:
            return {
                "state": "-",
                "queue_len": 0,
                "pick_world": None,
                "place_world": None,
                "placed_wafer_count": 0,
                "active_wafer_exists": False,
            }

        return {
            "state": getattr(bridge, "state", "-"),
            "queue_len": len(getattr(bridge, "pending_pick_positions", []) or []),
            "pick_world": self._to_jsonable(getattr(bridge, "current_pick_world", None)),
            "place_world": self._to_jsonable(getattr(bridge, "current_place_world", None)),
            "placed_wafer_count": len(getattr(bridge, "placed_wafer_names", []) or []),
            "active_wafer_exists": bool(getattr(bridge, "active_wafer", None) is not None),
        }

    def _build_ur10_payload(self, ur10) -> Dict[str, Any]:
        if ur10 is None:
            return {
                "phase": None,
                "phase_text": "-",
                "pick_world": None,
                "place_position": None,
                "robot_position": None,
            }

        phase = getattr(ur10, "phase", None)
        return {
            "phase": phase,
            "phase_text": self._phase_text(phase),
            "pick_world": self._to_jsonable(getattr(ur10, "pick_world", None)),
            "place_position": self._to_jsonable(getattr(ur10, "place_position", None)),
            "robot_position": self._to_jsonable(getattr(ur10, "robot_position", None)),
            "detection_class": getattr(ur10, "detection_class", None),
            "latched_detection_class": getattr(ur10, "latched_detection_class", None),
        }

    def _build_jobs(self, ohts: List[Dict[str, Any]], bridge_payload: Dict[str, Any], ur10_payload: Dict[str, Any]) -> List[str]:
        jobs = []
        for oht in ohts:
            if oht["carrying"]:
                jobs.append(f'{oht["id"]}: LOAD -> UNLOAD 진행 중')
            elif oht["state"] in ("LOWER_PICK", "PICK_WAIT"):
                jobs.append(f'{oht["id"]}: LOAD 스테이션 픽업 중')
            elif oht["state"] in ("LOWER_DROP", "DROP_WAIT"):
                jobs.append(f'{oht["id"]}: UNLOAD 스테이션 드롭 중')
            else:
                jobs.append(f'{oht["id"]}: {oht["target_station"] or "-"} 대기 / 이동')

        detection_class = ur10_payload.get("detection_class")
        if bridge_payload.get("state") == "READY" and detection_class in (None, "empty"):
            jobs.insert(0, "UR10 Job: defect result waiting before pickup")
        elif bridge_payload.get("state") == "READY":
            jobs.insert(0, f"UR10 Job: classified pickup pending ({detection_class})")
        elif ur10_payload.get("phase") not in (None, 0, 13):
            jobs.insert(0, f'UR10 Job: {ur10_payload.get("phase_text", "-")}')
        return jobs

    def _build_alarms(self, summary: Dict[str, Any], bridge_payload: Dict[str, Any], ur10_payload: Dict[str, Any]) -> List[List[str]]:
        alarms = []
        if summary["pickup_stock"] == 0:
            alarms.append(["INFO", "LOAD 측 POD 재공급 대기 중"])
        if summary["blocked_count"] > 0:
            alarms.append(["WARN", f'차간거리로 정지한 OHT {summary["blocked_count"]}대'])
        if bridge_payload.get("state") == "READY" and ur10_payload.get("detection_class") in (None, "empty"):
            alarms.append(["WAIT", "드롭 전달 웨이퍼 결함 판정 대기 중"])
        elif bridge_payload.get("state") == "READY":
            alarms.append(["READY", f"분류 완료 웨이퍼 픽업 대기 ({ur10_payload.get('detection_class')})"])
        if ur10_payload.get("phase") not in (None, 0, 13):
            alarms.append(["RUN", f'UR10 동작 중 - {ur10_payload.get("phase_text", "-")}'])
        if not alarms:
            alarms.append(["OK", "활성 알람 없음"])
        return alarms

    def build_payload(self, graph, layout, ctrl, fleet, bridge, ur10) -> Dict[str, Any]:
        ohts = self._build_oht_payload(fleet)
        carrying_count = sum(1 for x in ohts if x["carrying"])
        blocked_count = sum(1 for x in ohts if x["blocked"])

        bridge_payload = self._build_bridge_payload(bridge)
        ur10_payload = self._build_ur10_payload(ur10)

        pickup_stock = 0
        if ctrl is not None and hasattr(ctrl, "pickup_remaining_count"):
            try:
                pickup_stock = int(ctrl.pickup_remaining_count())
            except Exception:
                pickup_stock = 0

        summary = {
            "sim_playing": self._timeline_playing(),
            "oht_total": len(ohts),
            "carrying_count": carrying_count,
            "blocked_count": blocked_count,
            "pickup_stock": pickup_stock,
            "bridge_queue": int(bridge_payload.get("queue_len", 0)),
            "bridge_state": bridge_payload.get("state", "-"),
            "placed_wafer_count": int(bridge_payload.get("placed_wafer_count", 0)),
            "ur10_phase_text": ur10_payload.get("phase_text", "-"),
        }

        jobs = self._build_jobs(ohts, bridge_payload, ur10_payload)
        alarms = self._build_alarms(summary, bridge_payload, ur10_payload)

        payload = {
            "timestamp": time.time(),
            "camera": {
                "selected_path": self.selected_camera_path,
                "selected_label": self.selected_camera_label,
                "preview": self._preview_meta(),
            },
            "summary": summary,
            "layout": self._build_layout_payload(layout),
            "graph": self._build_graph_payload(graph),
            "ohts": ohts,
            "bridge": bridge_payload,
            "ur10": ur10_payload,
            "jobs": jobs,
            "alarms": alarms,
        }
        return payload

    # ------------------------------------------------------------------
    # 외부 공개 메서드
    # ------------------------------------------------------------------
    def publish(self, graph, layout, ctrl, fleet, bridge, ur10):
        now = time.time()
        if now - self._last_publish_time < self.publish_interval:
            return
        payload = self.build_payload(graph, layout, ctrl, fleet, bridge, ur10)
        self._atomic_write_json(self.status_path, payload)
        self._last_publish_time = now

    def _safe_play(self, world):
        try:
            if hasattr(world, "play_async"):
                asyncio.ensure_future(world.play_async())
                self.request_preview_capture(force=True)
                return
        except Exception:
            pass
        try:
            world.play()
            self.request_preview_capture(force=True)
            return
        except Exception:
            pass
        try:
            omni.timeline.get_timeline_interface().play()
            self.request_preview_capture(force=True)
        except Exception:
            pass

    def _safe_set_active_camera(self, camera_path: str, label: Optional[str] = None):
        if not camera_path or not _stage_prim_valid(camera_path):
            return False
        try:
            viewport = vp_util.get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera(camera_path)
                self.selected_camera_path = camera_path
                self.selected_camera_label = label or camera_path
                self.request_preview_capture(force=True)
                return True
        except Exception as exc:
            print(f"[WEB CAM] failed to set active camera: {exc}")
        return False

    def _safe_pause(self, world):
        try:
            world.pause()
            self.request_preview_capture(force=True)
            return
        except Exception:
            pass
        try:
            omni.timeline.get_timeline_interface().pause()
            self.request_preview_capture(force=True)
        except Exception:
            pass

    def _safe_stop(self, world):
        try:
            world.stop()
            self.request_preview_capture(force=True)
            return
        except Exception:
            pass
        try:
            omni.timeline.get_timeline_interface().stop()
            self.request_preview_capture(force=True)
        except Exception:
            pass

    def _safe_reset(self, world):
        try:
            world.pause()
        except Exception:
            pass
        try:
            world.reset()
        except Exception:
            pass
        try:
            omni.timeline.get_timeline_interface().pause()
        except Exception:
            pass
        self.request_preview_capture(force=True)

    def handle_commands(self, world):
        if not self.command_path.exists():
            return None

        try:
            with self.command_path.open("r", encoding="utf-8") as f:
                cmd = json.load(f)
        except Exception:
            return None

        cmd_id = cmd.get("id")
        action = str(cmd.get("action", "")).strip().lower()
        created_at = float(cmd.get("created_at", 0.0) or 0.0)
        camera_path = str(cmd.get("camera_path", "") or "").strip()
        label = str(cmd.get("label", "") or "").strip()

        if not cmd_id or cmd_id in self._handled_command_ids:
            return None
        if created_at < self._started_at - 1.0:
            self._handled_command_ids.add(cmd_id)
            return None

        if action == "play":
            self._safe_play(world)
        elif action == "pause":
            self._safe_pause(world)
        elif action == "stop":
            self._safe_stop(world)
        elif action == "reset":
            self._safe_reset(world)
        elif action == "view_camera":
            ok = self._safe_set_active_camera(camera_path, label or camera_path)
            if not ok:
                self._handled_command_ids.add(cmd_id)
                return None
        elif action == "view_overview_camera":
            ok = self._safe_set_active_camera("/World/ScriptCamera", "Overview")
            if not ok:
                self._handled_command_ids.add(cmd_id)
                return None
        else:
            self._handled_command_ids.add(cmd_id)
            return None

        self._handled_command_ids.add(cmd_id)
        print(f"[WEB CMD] handled: {action} ({cmd_id})")
        return action


# =============================================================================
# code_7 드롭 이벤트 -> another 물리 웨이퍼 브리지
# =============================================================================
class Code7TransferWaferBridge:
    """
    code_7의 OHT 드롭 이벤트를 another 스타일의 물리 웨이퍼로 바꿔 주는 브리지.

    이번 수정 포인트
    1) OHT가 drop 하면 POD box(cargo)는 즉시 숨김
    2) UR10이 place 하면 그 웨이퍼는 월드에 그대로 남김
    3) 다음 사이클용 전달 웨이퍼는 새 prim으로 다시 생성
    """

    def __init__(
        self,
        world: World,
        wafer_scale: Optional[np.ndarray] = None,
        wafer_color: Optional[np.ndarray] = None,
        ready_lock_steps: int = 8,
    ):
        self.world = world
        self.wafer_scale = np.array(
            wafer_scale if wafer_scale is not None else [0.05, 0.05, 0.05],
            dtype=float,
        )
        self.wafer_color = np.array(
            wafer_color if wafer_color is not None else [0.5, 0.2, 0.1],
            dtype=float,
        )
        self.ready_lock_steps = int(ready_lock_steps)

        self.state = "IDLE"
        self.pending_pick_positions = deque()
        self.prev_drop_active: Dict[str, bool] = {}
        self.current_pick_world: Optional[np.ndarray] = None
        self.current_place_world: Optional[np.ndarray] = None
        self._ready_lock_counter = 0

        self.active_wafer = None
        self.active_wafer_index = 0
        self.placed_wafer_names: List[str] = []

    def _spawn_new_wafer(self, pos: np.ndarray):
        self.active_wafer_index += 1
        prim_path = f"/World/TransferredWafers/Wafer_{self.active_wafer_index:04d}"
        name = f"code7_transfer_wafer_{self.active_wafer_index:04d}"
        self.active_wafer = self.world.scene.add(
            DynamicCylinder(
                prim_path=prim_path,
                name=name,
                position=np.array(pos, dtype=float),
                scale=self.wafer_scale,
                color=self.wafer_color,
            )
        )
        self.placed_wafer_names.append(name)
        self._set_active_pose(pos)

    def _set_active_pose(self, pos: np.ndarray):
        if self.active_wafer is None:
            self._spawn_new_wafer(pos)
            return
        self.active_wafer.set_world_pose(
            position=np.array(pos, dtype=float),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )
        self.active_wafer.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=float))
        self.active_wafer.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=float))

    def get_active_wafer(self):
        return self.active_wafer

    def is_ready(self) -> bool:
        return self.state == "READY" and self.current_pick_world is not None and self.active_wafer is not None

    def get_pick_world_pos(self) -> Optional[np.ndarray]:
        if not self.is_ready():
            return None
        return self.current_pick_world.copy()

    def on_robot_attached(self):
        if self.state == "READY":
            self.state = "HELD"

    def lock_at_place(self, place_world: np.ndarray):
        if self.active_wafer is None:
            return
        self.current_place_world = np.array(place_world, dtype=float)
        self._set_active_pose(self.current_place_world)
        self.state = "PLACED"

    def reset_after_place(self):
        self.current_pick_world = None
        self.current_place_world = None
        self._ready_lock_counter = 0
        self.state = "IDLE"
        self.active_wafer = None

    def _hide_oht_drop_box(self, oht: GraphOHT):
        try:
            oht.drop_display_timer = 0.0
            oht.wafer_state = "AT_PICKUP"
            oht.cargo.set_pose(np.array([0.0, 0.0, -100.0]), oht.body_yaw, visible=False)
        except Exception:
            pass

    def _enqueue_new_drop_events(self, fleet: List[GraphOHT]):
        for oht in fleet:
            active = bool(oht.wafer_state == "AT_DROP" and oht.drop_display_timer > 0.0)
            prev = self.prev_drop_active.get(oht.id, False)
            if active and not prev:
                self.pending_pick_positions.append(np.array(oht.drop_world_pos, dtype=float))
                self._hide_oht_drop_box(oht)
                print(f"[BRIDGE] code_7 drop event queued from {oht.id} at {oht.drop_world_pos}")
            self.prev_drop_active[oht.id] = active

    def update(self, fleet: List[GraphOHT], dt: float):
        self._enqueue_new_drop_events(fleet)

        if self.state == "IDLE" and self.pending_pick_positions:
            self.current_pick_world = self.pending_pick_positions.popleft()
            self._spawn_new_wafer(self.current_pick_world)
            self._ready_lock_counter = 0
            self.state = "READY"
            print(f"[BRIDGE] Transfer wafer READY at {self.current_pick_world}")

        if self.state == "READY" and self.current_pick_world is not None and self.active_wafer is not None:
            if self._ready_lock_counter < self.ready_lock_steps:
                self._set_active_pose(self.current_pick_world)
                self._ready_lock_counter += 1

        elif self.state == "PLACED" and self.current_place_world is not None and self.active_wafer is not None:
            self._set_active_pose(self.current_place_world)


# =============================================================================
# another.py UR10 픽앤플레이스 통합
# =============================================================================

class DetectionAwareIntegratedWaferPickupRobot:
    """
    file1 의 UR10 픽앤플레이스 골격에 file3 의 결함 분류 분기 로직을 합친 버전.

    핵심 동작
    - code_7 OHT drop -> Code7TransferWaferBridge 가 물리 웨이퍼 생성
    - /def_det_result 가 empty 가 아닐 때까지 대기
    - none  -> NORMAL place
    - scratch/donut -> DEFECT place
    - pick/place 후 홈 복귀, DetectionSubscriber reset
    """

    PLACE_POS_NORMAL = np.array([-2.75777, -8.8769, 2.125], dtype=float)
    PLACE_POS_DEFECT = np.array([-0.73097, -8.94011, 2.09002], dtype=float)

    def __init__(self, bridge: Code7TransferWaferBridge, detection_subscriber: Optional[DetectionSubscriber] = None):
        self.bridge = bridge
        self.detection_subscriber = detection_subscriber
        self.BROWN = np.array([0.5, 0.2, 0.1], dtype=float)

        self.robot_position = np.array([-1.69498, -9.02501299679875, 2.1], dtype=float)
        self.place_position = self.PLACE_POS_NORMAL.copy()

        self.use_fixed_pick_approach = True
        self.fixed_pick_approach_world = np.array([-2.59, -8.12, 1.92], dtype=float)
        self.fixed_pick_approach_tolerance = 0.01

        self.phase = 0
        self.wait_counter = 0
        self.attach_wait_counter = 0
        self.post_release_counter = 0
        self.pick_world = None
        self.debug_printed = False

        self.approach_offset_z = 0.20
        self.pick_lift_clearance_z = 0.72
        self.travel_clearance_z = 0.82
        self.move_tolerance = 0.04
        self.pick_tolerance = 0.09
        self.attach_wait_steps = 3
        self.post_release_hold_steps = 12

        self.robots = None
        self.cspace_controller = None
        self.wafer = None

        self.robot_base_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.home_joint_positions = np.array(
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0],
            dtype=float,
        )
        self.idle_home_hold_enabled = True
        self.idle_home_hold_tolerance = 0.02

        self.detection_class = "empty"
        self.latched_detection_class = "empty"
        self._detection_candidate = "empty"
        self._detection_stable_counter = 0
        self.detection_stable_steps = 5

    def setup_scene(self, world: World):
        assets_root_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
        ur10_usd = assets_root_path + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
        robot_prim = add_reference_to_stage(usd_path=ur10_usd, prim_path="/World/UR10")
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
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0], dtype=float)
        )
        self._boost_gripper_force()

    def _boost_gripper_force(self):
        try:
            stage = omni.usd.get_context().get_stage()
            gripper_prim_path = "/World/UR10/ee_link/SurfaceGripper"
            gripper_prim = stage.GetPrimAtPath(gripper_prim_path)
            if not gripper_prim.IsValid():
                return

            attr_map = {
                "physxSurfaceGripper:forceLimit": 1.0e2,
                "physxSurfaceGripper:torqueLimit": 1.0e4,
                "physxSurfaceGripper:gripThreshold": 0.02,
                "physxSurfaceGripper:retryClose": True,
            }
            for attr_name, value in attr_map.items():
                attr = gripper_prim.GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(value)
        except Exception as exc:
            print(f"[UR10] gripper tuning skipped: {exc}")

    def bind_world_objects(self, world: World):
        self.robots = world.scene.get_object("my_ur10")
        self.robots.set_world_pose(
            position=self.robot_position,
            orientation=self.robot_base_orientation,
        )
        self.cspace_controller = RMPFlowController(
            name="merged_ur10_cspace_controller",
            robot_articulation=self.robots,
            attach_gripper=True,
        )
        self._apply_home_action()
        self.cspace_controller.reset()

    def _target_rel(self, world_pos: np.ndarray) -> np.ndarray:
        return np.array(world_pos, dtype=float) - self.robot_position

    def _make_articulation_action(self):
        if ISAAC_VERSION == "isaacsim":
            from isaacsim.core.utils.types import ArticulationAction
        else:
            from omni.isaac.core.utils.types import ArticulationAction
        return ArticulationAction

    def _apply_home_action(self):
        if self.robots is None:
            return
        ArticulationAction = self._make_articulation_action()
        self.robots.apply_action(ArticulationAction(joint_positions=self.home_joint_positions))

    def _hold_idle_home(self):
        if not self.idle_home_hold_enabled or self.robots is None:
            return

        current_joint_positions = self.robots.get_joint_positions()
        if current_joint_positions is None or len(current_joint_positions) < 6:
            return

        joint_error = np.max(np.abs(current_joint_positions[:6] - self.home_joint_positions))
        if joint_error > self.idle_home_hold_tolerance:
            self._apply_home_action()

    def _move_to_world_pose(self, target_world: np.ndarray, tol: float, label: str) -> bool:
        target_rel = self._target_rel(target_world)
        target_quat = euler_angles_to_quat(np.array([0.0, np.pi / 2.0, 0.0], dtype=float))

        action = self.cspace_controller.forward(
            target_end_effector_position=target_rel,
            target_end_effector_orientation=target_quat,
        )
        self.robots.apply_action(action)

        current_joint_positions = self.robots.get_joint_positions()
        reached = False
        if action.joint_positions is not None:
            reached = np.all(np.abs(current_joint_positions[:6] - action.joint_positions) < tol)

        if reached:
            self.cspace_controller.reset()
            print(f"[UR10] {label} reached")
        return reached

    def _move_home(self) -> bool:
        if self.robots is None:
            return False

        self._apply_home_action()
        current_joint_positions = self.robots.get_joint_positions()
        if current_joint_positions is None or len(current_joint_positions) < 6:
            return False
        return np.all(np.abs(current_joint_positions[:6] - self.home_joint_positions) < self.idle_home_hold_tolerance)

    def _safe_hover_z(self) -> float:
        z = max(
            float(self.place_position[2]) + self.travel_clearance_z,
            float(self.pick_world[2]) + self.pick_lift_clearance_z if self.pick_world is not None else 0.0,
            float(self.robot_position[2]) + 0.60,
        )
        return max(z, 2.85)

    def _pick_approach_world(self) -> np.ndarray:
        if self.use_fixed_pick_approach:
            return self.fixed_pick_approach_world.copy()
        target = self.pick_world.copy()
        target[2] = self._safe_hover_z()
        return target

    def _pick_retreat_world(self) -> np.ndarray:
        target = self.pick_world.copy()
        target[2] = self._safe_hover_z()
        return target

    def _place_hover_world(self) -> np.ndarray:
        target = self.place_position.copy()
        target[2] = self._safe_hover_z()
        return target

    def _transit_world(self) -> np.ndarray:
        return np.array([
            (float(self.robot_position[0]) + float(self.place_position[0])) * 0.5,
            float(self.robot_position[1]) - 0.02,
            self._safe_hover_z(),
        ], dtype=float)

    def _current_detection(self) -> str:
        if self.detection_subscriber is None:
            return "empty"
        try:
            value = self.detection_subscriber.get_class().strip().lower()
        except Exception:
            return "empty"
        return value if value in DetectionSubscriber.VALID_CLASSES else "empty"

    def _target_from_detection(self, detection_class: str) -> Optional[np.ndarray]:
        if detection_class == "none":
            return self.PLACE_POS_NORMAL.copy()
        if detection_class in ("scratch", "donut"):
            return self.PLACE_POS_DEFECT.copy()
        return None

    def _poll_detection_before_pick(self) -> bool:
        self.detection_class = self._current_detection()
        target = self._target_from_detection(self.detection_class)
        if target is None:
            self._detection_candidate = "empty"
            self._detection_stable_counter = 0
            return False

        if self.detection_class == self._detection_candidate:
            self._detection_stable_counter += 1
        else:
            self._detection_candidate = self.detection_class
            self._detection_stable_counter = 1

        self.place_position = target.copy()
        return self._detection_stable_counter >= self.detection_stable_steps

    def _reset_cycle(self):
        self.phase = 0
        self.wait_counter = 0
        self.attach_wait_counter = 0
        self.post_release_counter = 0
        self.pick_world = None
        self.debug_printed = False
        self.wafer = None
        self.detection_class = "empty"
        self.latched_detection_class = "empty"
        self._detection_candidate = "empty"
        self._detection_stable_counter = 0
        if self.cspace_controller is not None:
            self.cspace_controller.reset()
        self._apply_home_action()
        self.bridge.reset_after_place()
        if self.detection_subscriber is not None:
            try:
                self.detection_subscriber.reset()
            except Exception:
                pass
        print("[UR10] cycle reset, waiting next classified code_7 drop")

    def physics_step(self, step_size: float):
        try:
            self._physics_step_impl(float(step_size))
        except Exception as e:
            if "Failed to get DOF" in str(e):
                return
            raise

    def _physics_step_impl(self, dt: float):
        if self.robots is None or self.cspace_controller is None:
            return

        self.wafer = self.bridge.get_active_wafer()

        if not self.debug_printed and self.phase == 0 and self.wafer is not None:
            robot_actual_pos, _ = self.robots.get_world_pose()
            wafer_actual_pos, _ = self.wafer.get_world_pose()
            print("=" * 60)
            print(f"[DEBUG] Robot actual pose : {robot_actual_pos}")
            print(f"[DEBUG] Robot base pose   : {self.robot_position}")
            print(f"[DEBUG] Transfer wafer   : {wafer_actual_pos}")
            print(f"[DEBUG] Fixed approach   : {self.fixed_pick_approach_world}")
            print("=" * 60)
            self.debug_printed = True

        if self.phase == 0:
            self._hold_idle_home()
            if self.bridge.is_ready():
                ready_to_pick = self._poll_detection_before_pick()
                if ready_to_pick:
                    self.wait_counter += 1
                    if self.wait_counter >= 10:
                        self.pick_world = self.bridge.get_pick_world_pos()
                        self.latched_detection_class = self.detection_class
                        print(
                            f"[UR10][Phase 0] classified wafer ready -> {self.pick_world} "
                            f"| detection='{self.latched_detection_class}' "
                            f"| place={self.place_position}"
                        )
                        self.phase = 1
                        self.wait_counter = 0
                else:
                    self.wait_counter = 0
            else:
                self.wait_counter = 0
                self.detection_class = "empty"
                self._detection_candidate = "empty"
                self._detection_stable_counter = 0
            return

        if self.pick_world is None:
            return

        if self.phase == 1:
            label = "Phase 1 fixed pick approach" if self.use_fixed_pick_approach else "Phase 1 pick hover"
            tol = self.fixed_pick_approach_tolerance if self.use_fixed_pick_approach else self.move_tolerance
            if self._move_to_world_pose(self._pick_approach_world(), tol, label):
                self.phase = 2
            return

        if self.phase == 2:
            if self._move_to_world_pose(self.pick_world, self.pick_tolerance, "Phase 2 descend"):
                self.phase = 3
            return

        if self.phase == 3:
            self.robots.gripper.close()
            self.bridge.on_robot_attached()
            self.attach_wait_counter = 0
            print("[UR10][Phase 3] gripper close -> Phase 4")
            self.phase = 4
            return

        if self.phase == 4:
            self.attach_wait_counter += 1
            if self.attach_wait_counter >= self.attach_wait_steps:
                self.phase = 5
            return

        if self.phase == 5:
            if self._move_to_world_pose(self._pick_retreat_world(), self.move_tolerance, "Phase 5 retreat up"):
                self.phase = 6
            return

        if self.phase == 6:
            if self._move_to_world_pose(self._transit_world(), self.move_tolerance, "Phase 6 transit"):
                self.phase = 7
            return

        if self.phase == 7:
            if self._move_to_world_pose(self._place_hover_world(), self.move_tolerance, "Phase 7 place hover"):
                self.phase = 8
            return

        if self.phase == 8:
            if self._move_to_world_pose(self.place_position, self.pick_tolerance, "Phase 8 place descend"):
                self.phase = 9
            return

        if self.phase == 9:
            self.bridge.lock_at_place(self.place_position)
            self.robots.gripper.open()
            self.post_release_counter = 0
            print(
                f"[UR10][Phase 9] gripper open -> Phase 10 "
                f"| detection='{self.latched_detection_class}'"
            )
            self.phase = 10
            return

        if self.phase == 10:
            if self.post_release_counter < self.post_release_hold_steps:
                self.bridge.lock_at_place(self.place_position)
                self.post_release_counter += 1
                return
            self.phase = 11
            return

        if self.phase == 11:
            if self._move_to_world_pose(self._place_hover_world(), self.move_tolerance, "Phase 11 retreat up"):
                self.phase = 12
            return

        if self.phase == 12:
            if self._move_to_world_pose(self._transit_world(), self.move_tolerance, "Phase 12 return transit"):
                self.phase = 13
            return

        if self.phase == 13:
            if self._move_home():
                print("[UR10][Phase 13] home reached")
                self._reset_cycle()
            return


# =============================================================================
# 실행 설정
# =============================================================================
CFG = IntegrationConfig(
    map_usd_path="/home/rokey/Downloads/smcnd_factory_v18.usd",
    map_root_path="/World/UserMap",

    use_manual_rail_world_z=True,
    manual_rail_world_z=9.38,
    use_manual_rail_height_from_floor=False,
    manual_rail_height_from_floor=2.55,

    manual_rect_mode=True,
    manual_left_x=-20.425,
    manual_right_x=20.575,
    manual_top_y=-8.00,
    manual_bottom_y=-30.20,

    manual_load_x=1.0,
    manual_unload_x=-3.0,
    load_station_edge=None,
    unload_station_edge=None,

    num_ohts=2,
    oht_speed=4,
    oht_safe_distance=3.6,

    enable_pick_drop_cycle=True,

    pickup_use_world_xy_for_load_node=True,
    pickup_world_pos=(0.0, -30.20, 0.70),

    drop_use_world_xy_for_unload_node=True,
    use_local_conveyor_drop=False,
    conveyor_local_pos=(-2.0, -8.0, 1.4),
    conveyor_drop_z_offset=0.9,
    drop_world_pos=(-2.85, -8.0, 1.95),

    pickup_dwell_s=0.15,
    drop_dwell_s=0.15,
    drop_display_s=0.8,
    recycle_wafer_after_drop=True,
    head_target_clearance=0.02,
    min_hoist_offset=-12.0,

    hoist_top_offset=-0.08,
    hoist_hold_offset=-0.08,
    hoist_speed=5.0,

    build_pickup_transfer_zone=True,

    wafer_image_path="/home/rokey/work/wafer/image",
    pod_internal_wafer_count=1,
    pod_internal_wafer_size=(0.26, 0.26, 1.0),

    pod_body_size=(0.56, 0.56, 0.36),
    pod_lid_size=(0.70, 0.70, 0.07),
    pod_grip_clearance=0.05,
    pod_carried_offset_z=-0.24,

    pickup_refill_delay_s=10.0,
    pickup_conveyor_speed=0.38,

    visualize_rect_debug=False,
    auto_print_stage_paths=False,
    debug_graph=True,
)

_WORLD = None
_GRAPH = None
_CTRL = None
_FLEET = None
_LAYOUT = None
_BRIDGE = None
_UR10 = None
_DETECTION_SUB = None
_WEB_BRIDGE = None
_WEB_CMD_SUB = None


def poll_web_commands_on_app_update(e=None):
    global _WEB_BRIDGE, _WORLD

    if _WEB_BRIDGE is None or _WORLD is None:
        return
    if not _stage_prim_valid("/World"):
        return

    try:
        _WEB_BRIDGE.handle_commands(_WORLD)
        _WEB_BRIDGE.tick_preview_capture()
    except Exception as exc:
        print(f"[WEB CMD] app-update poll error: {exc}")


def merged_on_physics_step(step_size: float):
    global _CTRL, _FLEET, _BRIDGE, _UR10, _WORLD, _GRAPH, _LAYOUT, _WEB_BRIDGE

    if _CTRL is None or _FLEET is None or _BRIDGE is None or _UR10 is None:
        return
    if not _stage_prim_valid("/World"):
        return

    dt = max(1.0 / 120.0, float(step_size))

    if _WEB_BRIDGE is not None and _WORLD is not None:
        _WEB_BRIDGE.handle_commands(_WORLD)

    _CTRL.update(dt)

    for oht in list(_FLEET):
        try:
            if oht._alive():
                oht.update(dt)
        except RuntimeError as e:
            if "expired" in str(e).lower():
                continue
            raise

    _BRIDGE.update(_FLEET, dt)
    _UR10.physics_step(dt)

    if _WEB_BRIDGE is not None:
        _WEB_BRIDGE.publish(_GRAPH, _LAYOUT, _CTRL, _FLEET, _BRIDGE, _UR10)
        _WEB_BRIDGE.tick_preview_capture()


def _unsubscribe_previous_web_cmd_subscription():
    global _WEB_CMD_SUB
    if _WEB_CMD_SUB is not None:
        try:
            _WEB_CMD_SUB.unsubscribe()
        except Exception:
            pass
        _WEB_CMD_SUB = None
    sub = getattr(builtins, "_acs_web_cmd_sub", None)
    if sub is not None:
        try:
            sub.unsubscribe()
        except Exception:
            pass
        builtins._acs_web_cmd_sub = None


async def main():
    global _WORLD, _GRAPH, _CTRL, _FLEET, _LAYOUT, _BRIDGE, _UR10, _DETECTION_SUB, _WEB_BRIDGE, _WEB_CMD_SUB

    _unsubscribe_previous_web_cmd_subscription()

    world = World.instance()
    if world is not None:
        try:
            world.stop()
        except Exception:
            pass
        world.clear_instance()

    await omni.usd.get_context().new_stage_async()

    cfg = CFG
    if cfg.map_usd_path and (not os.path.exists(cfg.map_usd_path)):
        print(f"[INFO] map_usd_path not found -> {cfg.map_usd_path}")
        print("[INFO] background 없이 code_7 + defect-aware UR10 통합 씬을 생성합니다.")
        cfg.map_usd_path = None

    _WORLD, _GRAPH, _CTRL, _FLEET, _LAYOUT = build_integrated_oht_scene(cfg)

    if hasattr(_WORLD, "initialize_simulation_context_async"):
        try:
            await _WORLD.initialize_simulation_context_async()
        except Exception as e:
            print(f"[WARN] initialize_simulation_context_async skipped: {e}")

    setup_camera_and_light()

    _BRIDGE = Code7TransferWaferBridge(_WORLD)
    _DETECTION_SUB = DetectionSubscriber("/def_det_result")
    _UR10 = DetectionAwareIntegratedWaferPickupRobot(_BRIDGE, detection_subscriber=_DETECTION_SUB)
    _UR10.setup_scene(_WORLD)

    if hasattr(_WORLD, "reset_async"):
        await _WORLD.reset_async()
    else:
        _WORLD.reset()

    _UR10.bind_world_objects(_WORLD)

    _WEB_BRIDGE = IsaacFlaskStateBridge(
        status_path="/tmp/acs_monitor/status.json",
        command_path="/tmp/acs_monitor/command.json",
        publish_interval=0.20,
        preview_dir="/tmp/acs_monitor/previews",
        preview_interval=0.80,
    )
    print("[WEB] Flask status bridge ready -> /tmp/acs_monitor/status.json")

    _WEB_BRIDGE._safe_set_active_camera("/World/ScriptCamera", "Overview")

    app = omni.kit.app.get_app()
    _WEB_CMD_SUB = app.get_update_event_stream().create_subscription_to_pop(
        poll_web_commands_on_app_update,
        name="acs_web_command_poll",
    )
    builtins._acs_web_cmd_sub = _WEB_CMD_SUB
    print("[WEB] app-update command polling enabled")

    if hasattr(_WORLD, "add_physics_callback"):
        _WORLD.add_physics_callback("merged_oht_ur10_step", callback_fn=merged_on_physics_step)
    else:
        raise RuntimeError("World.add_physics_callback 를 사용할 수 없습니다.")

    print("\n[Merged scene ready]")
    print(f"- import mode          : {ISAAC_VERSION}")
    print(f"- map root             : {_LAYOUT['root_prim'].GetPath() if _LAYOUT['root_prim'] else 'NOT FOUND'}")
    print(f"- rail z               : {_LAYOUT['rail_z']}")
    print(f"- load pos             : {_LAYOUT['load_pos']}")
    print(f"- unload pos           : {_LAYOUT['unload_pos']}")
    print(f"- OHT count            : {len(_FLEET)}")
    print(f"- pickup stock count   : {_CTRL.pickup_remaining_count()}")
    print(f"- UR10 base            : {_UR10.robot_position}")
    print(f"- UR10 normal place    : {_UR10.PLACE_POS_NORMAL}")
    print(f"- UR10 defect place    : {_UR10.PLACE_POS_DEFECT}")
    print("- code_7 full OHT cycle active")
    print("- file3 defect-aware UR10 waits for code_7 drop events")

    if hasattr(_WORLD, "play_async"):
        await _WORLD.play_async()
    else:
        _WORLD.play()

    print("[Merged simulation started]")


asyncio.ensure_future(main())