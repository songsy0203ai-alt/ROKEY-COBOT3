# =============================================================================
# [코드 기능]
#   Isaac Sim 환경에서 OHT(Overhead Hoist Transport, 천장 주행 반송 로봇) 시뮬레이션을 구성한다.
#   공장 맵(USD)에서 컨베이어/로봇 prim의 위치를 자동으로 탐지하고,
#   두 장비 사이에 타원형 폐루프 레일을 생성한 뒤,
#   OHT 차량을 올려 FOUP(웨이퍼 박스)를 자동으로 픽업/드롭하는 사이클을 반복한다.
#
# [입력(Input)]
#   - 이미 Isaac Sim Stage에 열려 있는 USD 맵 (또는 CFG.map_usd_path로 지정한 USD 파일)
#   - IntegrationConfig(CFG) : 사용자 설정값 구조체
#       · map_usd_path           : 불러올 USD 파일 경로 (None이면 현재 Stage 사용)
#       · conveyor_anchor_path   : 컨베이어 prim의 정확한 USD path (모르면 None)
#       · robot_anchor_path      : 로봇 prim의 정확한 USD path (모르면 None)
#       · load/unload_xy_offset  : prim 원점과 실제 서비스 포인트 간 보정 벡터
#       · rail_height_above_anchor, rail_return_y_offset, turn_segments 등 레일 형상 파라미터
#       · spawn_two_ohts, oht_speed, oht_safe_distance 등 OHT 운행 파라미터
#
# [출력(Output)]
#   - _WORLD    : Isaac Sim World 객체 (물리 시뮬레이션 컨텍스트)
#   - _GRAPH    : RailGraphSystem 객체 (노드/엣지로 구성된 레일 그래프)
#   - _CTRL     : CentralController 객체 (경로 계획 및 엣지 진입 제어)
#   - _FLEET    : GraphOHT 객체 리스트 (실제로 움직이는 OHT 차량들)
#   - _ANCHORS  : 탐지된 prim 및 앵커 좌표 딕셔너리
#   - 콘솔 출력 : 탐지된 prim path 및 앵커 좌표
#   - 시각 오브젝트 : Stage 위에 레일, 침목, 스테이션 패드, OHT 차체 Cuboid가 생성됨
#   - Physics step 콜백 등록 : 매 물리 스텝마다 OHT 위치/호이스트 자동 갱신
# =============================================================================

import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.usd
import omni.kit.commands
import omni.physx

from pxr import Usd, Gf, Sdf

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid


# =============================================================================
# 유틸
# =============================================================================

def np3(v) -> np.ndarray:
    """
    [기능] 길이 3짜리 시퀀스를 float64 numpy 배열로 변환한다.
    [입력] v : 길이 3의 임의 시퀀스 (리스트, 튜플, Gf.Vec3 등)
    [출력] np.ndarray shape=(3,) dtype=float64
    """
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    [기능] 두 벡터 a, b 사이를 t(0~1)로 선형 보간한다.
    [입력]
        a : 시작 벡터 (np.ndarray)
        b : 끝 벡터   (np.ndarray)
        t : 보간 비율  (float, 0=a, 1=b)
    [출력] 보간된 위치 벡터 (np.ndarray)
    """
    return (1.0 - t) * a + t * b


# =============================================================================
# 설정
# =============================================================================

@dataclass
class IntegrationConfig:
    """
    [기능] OHT 시뮬레이션 전체의 사용자 설정값을 담는 데이터 클래스.
           맵 경로, prim 탐색 키워드, 레일 형상, OHT 운행 파라미터 등을 포함한다.
    [입력] 각 필드에 직접 값을 대입하거나 기본값 사용
    [출력] build_integrated_oht_scene() 등 빌더 함수에 인자로 전달됨
    """

    # 이미 Stage에 네 USD 맵이 열려 있으면 None 유지
    # 스크립트에서 직접 맵을 열고 싶으면 절대경로 지정
    map_usd_path: Optional[str] = None

    # Stage를 유지한 채 OHT 인프라만 추가
    keep_existing_world: bool = True

    # 가장 좋은 방식: 네 맵 안에 정확한 prim path를 직접 적는다
    conveyor_anchor_path: Optional[str] = None   # 예: "/World/MyFactory/Conveyor_01"
    robot_anchor_path: Optional[str] = None      # 예: "/World/MyFactory/Doosan_M0609"

    # prim path를 모를 때 이름 키워드로 자동 탐색
    conveyor_prim_candidates: Tuple[str, ...] = (
        "Conveyor", "conveyor", "Belt", "belt", "Conv"
    )
    robot_prim_candidates: Tuple[str, ...] = (
        "Robot", "robot", "Arm", "arm", "Doosan", "M0609", "m0609"
    )

    # 컨베이어 prim 원점과 실제 픽업 위치의 차이를 보정하는 오프셋 벡터
    load_xy_offset_from_conveyor: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    # 로봇 prim 원점과 실제 드롭 위치의 차이를 보정하는 오프셋 벡터
    unload_xy_offset_from_robot: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )

    # 레일 형상 파라미터
    rail_height_above_anchor: float = 3.6    # 앵커 높이 기준으로 레일을 띄우는 높이 (m)
    rail_return_y_offset: float = -4.0       # 복귀 레일(하단)의 Y 방향 오프셋 (m)
    left_margin: float = 2.4                 # 루프 좌측 여유 거리 (m)
    right_margin: float = 2.4               # 루프 우측 여유 거리 (m)
    min_loop_length: float = 6.0            # LOAD~UNLOAD 최소 거리 (이보다 짧으면 강제 확장)
    turn_segments: int = 8                  # 반원 턴을 몇 개의 선분으로 나눌지

    # OHT 운행 파라미터
    spawn_two_ohts: bool = True             # True면 OHT 2대 생성
    oht_speed: float = 2.2                  # OHT 이동 속도 (m/s)
    oht_safe_distance: float = 3.6          # 앞 OHT와 유지해야 할 최소 거리 (m)

    # 디버그 옵션
    auto_print_stage_paths: bool = True     # True면 Stage의 모든 prim path를 콘솔 출력
    visualize_inferred_anchors: bool = True # True면 앵커 위치에 색깔 Cuboid 마커 표시
    use_box_fallback_if_anchor_missing: bool = True  # prim 미발견 시 기본 좌표로 대체


# =============================================================================
# 그래프 데이터 구조
# =============================================================================

@dataclass
class RailNode:
    """
    [기능] 레일 그래프의 노드(정점) 하나를 나타낸다.
    [필드]
        name : 노드 고유 이름 (예: "LOAD", "TOP_LEFT")
        pos  : 3D 좌표 (np.ndarray shape=(3,))
        kind : 노드 유형 ("normal" 또는 "station")
    """
    name: str
    pos: np.ndarray
    kind: str = "normal"


@dataclass
class RailEdge:
    """
    [기능] 레일 그래프의 엣지(간선) 하나를 나타낸다.
    [필드]
        name    : 엣지 고유 이름
        start   : 출발 노드 이름
        end     : 도착 노드 이름
        enabled : 해당 엣지의 활성화 여부
        powered : 전력 공급 여부 (비활성 엣지는 경로 계획에서 제외)
        length  : 엣지 실제 길이 (m, 두 노드 사이 유클리드 거리)
        meta    : 추가 메타데이터 딕셔너리
    """
    name: str
    start: str
    end: str
    enabled: bool = True
    powered: bool = True
    length: float = 1.0
    meta: dict = field(default_factory=dict)


class RailGraphSystem:
    """
    [기능] 노드와 방향 엣지로 구성된 레일 네트워크를 관리한다.
           OHT의 위치 샘플링, 루프 진행거리 계산, 레일 시각화도 담당한다.
    """

    def __init__(self, rail_z: float = 4.8):
        """
        [입력] rail_z : 레일 기준 높이 (m, float)
        [출력] 빈 그래프 초기화
        """
        self.rail_z = rail_z                        # 레일 전체 기준 Z 좌표
        self.nodes: Dict[str, RailNode] = {}        # 노드 이름 → RailNode
        self.edges: Dict[str, RailEdge] = {}        # 엣지 이름 → RailEdge
        self.outgoing_edges: Dict[str, List[str]] = {}  # 노드 이름 → 출발 엣지 이름 리스트

        # 폐루프 진행거리 계산용 변수
        self.loop_edge_order: List[str] = []            # 루프를 구성하는 엣지 순서 리스트
        self.edge_progress_offset: Dict[str, float] = {} # 엣지 이름 → 루프 시작부터의 누적 거리
        self.node_progress_offset: Dict[str, float] = {} # 노드 이름 → 루프 시작부터의 누적 거리
        self.total_loop_length: float = 1.0           # 루프 전체 길이 (m)

        self.rail_visuals = {}   # 엣지 이름 → 시각화 오브젝트 리스트 (디버그용)

    def add_node(self, name: str, pos_xyz: np.ndarray, kind: str = "normal"):
        """
        [기능] 그래프에 노드를 추가하고 outgoing_edges 딕셔너리를 초기화한다.
        [입력]
            name    : 노드 이름 (str)
            pos_xyz : 3D 좌표 (np.ndarray)
            kind    : 노드 유형 문자열 (기본 "normal")
        [출력] 없음 (self.nodes, self.outgoing_edges 갱신)
        """
        self.nodes[name] = RailNode(name=name, pos=pos_xyz.astype(float), kind=kind)
        self.outgoing_edges.setdefault(name, [])

    def add_edge(self, name: str, start: str, end: str, meta: Optional[dict] = None):
        """
        [기능] start→end 방향 엣지를 추가하고 두 노드 간 유클리드 거리를 length로 저장한다.
        [입력]
            name  : 엣지 이름 (str)
            start : 출발 노드 이름 (str)
            end   : 도착 노드 이름 (str)
            meta  : 추가 메타데이터 딕셔너리 (선택)
        [출력] 없음 (self.edges, self.outgoing_edges 갱신)
        """
        p0 = self.nodes[start].pos
        p1 = self.nodes[end].pos
        # 두 노드 간 거리를 엣지 길이로 계산 (최소 1e-6 보호)
        length = max(float(np.linalg.norm(p1 - p0)), 1e-6)
        edge = RailEdge(name=name, start=start, end=end, length=length, meta=meta or {})
        self.edges[name] = edge
        self.outgoing_edges[start].append(name)

    def sample_edge(self, edge_name: str, t: float) -> np.ndarray:
        """
        [기능] 특정 엣지 위의 t(0~1) 비율 위치 좌표를 반환한다.
        [입력]
            edge_name : 엣지 이름 (str)
            t         : 보간 비율 (float, 0=start, 1=end)
        [출력] 해당 위치의 3D 좌표 (np.ndarray)
        """
        edge = self.edges[edge_name]
        return lerp(self.nodes[edge.start].pos, self.nodes[edge.end].pos, t)

    def finalize_loop(self, edge_order: List[str]):
        """
        [기능] 루프를 구성하는 엣지 순서를 확정하고, 각 엣지/노드의 루프 내 누적 거리를 계산한다.
        [입력] edge_order : 루프를 이루는 엣지 이름 리스트 (순서 중요)
        [출력] 없음 (self.loop_edge_order, edge_progress_offset, node_progress_offset, total_loop_length 갱신)
        """
        self.loop_edge_order = edge_order[:]
        self.edge_progress_offset.clear()
        self.node_progress_offset.clear()

        s = 0.0
        for edge_name in self.loop_edge_order:
            edge = self.edges[edge_name]
            self.edge_progress_offset[edge_name] = s       # 엣지 시작점의 루프 누적 거리
            self.node_progress_offset[edge.start] = s      # 해당 출발 노드의 루프 누적 거리
            s += edge.length                               # 다음 엣지 누적 거리 갱신

        self.total_loop_length = max(s, 1e-6)             # 루프 전체 길이 저장

    def progress_of(self, current_node: Optional[str], current_edge: Optional[str], edge_t: float) -> float:
        """
        [기능] OHT의 현재 위치(노드 또는 엣지 위)를 루프 기준 누적 진행거리(m)로 반환한다.
        [입력]
            current_node : 현재 정차 중인 노드 이름 (엣지 이동 중이면 None)
            current_edge : 현재 이동 중인 엣지 이름 (노드 정차 중이면 None)
            edge_t       : 엣지 내 이동 비율 (0~1)
        [출력] 루프 기준 누적 진행거리 (float, 0 ~ total_loop_length)
        """
        if current_edge is not None:
            # 엣지 시작 누적거리 + 엣지 내 비율 × 엣지 길이 → 루프 내 절대 진행거리
            return (
                self.edge_progress_offset.get(current_edge, 0.0)
                + edge_t * self.edges[current_edge].length
            ) % self.total_loop_length

        if current_node is not None:
            # 노드 자체의 루프 내 누적 거리 반환
            return self.node_progress_offset.get(current_node, 0.0) % self.total_loop_length

        return 0.0

    def build_rail_visuals(self, world: World, samples_per_edge: int = 8):
        """
        [기능] 루프의 각 엣지를 samples_per_edge 개의 구간으로 나눠
               레일 2줄(FixedCuboid)과 침목(FixedCuboid)을 Stage에 추가한다.
        [입력]
            world           : Isaac Sim World 객체
            samples_per_edge: 엣지당 레일 시각 오브젝트 분할 수 (기본 8)
        [출력] 없음 (Stage에 시각 Cuboid 오브젝트 생성)
        """
        for edge_name in self.loop_edge_order:
            visuals = []

            p0 = self.nodes[self.edges[edge_name].start].pos
            p1 = self.nodes[self.edges[edge_name].end].pos

            # 엣지 방향 벡터를 정규화하여 레일 좌우 배치에 사용
            tangent = p1 - p0
            norm = float(np.linalg.norm(tangent[:2]))
            if norm < 1e-6:
                tangent = np.array([1.0, 0.0, 0.0], dtype=float)
                norm = 1.0
            tangent = tangent / norm
            # 2D 수직 벡터 계산 (레일 2줄을 좌우로 배치할 방향)
            side = np.array([-tangent[1], tangent[0], 0.0], dtype=float)

            for i in range(samples_per_edge):
                t = (i + 0.5) / samples_per_edge        # 구간 중앙 t값
                p = self.sample_edge(edge_name, t)       # 해당 t의 3D 좌표

                # 레일 2줄: 좌(-1)·우(+1) 각각 Cuboid 배치
                for lane_sign in (-1.0, 1.0):
                    world.scene.add(
                        FixedCuboid(
                            prim_path=f"/World/OHTInfra/Rails/{edge_name}/Rail_{i}_{int(lane_sign>0)}",
                            name=f"{edge_name}_rail_{i}_{int(lane_sign>0)}",
                            position=np.array([p[0], p[1], self.rail_z]) + side * 0.13 * lane_sign,
                            scale=np.array([0.45, 0.07, 0.06]),
                            color=np.array([0.24, 0.24, 0.28]),  # 어두운 회색
                        )
                    )

                # 침목(레일 사이를 가로지르는 직사각형 블록) 추가
                sleeper = world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/OHTInfra/Rails/{edge_name}/Sleeper_{i}",
                        name=f"{edge_name}_sleeper_{i}",
                        position=np.array([p[0], p[1], self.rail_z - 0.06]),
                        scale=np.array([0.08, 0.42, 0.04]),
                        color=np.array([0.34, 0.22, 0.10]),     # 갈색
                    )
                )
                visuals.append(sleeper)

            self.rail_visuals[edge_name] = visuals


# =============================================================================
# 중앙 컨트롤러
# =============================================================================

class CentralController:
    """
    [기능] 레일 그래프 위에서 OHT의 경로를 계획하고,
           엣지 진입/이탈을 관리하는 중앙 교통 제어 클래스.
    """

    def __init__(self, graph: RailGraphSystem):
        """
        [입력] graph : RailGraphSystem 객체
        [출력] 컨트롤러 초기화
        """
        self.graph = graph

    def plan_route(self, start_node: str, goal_node: str) -> List[str]:
        """
        [기능] 다익스트라 알고리즘으로 start_node → goal_node 최단 경로를 탐색한다.
               비활성(enabled=False) 또는 무전력(powered=False) 엣지는 제외한다.
        [입력]
            start_node : 출발 노드 이름 (str)
            goal_node  : 목표 노드 이름 (str)
        [출력] 경로를 구성하는 엣지 이름 리스트 (List[str]), 경로 없으면 빈 리스트
        """
        pq = [(0.0, start_node, [])]       # (누적비용, 현재노드, 경로엣지목록) 우선순위 큐
        best = {start_node: 0.0}           # 노드별 최소 비용 기록

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal_node:
                return path               # 목표 도달 시 경로 반환

            if cost > best.get(node, float("inf")):
                continue                  # 이미 더 좋은 경로가 있으면 스킵

            for edge_name in self.graph.outgoing_edges.get(node, []):
                edge = self.graph.edges[edge_name]
                if not edge.enabled or not edge.powered:
                    continue              # 비활성/무전력 엣지 제외

                nxt = edge.end
                ncost = cost + edge.length

                if ncost < best.get(nxt, float("inf")):
                    best[nxt] = ncost
                    heapq.heappush(pq, (ncost, nxt, path + [edge_name]))

        return []   # 경로 없음

    def request_edge_entry(self, oht_id: str, edge_name: str) -> Tuple[bool, str]:
        """
        [기능] OHT가 특정 엣지에 진입을 요청할 때 허가 여부를 반환한다.
               현재는 항상 허가(True)를 반환하는 스텁(stub) 구현.
        [입력]
            oht_id    : OHT 고유 ID (str)
            edge_name : 진입하려는 엣지 이름 (str)
        [출력] (허가 여부: bool, 메시지: str) 튜플
        """
        return True, "OK"

    def leave_edge(self, oht_id: str, edge_name: str):
        """
        [기능] OHT가 엣지를 벗어났음을 컨트롤러에 알린다.
               현재는 스텁 구현 (향후 점유 해제 로직 추가 가능).
        [입력]
            oht_id    : OHT 고유 ID (str)
            edge_name : 벗어난 엣지 이름 (str)
        [출력] 없음
        """
        return None


# =============================================================================
# OHT 모델
# =============================================================================

class GraphOHT:
    """
    [기능] 레일 그래프 위를 이동하며 FOUP을 픽업/드롭하는 OHT 차량 하나를 나타낸다.
           차체(Body), 호이스트(Hoist), 파이프(Pipe), 헤드(Head), FOUP Cuboid로 구성된다.
           상태 머신(state machine)으로 MOVING / LOWER_PICK / PICK_WAIT 등의 동작을 제어한다.
    """

    def __init__(
        self,
        world: World,
        graph: RailGraphSystem,
        controller: CentralController,
        oht_id: str,
        color: np.ndarray,
        foup_color: np.ndarray,
        start_node: str,
        move_speed: float,
        safe_distance: float,
    ):
        """
        [기능] OHT를 초기화하고 Stage에 시각 Cuboid를 생성한다.
        [입력]
            world        : Isaac Sim World 객체
            graph        : RailGraphSystem 객체
            controller   : CentralController 객체
            oht_id       : 차량 고유 ID 문자열 (예: "OHT_A")
            color        : 차체 색상 RGB (np.ndarray shape=(3,))
            foup_color   : FOUP 색상 RGB (np.ndarray shape=(3,))
            start_node   : 시작 노드 이름 (str)
            move_speed   : 이동 속도 (m/s, float)
            safe_distance: 앞 차량과 최소 유지 거리 (m, float)
        [출력] 초기화된 GraphOHT 인스턴스, Stage에 시각 오브젝트 생성
        """
        self.world = world
        self.graph = graph
        self.controller = controller
        self.id = oht_id
        self.color = color
        self.foup_color = foup_color

        self.fleet = []   # 동일 Fleet에 속한 다른 OHT 참조 리스트 (충돌 방지용)

        # 현재 위치 상태
        self.current_node = start_node   # 현재 정차 중인 노드 이름
        self.current_edge: Optional[str] = None  # 이동 중인 엣지 이름 (정차 중이면 None)
        self.edge_t = 0.0               # 현재 엣지 내 이동 비율 (0~1)

        # 경로 계획 상태
        self.route: List[str] = []      # 현재 목표까지의 엣지 이름 시퀀스
        self.route_index = 0            # 다음에 진입할 엣지의 인덱스

        # 상태 머신 초기 상태
        self.state = "WAIT_AT_NODE"
        self.target_station = "LOAD"    # 처음 목표는 LOAD 스테이션
        self.carrying = False           # FOUP 적재 여부

        self.move_speed = move_speed
        self.safe_distance = safe_distance
        self.edge_enter_cooldown = 0.0  # 엣지 진입 쿨다운 타이머 (초)

        # 차체 위치 및 호이스트 오프셋 초기화
        self.body_pos = self.graph.nodes[start_node].pos.copy()
        self.body_to_anchor = 0.23      # 차체 중심에서 체인 앵커까지 Z 오프셋 (m)
        self.head_to_foup = 0.24        # 헤드 중심에서 FOUP 하단까지 Z 오프셋 (m)

        self.hoist_top_offset = -0.18   # 호이스트가 올라간 상태(수납)의 Z 오프셋
        self.hoist_pick_offset = -3.05  # 호이스트가 내려간 상태(픽업)의 Z 오프셋
        self.current_hoist_offset = self.hoist_top_offset
        self.target_hoist_offset = self.hoist_top_offset
        self.hoist_speed = 3.2          # 호이스트 이동 속도 (m/s)

        self.timer = 0.0            # 픽업/드롭 대기 타이머
        self.drop_hold_time = 0.0   # 드롭 후 FOUP 시각 유지 시간

        # ── 시각 오브젝트 생성 ──────────────────────────────────────────────

        # 차체 Cuboid (황색 또는 녹색)
        self.body = self.world.scene.add(
            VisualCuboid(
                prim_path=f"/World/OHTFleet/{self.id}/Body",
                name=f"{self.id}_body",
                position=self.body_pos,
                scale=np.array([0.76, 0.50, 0.34]),
                color=self.color,
            )
        )

        # 호이스트 Cuboid (흰색 작은 박스, 체인 감개 역할)
        self.hoist = self.world.scene.add(
            VisualCuboid(
                prim_path=f"/World/OHTFleet/{self.id}/Hoist",
                name=f"{self.id}_hoist",
                position=self.body_pos + np.array([0.0, 0.0, -0.20]),
                scale=np.array([0.28, 0.28, 0.16]),
                color=np.array([0.92, 0.92, 0.95]),
            )
        )

        # 파이프 Cuboid (가느다란 수직 막대, 체인/와이어 표현)
        self.pipe = self.world.scene.add(
            VisualCuboid(
                prim_path=f"/World/OHTFleet/{self.id}/Pipe",
                name=f"{self.id}_pipe",
                position=self.body_pos + np.array([0.0, 0.0, -0.75]),
                scale=np.array([0.07, 0.07, 1.0]),
                color=np.array([0.65, 0.68, 0.72]),
            )
        )

        # 헤드 Cuboid (FOUP을 잡는 그리퍼 말단 표현)
        self.head = self.world.scene.add(
            VisualCuboid(
                prim_path=f"/World/OHTFleet/{self.id}/Head",
                name=f"{self.id}_head",
                position=self.body_pos + np.array([0.0, 0.0, -1.20]),
                scale=np.array([0.22, 0.22, 0.08]),
                color=np.array([0.15, 0.15, 0.18]),
            )
        )

        # FOUP Cuboid (웨이퍼 박스 표현)
        self.foup = self.world.scene.add(
            VisualCuboid(
                prim_path=f"/World/OHTFleet/{self.id}/FOUP",
                name=f"{self.id}_foup",
                position=np.array([0.0, 0.0, 1.15]),
                scale=np.array([0.30, 0.30, 0.30]),
                color=self.foup_color,
            )
        )

        self._ensure_route()            # 초기 경로 계획
        self._update_visuals(0.0)       # 초기 시각 오브젝트 위치 설정

    def station_body_xy(self, which: str) -> np.ndarray:
        """
        [기능] 지정한 스테이션 노드의 XY 좌표를 반환한다.
        [입력] which : 스테이션 노드 이름 ("LOAD" 또는 "UNLOAD")
        [출력] XY 좌표 (np.ndarray shape=(2,))
        """
        n = self.graph.nodes[which].pos
        return np.array([n[0], n[1]], dtype=float)

    def station_foup_pos(self, which: str) -> np.ndarray:
        """
        [기능] 지정한 스테이션의 FOUP 정치 위치(XY + 고정 Z)를 반환한다.
        [입력] which : 스테이션 노드 이름 ("LOAD" 또는 "UNLOAD")
        [출력] 3D 위치 (np.ndarray shape=(3,), Z=1.15로 고정)
        """
        xy = self.station_body_xy(which)
        return np.array([xy[0], xy[1], 1.15], dtype=float)

    def _position_on_graph(self) -> np.ndarray:
        """
        [기능] OHT의 현재 그래프 위 3D 좌표를 반환한다.
               엣지 이동 중이면 엣지 보간 위치, 노드 정차 중이면 노드 위치를 반환한다.
        [입력] 없음 (self 상태 참조)
        [출력] 3D 좌표 (np.ndarray shape=(3,))
        """
        if self.current_edge is not None:
            return self.graph.sample_edge(self.current_edge, self.edge_t)
        return self.graph.nodes[self.current_node].pos

    def _loop_progress(self) -> float:
        """
        [기능] 현재 OHT의 루프 내 누적 진행거리를 반환한다. (충돌 방지에 사용)
        [입력] 없음 (self 상태 참조)
        [출력] 루프 내 누적 거리 (float, 0 ~ total_loop_length)
        """
        return self.graph.progress_of(self.current_node, self.current_edge, self.edge_t)

    def _is_blocked_by_front_oht(self) -> bool:
        """
        [기능] 루프 진행 방향으로 safe_distance 이내에 다른 OHT가 있으면 True를 반환한다.
               동일 Fleet의 모든 OHT를 순회하여 gap을 계산한다.
        [입력] 없음 (self.fleet 참조)
        [출력] 차단 여부 (bool)
        """
        my_s = self._loop_progress()

        for other in self.fleet:
            if other.id == self.id:
                continue   # 자기 자신은 제외

            other_s = other._loop_progress()
            # 루프 방향(순방향) 기준 앞 차와의 거리 계산 (모듈로 처리로 루프 경계 처리)
            gap = (other_s - my_s) % self.graph.total_loop_length

            if 1e-4 < gap < self.safe_distance:
                return True   # 너무 가까우면 차단

        return False

    def _update_visuals(self, dt: float):
        """
        [기능] 매 스텝마다 OHT의 5개 시각 Cuboid 위치를 갱신한다.
               호이스트를 target_hoist_offset 방향으로 hoist_speed로 부드럽게 이동시킨다.
               적재 여부에 따라 FOUP의 위치도 갱신한다.
        [입력] dt : 경과 시간 (초, float)
        [출력] 없음 (Stage Cuboid의 set_world_pose 호출)
        """
        self.body_pos = self._position_on_graph()  # 차체 위치 갱신

        # 호이스트 오프셋 보간: 목표값으로 hoist_speed × dt만큼 이동
        diff = self.target_hoist_offset - self.current_hoist_offset
        step = self.hoist_speed * dt
        if abs(diff) <= step:
            self.current_hoist_offset = self.target_hoist_offset   # 목표 도달
        else:
            self.current_hoist_offset += step if diff > 0.0 else -step

        # 각 파트의 절대 위치 계산
        hoist_center = self.body_pos + np.array([0.0, 0.0, self.current_hoist_offset])
        anchor_top = self.body_pos + np.array([0.0, 0.0, -self.body_to_anchor])
        head_center = hoist_center + np.array([0.0, 0.0, -0.16])
        pipe_center = 0.5 * (anchor_top + head_center)  # 파이프 중간 지점
        pipe_length = max(0.12, abs(anchor_top[2] - head_center[2]))  # 파이프 길이 동적 계산

        # Stage의 각 Cuboid 위치 갱신
        self.body.set_world_pose(position=self.body_pos)
        self.hoist.set_world_pose(position=hoist_center)
        self.pipe.set_world_pose(position=pipe_center)
        self.pipe.set_local_scale(np.array([0.07, 0.07, pipe_length]))  # 파이프 길이 조정
        self.head.set_world_pose(position=head_center)

        # FOUP 위치: 적재 중이면 헤드에 부착, 아니면 스테이션 바닥에 표시
        if self.carrying:
            self.foup.set_world_pose(position=head_center + np.array([0.0, 0.0, -self.head_to_foup]))
        else:
            if self.drop_hold_time > 0.0:
                # 드롭 직후 잠깐 UNLOAD 스테이션 바닥에 보여줌
                self.foup.set_world_pose(position=self.station_foup_pos("UNLOAD"))
            else:
                # 평소엔 LOAD 스테이션 바닥에 표시
                self.foup.set_world_pose(position=self.station_foup_pos("LOAD"))

    def _head_matches_station(self, which: str) -> bool:
        """
        [기능] 헤드 위치가 지정 스테이션의 FOUP 위치와 충분히 가까운지 확인한다.
               픽업/드롭 가능 여부 판단에 사용된다.
        [입력] which : 스테이션 이름 ("LOAD" 또는 "UNLOAD")
        [출력] 위치 일치 여부 (bool, XYZ 각 방향 허용 오차 내이면 True)
        """
        head_pos, _ = self.head.get_world_pose()
        target = self.station_foup_pos(which)

        # X, Y, Z 각 방향 오차 체크
        x_ok = abs(head_pos[0] - target[0]) < 0.18
        y_ok = abs(head_pos[1] - target[1]) < 0.18
        z_ok = abs(head_pos[2] - (target[2] + 0.05)) < 0.15
        return x_ok and y_ok and z_ok

    def _ensure_route(self):
        """
        [기능] 현재 노드에서 target_station까지의 경로를 (재)계획한다.
               이미 목적지에 있으면 빈 경로로 설정한다.
        [입력] 없음 (self.current_node, self.target_station 참조)
        [출력] 없음 (self.route, self.route_index 갱신)
        """
        if self.current_node is None:
            return

        target_node = self.target_station
        if target_node == self.current_node:
            self.route = []
            self.route_index = 0
            return

        # 컨트롤러에 경로 계획 요청
        self.route = self.controller.plan_route(self.current_node, target_node)
        self.route_index = 0

    def _try_enter_next_edge(self):
        """
        [기능] 다음 엣지 진입을 시도한다.
               컨트롤러 허가를 받으면 상태를 "MOVING"으로 전환한다.
        [입력] 없음 (self.route, self.route_index 참조)
        [출력] 진입 성공 여부 (bool)
        """
        if self.current_node is None:
            return False

        # 경로가 없거나 다 소진되면 재계획
        if self.route_index >= len(self.route):
            self._ensure_route()

        if self.route_index >= len(self.route):
            self.state = "WAIT_NO_ROUTE"
            return False

        next_edge = self.route[self.route_index]
        ok, _ = self.controller.request_edge_entry(self.id, next_edge)

        if ok:
            # 엣지 진입: 상태 변수 업데이트
            self.current_edge = next_edge
            self.current_node = None   # 노드 점유 해제
            self.edge_t = 0.0          # 엣지 시작 위치
            self.state = "MOVING"
            return True

        return False

    def update(self, dt: float):
        """
        [기능] 매 물리 스텝(dt초)마다 OHT의 상태 머신을 실행하고 시각 오브젝트를 갱신한다.
               상태에 따라 이동/호이스트 하강/픽업 대기/호이스트 상승/드롭 대기 등을 처리한다.
        [입력] dt : 물리 스텝 경과 시간 (초, float)
        [출력] 없음 (self 상태 및 Stage Cuboid 위치 갱신)
        """
        # 드롭 후 FOUP 시각 유지 타이머 감소
        if self.drop_hold_time > 0.0 and not self.carrying:
            self.drop_hold_time = max(0.0, self.drop_hold_time - dt)

        # 엣지 진입 쿨다운 감소 (노드 도착 직후 즉시 재진입 방지)
        if self.edge_enter_cooldown > 0.0:
            self.edge_enter_cooldown = max(0.0, self.edge_enter_cooldown - dt)

        # 앞 OHT 충돌 방지: 차단 중이면 속도를 0으로 설정
        is_blocked = self._is_blocked_by_front_oht()
        current_speed = 0.0 if is_blocked else self.move_speed

        # ── 상태 머신 ────────────────────────────────────────────────────────

        if self.state == "LOWER_PICK":
            # 호이스트를 픽업 위치(하단)까지 내린다
            self.target_hoist_offset = self.hoist_pick_offset
            if abs(self.current_hoist_offset - self.hoist_pick_offset) < 0.03 and self._head_matches_station("LOAD"):
                self.timer = 0.0
                self.state = "PICK_WAIT"   # 픽업 위치 도달 → 대기 상태로 전환

        elif self.state == "PICK_WAIT":
            # 0.5초 대기 후 FOUP을 적재하고 호이스트를 올린다
            self.timer += dt
            if self.timer >= 0.5:
                self.carrying = True                        # FOUP 적재
                self.target_hoist_offset = self.hoist_top_offset
                self.state = "RAISE_AFTER_PICK"

        elif self.state == "RAISE_AFTER_PICK":
            # 호이스트를 수납 위치(상단)까지 올린다
            self.target_hoist_offset = self.hoist_top_offset
            if abs(self.current_hoist_offset - self.hoist_top_offset) < 0.03:
                # 호이스트 수납 완료 → UNLOAD 목적지로 경로 재계획 후 이동 재개
                self.target_station = "UNLOAD"
                self._ensure_route()
                self.edge_enter_cooldown = 0.05
                self.state = "WAIT_AT_NODE"

        elif self.state == "LOWER_DROP":
            # 호이스트를 드롭 위치(하단)까지 내린다
            self.target_hoist_offset = self.hoist_pick_offset
            if abs(self.current_hoist_offset - self.hoist_pick_offset) < 0.03 and self._head_matches_station("UNLOAD"):
                self.timer = 0.0
                self.state = "DROP_WAIT"   # 드롭 위치 도달 → 대기 상태로 전환

        elif self.state == "DROP_WAIT":
            # 0.5초 대기 후 FOUP을 내려놓고 호이스트를 올린다
            self.timer += dt
            if self.timer >= 0.5:
                self.carrying = False                       # FOUP 해제
                self.drop_hold_time = 1.3                  # 드롭 시각 1.3초 유지
                self.target_hoist_offset = self.hoist_top_offset
                self.state = "RAISE_AFTER_DROP"

        elif self.state == "RAISE_AFTER_DROP":
            # 호이스트를 수납 위치(상단)까지 올린다
            self.target_hoist_offset = self.hoist_top_offset
            if abs(self.current_hoist_offset - self.hoist_top_offset) < 0.03:
                # 호이스트 수납 완료 → LOAD 목적지로 경로 재계획 후 이동 재개
                self.target_station = "LOAD"
                self._ensure_route()
                self.edge_enter_cooldown = 0.05
                self.state = "WAIT_AT_NODE"

        else:
            # 이동 중 또는 노드 대기 상태: 호이스트는 수납 위치 유지
            self.target_hoist_offset = self.hoist_top_offset

            if self.current_edge is not None:
                # ── 엣지 이동 처리 ──────────────────────────────────────────
                edge = self.graph.edges[self.current_edge]
                delta_t = (current_speed * dt) / max(edge.length, 1e-6)  # t 증분 계산
                self.edge_t += delta_t

                if self.edge_t >= 1.0:
                    # 엣지 끝 도달: 노드로 전환하고 route_index 증가
                    self.edge_t = 1.0
                    finished_edge = self.current_edge
                    self.controller.leave_edge(self.id, finished_edge)   # 컨트롤러에 이탈 알림
                    self.current_node = self.graph.edges[finished_edge].end
                    self.current_edge = None
                    self.route_index += 1
                    self.edge_enter_cooldown = 0.05
                    self.state = "WAIT_AT_NODE"
            else:
                # ── 노드 정차 처리 ──────────────────────────────────────────
                if not is_blocked:
                    if self.current_node == "LOAD" and not self.carrying and self.drop_hold_time <= 0.0:
                        # LOAD 스테이션 도착 & 빈 상태 → 픽업 시작
                        self.state = "LOWER_PICK"
                    elif self.current_node == "UNLOAD" and self.carrying:
                        # UNLOAD 스테이션 도착 & 적재 상태 → 드롭 시작
                        self.state = "LOWER_DROP"
                    else:
                        # 그 외: 쿨다운이 끝나면 다음 엣지로 진입 시도
                        if self.edge_enter_cooldown <= 0.0:
                            self._try_enter_next_edge()

        self._update_visuals(dt)   # 모든 상태 처리 후 시각 갱신


# =============================================================================
# USD prim 연동
# =============================================================================

def get_world_translation_from_prim(prim: Usd.Prim) -> Optional[np.ndarray]:
    """
    [기능] USD Prim의 월드 좌표계 기준 절대 위치(Translation)를 반환한다.
    [입력] prim : USD Prim 객체 (Usd.Prim)
    [출력] 3D 위치 (np.ndarray shape=(3,)) 또는 prim이 유효하지 않으면 None
    """
    if prim is None or not prim.IsValid():
        return None

    # omni.usd 유틸로 월드 변환 행렬 획득 후 Translation 추출
    m = omni.usd.get_world_transform_matrix(prim)
    t = m.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def iter_stage_paths(stage: Usd.Stage):
    """
    [기능] Stage의 모든 Prim을 순회하며 (prim path 문자열, Prim 객체) 쌍을 생성(yield)한다.
    [입력] stage : Usd.Stage 객체
    [출력] Generator of (str, Usd.Prim) 튜플
    """
    for prim in stage.Traverse():
        yield str(prim.GetPath()), prim


def find_prim_by_exact_or_keywords(stage: Usd.Stage, exact_path: Optional[str], keywords: Tuple[str, ...]) -> Optional[Usd.Prim]:
    """
    [기능] 정확한 prim path 또는 키워드 기반 점수 탐색으로 Prim을 찾는다.
           exact_path가 있으면 우선 시도하고, 없으면 모든 prim path를 순회하며
           keywords 일치 점수가 가장 높은 prim을 반환한다.
    [입력]
        stage      : Usd.Stage 객체
        exact_path : 탐색할 정확한 USD path 문자열 (없으면 None)
        keywords   : 탐색에 사용할 키워드 튜플 (Tuple[str, ...])
    [출력] 찾은 Usd.Prim 객체, 없으면 None
    """
    # 정확한 경로가 있으면 먼저 시도
    if exact_path:
        prim = stage.GetPrimAtPath(exact_path)
        if prim and prim.IsValid():
            return prim

    # 키워드 점수 기반 탐색: (점수, 경로 길이, prim) 리스트 구성
    candidates = []
    for path_str, prim in iter_stage_paths(stage):
        low = path_str.lower()
        score = 0
        for kw in keywords:
            if kw.lower() in low:
                score += len(kw)       # 긴 키워드일수록 가산점 부여
        if score > 0:
            candidates.append((score, len(path_str), prim))

    if not candidates:
        return None

    # 점수 내림차순, 경로 길이 오름차순으로 정렬 후 최우선 후보 반환
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


def maybe_print_stage_paths(stage: Usd.Stage, enabled: bool):
    """
    [기능] enabled=True이면 Stage의 모든 prim path를 콘솔에 출력한다. (디버그용)
    [입력]
        stage   : Usd.Stage 객체
        enabled : 출력 여부 (bool)
    [출력] 콘솔 출력 (없으면 아무것도 하지 않음)
    """
    if not enabled:
        return

    print("\n========== Stage Prim Paths ==========")
    for path_str, _ in iter_stage_paths(stage):
        print(path_str)
    print("======================================\n")


# =============================================================================
# 디버그 마커 / 스테이션 시각화
# =============================================================================

def add_debug_anchor(world: World, prim_path: str, name: str, pos: np.ndarray, color: np.ndarray):
    """
    [기능] 앵커 위치에 작은 색깔 Cuboid를 추가하여 탐지된 위치를 시각적으로 표시한다.
    [입력]
        world     : Isaac Sim World 객체
        prim_path : Stage에 생성할 Cuboid prim path (str)
        name      : Cuboid 이름 (str)
        pos       : 3D 위치 (np.ndarray shape=(3,))
        color     : RGB 색상 (np.ndarray shape=(3,))
    [출력] 없음 (Stage에 FixedCuboid 추가)
    """
    world.scene.add(
        FixedCuboid(
            prim_path=prim_path,
            name=name,
            position=pos,
            scale=np.array([0.30, 0.30, 0.30]),
            color=color,
        )
    )


def add_station_io_visuals(world: World, load_anchor: np.ndarray, unload_anchor: np.ndarray, graph: RailGraphSystem, visualize_anchors: bool = True):
    """
    [기능] LOAD/UNLOAD 스테이션 위치에 패드(바닥 표시용 Cuboid)와
           앵커 마커(탐지 위치 확인용 Cuboid)를 Stage에 추가한다.
    [입력]
        world             : Isaac Sim World 객체
        load_anchor       : 컨베이어 앵커 3D 위치 (np.ndarray)
        unload_anchor     : 로봇 앵커 3D 위치 (np.ndarray)
        graph             : RailGraphSystem 객체 (LOAD/UNLOAD 노드 좌표 참조)
        visualize_anchors : True면 앵커 마커 Cuboid 추가 (bool)
    [출력] 없음 (Stage에 FixedCuboid 오브젝트 추가)
    """
    # LOAD 스테이션 바닥 패드 (녹색)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/OHTInfra/Stations/LoadPad",
            name="load_pad",
            position=np.array([graph.nodes["LOAD"].pos[0], graph.nodes["LOAD"].pos[1], 0.95]),
            scale=np.array([0.95, 0.95, 0.20]),
            color=np.array([0.18, 0.40, 0.18]),
        )
    )

    # UNLOAD 스테이션 바닥 패드 (적색)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/OHTInfra/Stations/UnloadPad",
            name="unload_pad",
            position=np.array([graph.nodes["UNLOAD"].pos[0], graph.nodes["UNLOAD"].pos[1], 0.95]),
            scale=np.array([0.95, 0.95, 0.20]),
            color=np.array([0.40, 0.18, 0.18]),
        )
    )

    if visualize_anchors:
        # 컨베이어 앵커 마커 (연두색)
        add_debug_anchor(
            world,
            "/World/OHTInfra/Stations/LoadAnchorMarker",
            "load_anchor_marker",
            np.array([load_anchor[0], load_anchor[1], max(0.2, load_anchor[2])]),
            np.array([0.1, 0.7, 0.2]),
        )
        # 로봇 앵커 마커 (적색)
        add_debug_anchor(
            world,
            "/World/OHTInfra/Stations/UnloadAnchorMarker",
            "unload_anchor_marker",
            np.array([unload_anchor[0], unload_anchor[1], max(0.2, unload_anchor[2])]),
            np.array([0.7, 0.2, 0.2]),
        )


# =============================================================================
# 루프 레일 생성
# =============================================================================

def build_semicircle_points(center_xy: np.ndarray, radius: float, start_deg: float, end_deg: float, segments: int, z: float):
    """
    [기능] 반원(또는 임의 호) 위의 points를 segments 등분하여 3D 좌표 리스트로 반환한다.
           레일 루프의 좌우 턴 구간을 부드럽게 표현하는 데 사용된다.
    [입력]
        center_xy : 원 중심 XY (np.ndarray shape=(2,))
        radius    : 반지름 (m, float)
        start_deg : 시작 각도 (도, float)
        end_deg   : 끝 각도 (도, float)
        segments  : 분할 수 (int)
        z         : 레일 높이 Z (float)
    [출력] 3D 좌표 리스트 (List[np.ndarray], 각 원소 shape=(3,))
    """
    pts = []
    for i in range(1, segments + 1):
        # 선형 보간으로 각도 계산 후 XY 좌표 생성
        a = math.radians(start_deg + (end_deg - start_deg) * (i / segments))
        pts.append(
            np.array(
                [
                    center_xy[0] + radius * math.cos(a),
                    center_xy[1] + radius * math.sin(a),
                    z,
                ],
                dtype=float,
            )
        )
    return pts


def build_loop_graph_from_anchors(load_anchor: np.ndarray, unload_anchor: np.ndarray, cfg: IntegrationConfig) -> RailGraphSystem:
    """
    [기능] 컨베이어 앵커(LOAD)와 로봇 앵커(UNLOAD)의 3D 좌표를 기반으로
           타원형 폐루프 레일 그래프를 생성하고 반환한다.
           상단 직선(LOAD→UNLOAD 방향), 우측 반원 턴, 하단 복귀 직선, 좌측 반원 턴으로 구성된다.
    [입력]
        load_anchor   : 컨베이어 서비스 포인트 3D 좌표 (np.ndarray)
        unload_anchor : 로봇 서비스 포인트 3D 좌표 (np.ndarray)
        cfg           : IntegrationConfig 설정 객체
    [출력] 완성된 RailGraphSystem 객체 (노드, 엣지, 루프 순서 확정)
    """
    # LOAD~UNLOAD 거리가 너무 짧으면 최소 루프 길이 확보
    dx = unload_anchor[0] - load_anchor[0]
    if abs(dx) < cfg.min_loop_length:
        unload_anchor = unload_anchor.copy()
        unload_anchor[0] = load_anchor[0] + cfg.min_loop_length

    # 두 앵커 중 높은 쪽에 레일 높이를 더해 레일 Z 좌표 결정
    rail_z = max(load_anchor[2], unload_anchor[2]) + cfg.rail_height_above_anchor
    graph = RailGraphSystem(rail_z=rail_z)

    # 상단 레일(메인 라인)의 Y 좌표: 두 앵커의 Y 평균
    load = np.array([load_anchor[0], load_anchor[1], rail_z], dtype=float)
    unload = np.array([unload_anchor[0], unload_anchor[1], rail_z], dtype=float)

    # 루프 좌우 경계 X 좌표
    x_left = min(load[0], unload[0]) - cfg.left_margin
    x_right = max(load[0], unload[0]) + cfg.right_margin

    # 상단/하단 레일 Y 좌표
    y_top = 0.5 * (load[1] + unload[1])
    y_bottom = y_top + cfg.rail_return_y_offset

    # y_bottom이 너무 y_top에 가까우면 강제로 4m 아래로 설정
    if y_bottom > y_top - 2.5:
        y_bottom = y_top - 4.0

    # 반원 반지름: 상하 간격의 절반 (최소 1.4m 보장)
    radius = max(1.4, 0.5 * abs(y_top - y_bottom))

    # ── 노드 등록 ────────────────────────────────────────────────────────────
    graph.add_node("LOAD",      np.array([load[0],   y_top,    rail_z]), kind="station")
    graph.add_node("UNLOAD",    np.array([unload[0], y_top,    rail_z]), kind="station")
    graph.add_node("TOP_LEFT",  np.array([x_left,    y_top,    rail_z]))
    graph.add_node("TOP_RIGHT", np.array([x_right,   y_top,    rail_z]))
    graph.add_node("BOT_LEFT",  np.array([x_left,    y_bottom, rail_z]))
    graph.add_node("BOT_RIGHT", np.array([x_right,   y_bottom, rail_z]))

    # 상단 레일 중간 보조 노드 (3등분점)
    top_mid1 = np.array([0.5 * (x_left  + load[0]),   y_top,    rail_z])
    top_mid2 = np.array([0.5 * (load[0] + unload[0]), y_top,    rail_z])
    top_mid3 = np.array([0.5 * (unload[0] + x_right), y_top,    rail_z])

    # 하단 레일 중간 보조 노드 (역방향 3등분점)
    bot_mid1 = np.array([0.5 * (x_right  + unload[0]), y_bottom, rail_z])
    bot_mid2 = np.array([0.5 * (unload[0] + load[0]),  y_bottom, rail_z])
    bot_mid3 = np.array([0.5 * (load[0]  + x_left),    y_bottom, rail_z])

    graph.add_node("TOP_M1", top_mid1)
    graph.add_node("TOP_M2", top_mid2)
    graph.add_node("TOP_M3", top_mid3)
    graph.add_node("BOT_M1", bot_mid1)
    graph.add_node("BOT_M2", bot_mid2)
    graph.add_node("BOT_M3", bot_mid3)

    # 좌우 반원 턴의 중심 좌표
    right_center = np.array([x_right, 0.5 * (y_top + y_bottom)], dtype=float)
    left_center  = np.array([x_left,  0.5 * (y_top + y_bottom)], dtype=float)

    # 우측 반원: 위(90°)→아래(-90°), 시계방향
    right_pts = build_semicircle_points(right_center, radius,  90.0, -90.0, cfg.turn_segments, rail_z)
    # 좌측 반원: 아래(-90°)→위(90°), 반시계방향
    left_pts  = build_semicircle_points(left_center,  radius, -90.0,  90.0, cfg.turn_segments, rail_z)

    edge_order = []   # 루프를 구성하는 엣지 순서 기록

    # ── 상단 레일 엣지 등록 (TOP_LEFT → LOAD → UNLOAD → TOP_RIGHT) ─────────
    top_sequence = ["TOP_LEFT", "TOP_M1", "LOAD", "TOP_M2", "UNLOAD", "TOP_M3", "TOP_RIGHT"]
    for a, b in zip(top_sequence[:-1], top_sequence[1:]):
        en = f"E_{a}_TO_{b}"
        graph.add_edge(en, a, b)
        edge_order.append(en)

    # ── 우측 반원 턴 엣지 등록 ─────────────────────────────────────────────
    prev_name = "TOP_RIGHT"
    for i, p in enumerate(right_pts, start=1):
        name = f"R{i}"
        graph.add_node(name, p)
        en = f"E_{prev_name}_TO_{name}"
        graph.add_edge(en, prev_name, name)
        edge_order.append(en)
        prev_name = name

    # 반원 마지막 점 → BOT_RIGHT 연결
    en = f"E_{prev_name}_TO_BOT_RIGHT"
    graph.add_edge(en, prev_name, "BOT_RIGHT")
    edge_order.append(en)

    # ── 하단 복귀 레일 엣지 등록 (BOT_RIGHT → BOT_LEFT) ───────────────────
    bottom_sequence = ["BOT_RIGHT", "BOT_M1", "BOT_M2", "BOT_M3", "BOT_LEFT"]
    for a, b in zip(bottom_sequence[:-1], bottom_sequence[1:]):
        en = f"E_{a}_TO_{b}"
        graph.add_edge(en, a, b)
        edge_order.append(en)

    # ── 좌측 반원 턴 엣지 등록 ─────────────────────────────────────────────
    prev_name = "BOT_LEFT"
    for i, p in enumerate(left_pts, start=1):
        name = f"L{i}"
        graph.add_node(name, p)
        en = f"E_{prev_name}_TO_{name}"
        graph.add_edge(en, prev_name, name)
        edge_order.append(en)
        prev_name = name

    # 반원 마지막 점 → TOP_LEFT 연결 (루프 폐쇄)
    en = f"E_{prev_name}_TO_TOP_LEFT"
    graph.add_edge(en, prev_name, "TOP_LEFT")
    edge_order.append(en)

    # 루프 확정: 엣지 순서 및 누적 거리 계산
    graph.finalize_loop(edge_order)
    return graph


# =============================================================================
# 맵 연동: 컨베이어/로봇 prim -> 서비스 앵커 추론
# =============================================================================

def infer_service_anchors(stage: Usd.Stage, cfg: IntegrationConfig):
    """
    [기능] Stage에서 컨베이어/로봇 prim을 탐색하고,
           오프셋 보정을 적용하여 LOAD/UNLOAD 서비스 앵커 좌표를 반환한다.
           prim을 찾지 못하면 cfg.use_box_fallback_if_anchor_missing에 따라
           기본값 사용 또는 예외 발생.
    [입력]
        stage : Usd.Stage 객체
        cfg   : IntegrationConfig 설정 객체
    [출력] 딕셔너리 {
        "conveyor_prim" : Usd.Prim 또는 None,
        "robot_prim"    : Usd.Prim 또는 None,
        "load_anchor"   : np.ndarray (컨베이어 서비스 포인트 좌표),
        "unload_anchor" : np.ndarray (로봇 서비스 포인트 좌표)
    }
    """
    # 컨베이어/로봇 prim 탐색
    conveyor_prim = find_prim_by_exact_or_keywords(stage, cfg.conveyor_anchor_path, cfg.conveyor_prim_candidates)
    robot_prim    = find_prim_by_exact_or_keywords(stage, cfg.robot_anchor_path,    cfg.robot_prim_candidates)

    conveyor_pos = get_world_translation_from_prim(conveyor_prim) if conveyor_prim else None
    robot_pos    = get_world_translation_from_prim(robot_prim)    if robot_prim    else None

    # LOAD 앵커: 컨베이어 위치 + 오프셋, 없으면 기본값 또는 예외
    if conveyor_pos is not None:
        load_anchor = conveyor_pos + cfg.load_xy_offset_from_conveyor
    elif cfg.use_box_fallback_if_anchor_missing:
        load_anchor = np.array([2.0, 0.0, 1.0], dtype=float)   # 기본 폴백 좌표
    else:
        raise RuntimeError("Conveyor prim을 찾지 못했습니다. conveyor_anchor_path를 지정하세요.")

    # UNLOAD 앵커: 로봇 위치 + 오프셋, 없으면 기본값 또는 예외
    if robot_pos is not None:
        unload_anchor = robot_pos + cfg.unload_xy_offset_from_robot
    elif cfg.use_box_fallback_if_anchor_missing:
        unload_anchor = np.array([14.0, 0.0, 1.0], dtype=float)  # 기본 폴백 좌표
    else:
        raise RuntimeError("Robot prim을 찾지 못했습니다. robot_anchor_path를 지정하세요.")

    return {
        "conveyor_prim": conveyor_prim,
        "robot_prim":    robot_prim,
        "load_anchor":   load_anchor,
        "unload_anchor": unload_anchor,
    }


# =============================================================================
# Stage / World 준비
# =============================================================================

def prepare_stage_and_world(cfg: IntegrationConfig) -> Tuple[Usd.Stage, World]:
    """
    [기능] Isaac Sim Stage와 World를 준비한다.
           cfg.map_usd_path가 지정된 경우 Stage를 초기화하고 USD를 레퍼런스로 불러온다.
           기존 World 인스턴스가 있으면 초기화 후 새로 생성한다.
    [입력] cfg : IntegrationConfig 설정 객체
    [출력] (Usd.Stage, World) 튜플
    """
    stage = omni.usd.get_context().get_stage()  # 현재 활성 Stage 획득

    # map_usd_path가 지정된 경우: 기존 /World prim 삭제 후 USD 레퍼런스 로드
    if cfg.map_usd_path:
        if stage.GetPrimAtPath("/World").IsValid():
            omni.kit.commands.execute("DeletePrims", paths=["/World"])

        omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Xform", prim_path="/World")
        omni.kit.commands.execute(
            "CreateReferenceCommand",
            path_to=Sdf.Path("/World/UserMap"),
            asset_path=cfg.map_usd_path,
            instanceable=False,
        )
        stage = omni.usd.get_context().get_stage()  # 레퍼런스 로드 후 Stage 재획득

    # 기존 World 인스턴스가 있으면 초기화
    if World.instance() is not None:
        World.instance().clear_instance()

    world = World()  # 새 World 생성

    # /World prim이 없으면 새로 생성
    if not stage.GetPrimAtPath("/World").IsValid():
        omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_type="Xform", prim_path="/World")

    return stage, world


# =============================================================================
# 통합 빌더
# =============================================================================

def build_integrated_oht_scene(cfg: IntegrationConfig):
    """
    [기능] IntegrationConfig를 받아 전체 OHT 시뮬레이션 씬을 구성한다.
           Stage 준비 → 앵커 추론 → 레일 그래프 생성 → 시각화 → OHT 생성 순으로 진행한다.
    [입력] cfg : IntegrationConfig 설정 객체
    [출력] (World, RailGraphSystem, CentralController, List[GraphOHT], dict) 튜플
        · World              : 물리 시뮬레이션 컨텍스트
        · RailGraphSystem    : 레일 그래프
        · CentralController  : 교통 제어기
        · List[GraphOHT]     : 생성된 OHT 차량 리스트
        · dict               : 탐지된 prim 및 앵커 좌표 딕셔너리
    """
    stage, world = prepare_stage_and_world(cfg)            # Stage/World 초기화
    maybe_print_stage_paths(stage, cfg.auto_print_stage_paths)  # 디버그: prim path 출력

    # 컨베이어/로봇 위치에서 LOAD/UNLOAD 앵커 좌표 추론
    anchors = infer_service_anchors(stage, cfg)
    load_anchor   = anchors["load_anchor"]
    unload_anchor = anchors["unload_anchor"]

    # 앵커 좌표를 기반으로 폐루프 레일 그래프 생성
    graph = build_loop_graph_from_anchors(load_anchor, unload_anchor, cfg)
    graph.build_rail_visuals(world, samples_per_edge=8)    # 레일 Cuboid 시각화

    # 스테이션 패드 및 앵커 마커 추가
    add_station_io_visuals(
        world,
        load_anchor,
        unload_anchor,
        graph,
        visualize_anchors=cfg.visualize_inferred_anchors,
    )

    controller = CentralController(graph)   # 중앙 컨트롤러 초기화

    # OHT 시작 위치 노드 지정
    start_a = "TOP_LEFT"
    start_b = "BOT_M2"

    # 첫 번째 OHT 생성 (황색 차체, 청색 FOUP)
    oht_a = GraphOHT(
        world=world,
        graph=graph,
        controller=controller,
        oht_id="OHT_A",
        color=np.array([1.0, 0.78, 0.10]),
        foup_color=np.array([0.10, 0.60, 0.95]),
        start_node=start_a,
        move_speed=cfg.oht_speed,
        safe_distance=cfg.oht_safe_distance,
    )

    fleet = [oht_a]

    # 두 번째 OHT 생성 (녹색 차체, 적색 FOUP)
    if cfg.spawn_two_ohts:
        oht_b = GraphOHT(
            world=world,
            graph=graph,
            controller=controller,
            oht_id="OHT_B",
            color=np.array([0.20, 0.80, 0.40]),
            foup_color=np.array([0.95, 0.30, 0.30]),
            start_node=start_b,
            move_speed=cfg.oht_speed,
            safe_distance=cfg.oht_safe_distance,
        )
        fleet.append(oht_b)

    # 모든 OHT에 fleet 리스트 공유 (상호 충돌 감지용)
    for oht in fleet:
        oht.fleet = fleet

    world.reset()   # 물리 시뮬레이션 초기화
    return world, graph, controller, fleet, anchors


# =============================================================================
# 사용자 설정
# =============================================================================
CFG = IntegrationConfig(
    # 이미 네 USD 맵을 Isaac Sim에서 열어둔 상태면 None 유지
    map_usd_path=None,

    # 권장: 아래 두 개를 네 맵의 실제 prim path로 채워라
    # conveyor_anchor_path="/World/MyFactory/Conveyor_01",
    # robot_anchor_path="/World/MyFactory/Doosan_M0609",
    conveyor_anchor_path=None,
    robot_anchor_path=None,

    # prim 원점과 실제 서비스 포인트가 안 맞으면 조정
    load_xy_offset_from_conveyor=np.array([0.0, 0.0, 0.0], dtype=float),
    unload_xy_offset_from_robot=np.array([1.0, 0.0, 0.0], dtype=float),

    # 레일 높이/복귀 레일 간격
    rail_height_above_anchor=3.6,
    rail_return_y_offset=-4.0,

    # 반원 turn 세분화
    turn_segments=8,

    # OHT
    spawn_two_ohts=True,
    oht_speed=2.2,
    oht_safe_distance=3.6,

    # 처음엔 True로 두고 콘솔에서 prim path 확인
    auto_print_stage_paths=True,
    visualize_inferred_anchors=True,
)

# 씬 전체 구성 실행 → 전역 변수에 저장
_WORLD, _GRAPH, _CTRL, _FLEET, _ANCHORS = build_integrated_oht_scene(CFG)

# 탐지 결과 콘솔 출력
print("Conveyor prim:", _ANCHORS["conveyor_prim"].GetPath() if _ANCHORS["conveyor_prim"] else "NOT FOUND")
print("Robot prim   :", _ANCHORS["robot_prim"].GetPath()    if _ANCHORS["robot_prim"]    else "NOT FOUND")
print("Load anchor  :", _ANCHORS["load_anchor"])
print("Unload anchor:", _ANCHORS["unload_anchor"])


# =============================================================================
# Physics step 콜백
# =============================================================================
global _graph_oht_subscription   # 구독 핸들을 전역으로 보관하여 GC로 인한 해제 방지

def on_physics_step(step_size):
    """
    [기능] Isaac Sim 물리 엔진의 매 스텝마다 호출되어 모든 OHT의 상태를 갱신한다.
    [입력] step_size : 물리 스텝 크기 (초, float). 최소 1/120초 하한 적용.
    [출력] 없음 (각 OHT의 update() 호출로 위치/상태 갱신)
    """
    dt = max(1.0 / 120.0, float(step_size))  # 너무 작은 dt로 인한 수치 불안정 방지
    for oht in _FLEET:
        oht.update(dt)

# 물리 스텝 이벤트에 콜백 등록 (시뮬레이션 실행 중 자동 호출됨)
_graph_oht_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(on_physics_step)